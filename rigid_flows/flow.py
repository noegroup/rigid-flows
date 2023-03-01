from dataclasses import asdict
from functools import partial
from typing import Any, cast

import equinox
import equinox as eqx
import jax
import lenses
from flox import geom
from flox._src.flow.impl import Affine, Moebius
from flox.flow import DoubleMoebius, Inverted, Pipe, Transform, Transformed, bind, pure
from flox.util import key_chain, unpack
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Float

from .data import DataWithAuxiliary
from .nnextra import AuxConditioner, PosConditioner, RotConditioner
from .rigid import Rigid
from .specs import CouplingSpecification, FlowSpecification
from .system import QUATERNION_DIM, SPATIAL_DIM, SimulationBox

KeyArray = jnp.ndarray | jax.random.PRNGKeyArray
Atoms = Float[Array, "... MOL SITES SPATIAL_DIM"]
PosArray = Float[Array, "... SPATIAL_DIM"]
QuatArray = Float[Array, "... QUATERNION_DIM"]

MOEBIUS_SLACK = 0.95
IDENTITY_GATE = 3.0


def PosEncoder(pos: PosArray, box: SimulationBox):
    rad = pos / box.size
    enc_pos = jnp.stack(
        [
            jnp.cos(2 * jnp.pi * rad - jnp.pi),
            jnp.sin(2 * jnp.pi * rad - jnp.pi),
        ],
        axis=-1,
    )
    return enc_pos


def PosDecoder(enc_pos, box: SimulationBox) -> PosArray:
    pos = jnp.arctan2(enc_pos[..., 1], enc_pos[..., 0])
    pos = (pos + jnp.pi) / (2 * jnp.pi)
    return pos * box.size


def QuatEncoder(rot: QuatArray):
    """Encodes a quaternion into a flip-invariant representation."""
    # rot = rot * jnp.sign(rot[..., :1])
    return rot


class RigidTransform(Transform[Atoms, Rigid]):
    def forward(self, inp: Atoms) -> Transformed[Rigid]:
        rigid = Rigid.from_array(inp)
        return Transformed(rigid, jnp.zeros(()))

    def inverse(self, inp: Rigid) -> Transformed[Atoms]:
        atom_rep = inp.asarray()
        return Transformed(atom_rep, jnp.zeros(()))


@pytree_dataclass(frozen=True)
class LayerStackedPipe(Pipe):

    use_scan: bool

    def scan(self, input: Any, inverse: bool) -> Transformed[Any]:
        params, static = eqx.partition(self.transforms, eqx.is_array)  # type: ignore
        params = jax.tree_map(
            lambda *args: jnp.stack(args),
            *params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )

        def body(x, param):
            fn: Transform = cast(Transform, eqx.combine(param, static[0]))
            if inverse:
                fn = Inverted(fn)
            x = bind(x, fn)
            return x, None

        out, _ = jax.lax.scan(body, pure(input), params, reverse=inverse)
        return out

    def forward(self, input: Array):
        if self.use_scan:
            return self.scan(input, inverse=False)
        else:
            return super().forward(input)

    def inverse(self, input: Array):
        if self.use_scan:
            return self.scan(input, inverse=True)
        else:
            return super().forward(input)


@pytree_dataclass
class RigidWithAuxiliary:
    """
    State being passed through the flow.

    Args:
        rot: quaternion describing current rotation
        pos: 3D vector describing current position
        ics: internal DoF of the system
        aux: auxiliary state
        box: simulation box
    """

    rigid: Rigid
    aux: Array | None
    box: SimulationBox


class QuatUpdate(eqx.Module):
    """Flow layer updating the quaternion part of a state"""

    net: RotConditioner
    # net: eqx.nn.MLP

    def __init__(
        self,
        use_auxiliary: bool,
        seq_len: int,
        num_blocks: int,
        num_heads: int,
        num_channels: int,
        key: KeyArray,
    ):
        """Flow layer updating the quaternion part of a state."""
        chain = key_chain(key)

        num_out = QUATERNION_DIM
        if use_auxiliary:
            num_aux = SPATIAL_DIM
        else:
            num_aux = None
        #     num_aux = 0
        # self.net = eqx.nn.MLP(
        #     seq_len * (num_aux +  2 * SPATIAL_DIM),
        #     seq_len * 2 * num_out,
        #     width_size=128,
        #     depth=2,
        #     key=next(chain),
        # )
        self.net = RotConditioner(
            seq_len,
            2 * num_out,
            num_aux,
            num_heads,
            num_channels,
            num_blocks,
            key=next(chain),
        )

    def params(self, input: RigidWithAuxiliary):
        """Compute the parameters for the double moebius transform

        Args:
            input (State): current state

        Returns:
            Array: the parameter (reflection) of the double moebius transform
        """
        enc_pos = PosEncoder(input.rigid.pos, input.box).reshape(
            (*input.rigid.pos.shape[:-1], 2 * SPATIAL_DIM)
        )

        # out = self.net(enc_pos.flatten())
        out = self.net(enc_pos, input.aux)

        reflection, gate = jnp.split(out, 2, axis=-1)

        reflection = reflection * jax.nn.sigmoid(gate - IDENTITY_GATE)

        reflection = reflection.reshape(input.rigid.rot.shape)
        reflection = jax.vmap(lambda x: x / (1 + geom.norm(x)) * MOEBIUS_SLACK)(
            reflection
        )

        return reflection

    def forward(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Forward transform"""
        reflection = self.params(input)

        def trasfo(ref, rot):
            return DoubleMoebius(ref).forward(rot)

        rot, ldj = unpack(jax.vmap(trasfo)(reflection, input.rigid.rot))
        output = lenses.bind(input).rigid.rot.set(rot)
        ldj = jnp.sum(ldj)
        return Transformed(output, ldj)

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Inverse transform"""
        reflection = self.params(input)

        def trasfo(ref, rot) -> Transformed[Array]:
            return DoubleMoebius(ref).inverse(rot)

        rot, ldj = unpack(jax.vmap(trasfo)(reflection, input.rigid.rot))
        output = lenses.bind(input).rigid.rot.set(rot)
        ldj = jnp.sum(ldj)
        return Transformed(output, ldj)


class AuxUpdate(eqx.Module):
    """Flow layer updating the auxiliary part of a state"""

    net: AuxConditioner

    def __init__(
        self,
        seq_len: int,
        num_blocks: int,
        num_heads: int,
        num_channels: int,
        key: KeyArray,
    ):
        """Flow layer updating the auxiliary part of a state."""
        chain = key_chain(key)
        num_aux = SPATIAL_DIM
        num_out = 2 * num_aux
        self.net = AuxConditioner(
            seq_len,
            2 * num_out,
            num_heads,
            num_channels,
            num_blocks,
            key=next(chain),
        )

    def params(self, input: RigidWithAuxiliary):
        """Compute the parameters for the affine transform

        Args:
            input (State): current state

        Returns:
            tuple[Array, Array]: the parameters (shift, scale) of the affine transform
        """
        enc_pos = PosEncoder(input.rigid.pos, input.box).reshape(
            (*input.rigid.pos.shape[:-1], 2 * SPATIAL_DIM)
        )
        out = self.net(enc_pos, QuatEncoder(input.rigid.rot))
        out = out.reshape(input.aux.shape[0], -1)
        out, gate = jnp.split(out, 2, axis=-1)
        out = out * jax.nn.sigmoid(gate - IDENTITY_GATE)

        shift, scale = jnp.split(out, 2, axis=-1)  # type: ignore
        return shift, scale

    def forward(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Forward transform"""
        shift, scale = self.params(input)
        pipe = Affine(shift, scale)
        aux, ldj = unpack(pipe.forward(input.aux))
        return Transformed(lenses.bind(input).aux.set(aux), ldj)

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Inverse transform"""
        shift, scale = self.params(input)
        pipe = Affine(shift, scale)
        aux, ldj = unpack(pipe.inverse(input.aux))
        return Transformed(lenses.bind(input).aux.set(aux), ldj)


class PosUpdate(eqx.Module):

    net: PosConditioner
    # net: eqx.nn.MLP

    def __init__(
        self,
        use_auxiliary: bool,
        seq_len: int,
        num_blocks: int,
        num_heads: int,
        num_channels: int,
        key: KeyArray,
    ):
        chain = key_chain(key)

        num_out = SPATIAL_DIM * 2
        if use_auxiliary:
            num_aux = SPATIAL_DIM
        else:
            num_aux = None
        #     num_aux = 0

        # self.net = eqx.nn.MLP(
        #     in_size=seq_len * (num_aux + QUATERNION_DIM),
        #     out_size=seq_len * 2 * num_out,
        #     width_size=128,
        #     depth=2,
        #     key=next(chain),
        # )

        self.net = PosConditioner(
            seq_len,
            2 * num_out,
            num_aux,
            num_heads,
            num_channels,
            num_blocks,
            key=next(chain),
        )

    def params(self, input: RigidWithAuxiliary):
        # out = self.net(QuatEncoder(input.rigid.rot).flatten())
        out = self.net(input.aux, QuatEncoder(input.rigid.rot))
        out = out.reshape(*input.rigid.pos.shape, -1)

        reflection, gate = jnp.split(out, 2, axis=-1)
        reflection = reflection * jax.nn.sigmoid(gate - 2 * IDENTITY_GATE)
        reflection = reflection.reshape(*input.rigid.pos.shape, 2)

        reflection = reflection[1:]  # quick fix to not move the first molecule

        reflection = jax.vmap(
            jax.vmap(lambda x: x / (1 + geom.norm(x)) * MOEBIUS_SLACK)
        )(reflection)
        return reflection

    def forward(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Forward transform"""
        reflection = self.params(input)

        @jax.vmap
        def trasfo(ref, enc_pos):
            return Moebius(ref).forward(enc_pos)

        enc_pos = PosEncoder(input.rigid.pos[1:], input.box)
        enc_pos, ldj = unpack(jax.vmap(trasfo)(reflection, enc_pos))
        pos = PosDecoder(enc_pos, input.box)
        pos = jnp.concatenate([input.rigid.pos[:1], pos])

        ldj = jnp.sum(ldj)
        output = lenses.bind(input).rigid.pos.set(pos)
        return Transformed(output, ldj)

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Inverse transform"""

        reflection = self.params(input)

        @jax.vmap
        def trasfo(ref, enc_pos):
            return Moebius(ref).inverse(enc_pos)

        enc_pos = PosEncoder(input.rigid.pos[1:], input.box)
        enc_pos, ldj = unpack(jax.vmap(trasfo)(reflection, enc_pos))
        pos = PosDecoder(enc_pos, input.box)
        pos = jnp.concatenate([input.rigid.pos[:1], pos])

        ldj = jnp.sum(ldj)
        output = lenses.bind(input).rigid.pos.set(pos)
        return Transformed(output, ldj)


class EuclideanToRigidTransform(equinox.Module):
    def forward(self, input: DataWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        transform = RigidTransform()
        rigid, _ = unpack(jax.vmap(transform.forward)(input.pos))
        ldj = jnp.zeros(())
        return Transformed(
            RigidWithAuxiliary(rigid, input.aux, input.box),
            ldj,
        )

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[DataWithAuxiliary]:
        transform = RigidTransform()
        pos, _ = unpack(jax.vmap(transform.inverse)(input.rigid))
        ldj = jnp.zeros(())
        sign = jnp.sign(input.rigid.rot[:, (0,)])
        return Transformed(
            DataWithAuxiliary(pos, input.aux, sign, input.box, None),
            ldj,
        )


def _coupling(
    key: KeyArray,
    num_molecules: int,
    use_auxiliary: bool,
    specs: CouplingSpecification,
) -> Transform[RigidWithAuxiliary, RigidWithAuxiliary]:
    """Creates a coupling block consisting of:

     - an update to the auxilaries
     - an update to the quaterions
     - an update to the positions

    Args:
        key (KeyArray): PRNG Key
        num_aux (int): number of auxilaries

    Returns:
        Pipe[State, State]: the coupling block
    """
    chain = key_chain(key)
    blocks = []
    for _ in range(specs.num_repetitions):
        if use_auxiliary:
            blocks.append(
                AuxUpdate(
                    seq_len=num_molecules,
                    **asdict(specs.auxiliary_update),
                    key=next(chain),
                )
            )
        blocks.append(
            PosUpdate(
                seq_len=num_molecules,
                use_auxiliary=use_auxiliary,
                **asdict(specs.position_update),
                key=next(chain),
            )
        )
        blocks.append(
            QuatUpdate(
                seq_len=num_molecules,
                use_auxiliary=use_auxiliary,
                **asdict(specs.quaternion_update),
                key=next(chain),
            )
        )
    return Pipe(blocks)


def build_flow(
    key: KeyArray,
    num_molecules: int,
    use_auxiliary: bool,
    specs: FlowSpecification,
) -> Pipe[DataWithAuxiliary, DataWithAuxiliary]:
    """Creates the final flow composed of:

     - a preprocessing transformation
     - multiple coupling blocks

    Args:
        key (KeyArray): PRNG key
        num_aux (int): number of auxilaries
        num_blocks (int, optional): number of coupling blocks. Defaults to 2.

    Returns:
        Pipe[AugmentedData, State]: the final flow
    """
    chain = key_chain(key)
    blocks = []
    for coupling in specs.couplings:
        blocks.append(_coupling(next(chain), num_molecules, use_auxiliary, coupling))

    couplings = LayerStackedPipe(blocks, use_scan=True)
    return Pipe(
        [
            EuclideanToRigidTransform(),
            couplings,
            Inverted(EuclideanToRigidTransform()),
        ]
    )
