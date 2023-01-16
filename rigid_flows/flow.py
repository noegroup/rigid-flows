from dataclasses import asdict, astuple
from functools import partial
from typing import Any, cast

import equinox
import equinox as eqx
import jax
import lenses
from flox import geom
from flox._src.flow import rigid
from flox._src.flow.impl import Affine
from flox._src.geom.euclidean import inner, norm, unit
from flox.flow import (DoubleMoebius, Pipe, Transform, Transformed,
                       VectorizedTransform)
from flox.util import key_chain, unpack
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Float

from .data import DataWithAuxiliary
from .density import OpenMMDensity
from .lowrank import LowRankFlow
from .nn import MLPMixer, QuatEncoder
from .specs import CouplingSpecification, FlowSpecification
from .system import SimulationBox

KeyArray = jnp.ndarray | jax.random.PRNGKeyArray


Scalar = Float[Array, ""] | float
Vector3 = Float[Array, "... 3"]
Quaternion = Float[Array, "... 4"]
Auxiliary = Float[Array, f"... AUX"]

AtomRepresentation = Float[Array, "... MOL 4 3"]

IDENTITY_FACTOR = 1  # 1e-2


@pytree_dataclass(frozen=True)
class InternalCoordinates:
    # see https://github.com/openmm/openmm/blob/master/wrappers/python/openmm/app/data/tip4pew.xml
    d_OH1: Scalar  # = jnp.array(0.09572)
    d_OH2: Scalar  # = jnp.array(0.09572)
    a_HOH: Scalar  # = jnp.array(1.8242182)
    d_OM: Scalar  # = jnp.array(0.0125)
    a_OM: Scalar  # = jnp.array(0.9121091)


@pytree_dataclass(frozen=True)
class RigidRepresentation:
    rot: Quaternion
    pos: Vector3
    ics: InternalCoordinates  # = InternalCoordinates()


def to_rigid(pos: AtomRepresentation) -> Transformed[RigidRepresentation]:
    q, p, d_OH1, d_OH2, a_HOH = rigid.from_euclidean(pos[:3])
    ldj = rigid.from_euclidean_log_jacobian(pos[:3])
    d_OM = geom.norm(pos[3] - p)

    r = pos - p[None]
    a_OM = jnp.arccos(inner(unit(r[1]), unit(r[3])))

    ldj -= jnp.log(4 * d_OM**4 + d_OM**2) / 2
    return Transformed(
        RigidRepresentation(q, p, InternalCoordinates(d_OH1, d_OH2, a_HOH, d_OM, a_OM)),
        ldj,
    )


def from_rigid(rp: RigidRepresentation) -> Transformed[AtomRepresentation]:
    r_OM = rp.ics.d_OM * jnp.array([jnp.sin(rp.ics.a_OM), 0.0, jnp.cos(rp.ics.a_OM)])
    r_OM = geom.qrot3d(rp.rot, r_OM)
    pos = rigid.to_euclidean(rp.rot, rp.pos, *astuple(rp.ics)[:3])
    ldj = rigid.to_euclidean_log_jacobian(rp.rot, rp.pos, *astuple(rp.ics)[:3])
    ldj += jnp.log(4 * rp.ics.d_OM**4 + rp.ics.d_OM**2) / 2
    pos = jnp.concatenate([pos, (pos[0] + r_OM)[None]], axis=0)
    return Transformed(pos, ldj)


class RigidTransform(Transform[AtomRepresentation, RigidRepresentation]):
    def forward(self, inp: AtomRepresentation) -> Transformed[RigidRepresentation]:
        return to_rigid(inp)

    def inverse(self, inp: RigidRepresentation) -> Transformed[AtomRepresentation]:
        return from_rigid(inp)


@pytree_dataclass(frozen=True)
class LDU:
    m: Array
    apply: str

    def forward(self, input: Array):
        n = input.shape[0]

        l = jnp.triu(self.m, 1)
        d = jnp.diag(self.m) * 1e-1
        u = jnp.tril(self.m, -1)

        m = (l + jnp.eye(n)) @ jnp.diag(jnp.exp(d)) @ (u + jnp.eye(n))

        ldj = d.sum()

        match self.apply:
            case "left":
                output = m @ input
            case "right":
                output = input @ m
            case _:
                raise ValueError(f"unknown application {self.apply}")

        return Transformed(output, ldj)

    def inverse(self, input: Array):
        n = input.shape[0]

        l = jnp.triu(self.m, 1)
        d = jnp.diag(self.m) * 1e-1
        u = jnp.tril(self.m, -1)

        m = (l + jnp.eye(n)) @ jnp.diag(jnp.exp(d)) @ (u + jnp.eye(n))
        m = jnp.linalg.inv(m)

        match self.apply:
            case "left":
                output = m @ input
            case "right":
                output = input @ m
            case _:
                raise ValueError(f"unknown application {self.apply}")

        ldj = -d.sum()
        return Transformed(output, ldj)


from flox.flow import Inverted, Transform, Transformed, bind, pure


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


def toggle_layer_stack(flow: Transform, toggle: bool) -> Transform:
    if isinstance(flow, LayerStackedPipe):
        return LayerStackedPipe(flow.transforms, use_scan=toggle)
    elif isinstance(flow, Pipe):
        return Pipe(
            tuple(map(partial(toggle_layer_stack, toggle=toggle), flow.transforms))
        )
    else:
        return flow


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

    rot: Array
    pos: Array
    ics: InternalCoordinates
    aux: Array
    box: SimulationBox


def affine_quat_fwd(q, A):
    A = jnp.eye(4) + A.reshape(4, 4)
    q_ = A @ q
    ldj = jnp.linalg.slogdet(A)[1] - 4 * jnp.log(geom.norm(q_))
    q_ = geom.unit(q_)
    return q_, ldj


def affine_quat_inv(q, A):
    A = jnp.eye(4) + A.reshape(4, 4)
    A = jnp.linalg.inv(A)
    q_ = A @ q
    ldj = jnp.linalg.slogdet(A)[1] - 4 * jnp.log(geom.norm(q_))
    q_ = geom.unit(q_)
    return q_, ldj


@pytree_dataclass
class QuaternionAffine:

    M: Array

    def forward(self, input: Array):
        new, ldj = affine_quat_fwd(input, self.M)
        return Transformed(new, ldj)

    def inverse(self, input: Array):
        new, ldj = affine_quat_inv(input, self.M)
        return Transformed(new, ldj)


def affine_forward(p, params):
    m, t = jnp.split(params, (9,), axis=0)  # type: ignore
    m = m.reshape(3, 3) + jnp.eye(3)
    p = m @ p + t
    ldj = jnp.log(jnp.abs(geom.det3x3(m)))
    return p, ldj


def affine_inverse(p, params):
    m, t = jnp.split(params, (9,), axis=0)  # type: ignore
    m = m.reshape(3, 3) + jnp.eye(3)
    p = jnp.linalg.inv(m) @ (p - t)
    ldj = -jnp.log(jnp.abs(geom.det3x3(m)))
    return p, ldj


@pytree_dataclass
class FullAffine:

    params: Array

    def forward(self, input: Array):
        p, ldj = affine_forward(input, self.params)
        return Transformed(p, ldj)

    def inverse(self, input: Array):
        p, ldj = affine_inverse(input, self.params)
        return Transformed(p, ldj)


class ActNorm(eqx.Module):

    lens: lenses.ui.UnboundLens
    mean: Array | None = None
    log_std: Array | None = None

    def forward(self, input: RigidWithAuxiliary):
        assert self.mean is not None
        assert self.log_std is not None
        val = self.lens.get()(input)
        scale = jax.nn.softplus(self.log_std) + 1e-6

        val = (val - self.mean) / scale

        ldj = -jnp.log(scale).sum()

        output = self.lens.set(val)(input)
        return Transformed(output, ldj)

    def inverse(self, input: RigidWithAuxiliary):
        assert self.mean is not None
        assert self.log_std is not None
        val = self.lens.get()(input)
        scale = jax.nn.softplus(self.log_std) + 1e-6

        val = val * scale + self.mean

        ldj = jnp.log(scale).sum()

        output = self.lens.set(val)(input)
        return Transformed(output, ldj)

    def initialize(self, batch: RigidWithAuxiliary):
        val = self.lens.get()(batch)
        std = jnp.std(val, axis=0)
        log_std = jnp.log(jnp.exp(std) - 1 + 1e-6)
        return ActNorm(
            self.lens,
            jnp.mean(val, axis=0),
            log_std,
        )


def initialize_actnorm(flow: Transform, batch: Any):

    if isinstance(flow, Pipe):
        layers = []
        for layer in flow.transforms:
            layer, batch = initialize_actnorm(layer, batch)
            layers.append(layer)
        return Pipe(layers), batch

    if isinstance(flow, ActNorm):
        flow = flow.initialize(batch)

    batch, _ = unpack(jax.vmap(flow.forward)(batch))
    return flow, batch


@pytree_dataclass(frozen=True)
class GatedShift:
    new: Array
    mix: Array
    initial_bias: Array

    def forward(self, input: Array):
        mix = jax.nn.sigmoid(self.mix + self.initial_bias)
        ldj = jax.nn.log_sigmoid(self.mix).sum()
        new = self.new * (1.0 - mix)
        output = input * mix + new
        return Transformed(output, ldj)

    def inverse(self, input: Array):
        mix = jax.nn.sigmoid(self.mix + self.initial_bias)
        ldj = -jax.nn.log_sigmoid(self.mix).sum()
        new = self.new * (1.0 - mix)
        output = (input - new) / mix
        return Transformed(output, ldj)


class QuatUpdate(eqx.Module):
    """Flow layer updating the quaternion part of a state"""

    net: MLPMixer

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int = 3,
        *,
        key: KeyArray,
        **kwargs,
    ):
        """Flow layer updating the quaternion part of a state.

        Args:
            auxiliary_shape (tuple[int, ...]): shape of auxilaries
            num_pos (int, optional): number of position DoF. Defaults to 3.
            num_rot (int, optional): number of quaternion DoF. Defaults to 4.
            num_heads (int, optional): number of transformer heads. Defaults to 4.
            num_dims (int, optional): node dimension within the transformer stack. Defaults to 64.
            num_hidden (int, optional): hidden dim of transformer. Defaults to 64.
            num_blocks (int, optional): number of transformer blocks. Defaults to 1.
            key (KeyArray): PRNGKey for param initialization
        """
        self.net = MLPMixer(
            num_inp=auxiliary_shape[-1] + num_pos,  # * 2,
            num_out=4,  # num_rot,
            key=key,
            **kwargs,
        )

    def params(self, input: RigidWithAuxiliary):
        """Compute the parameters for the double moebius transform

        Args:
            input (State): current state

        Returns:
            Array: the parameter (reflection) of the double moebius transform
        """
        aux = input.aux
        pos = input.pos
        if len(aux.shape) == 1:
            aux = jnp.tile(aux[None], (pos.shape[0], 1))
        feats = jnp.concatenate([aux, pos], axis=-1)
        out = self.net(feats) * IDENTITY_FACTOR

        reflection = out

        reflection = reflection.reshape(input.rot.shape)
        reflection = jax.vmap(lambda x: x / (1 + geom.norm(x)) * 0.9999)(reflection)
        reflection = reflection

        return reflection

    def forward(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Forward transform"""
        reflection = self.params(input)
        new, ldj = unpack(
            VectorizedTransform(DoubleMoebius(reflection)).forward(input.rot)
        )
        return Transformed(lenses.bind(input).rot.set(new), ldj)

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Inverse transform"""
        reflection = self.params(input)
        new, ldj = unpack(
            VectorizedTransform(DoubleMoebius(reflection)).inverse(input.rot)
        )
        return Transformed(lenses.bind(input).rot.set(new), ldj)


class AuxUpdate(eqx.Module):
    """Flow layer updating the auxiliary part of a state"""

    symmetrizer: QuatEncoder
    net: MLPMixer  # | eqx.nn.Sequential | Conv
    auxiliary_shape: tuple[int, ...]
    num_low_rank: int
    low_rank_regularizer: float
    transform: str
    seq_len: int

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int,
        num_dims: int,
        num_low_rank: int,
        low_rank_regularizer: float,
        transform: str,
        *,
        key: KeyArray,
        **kwargs,
    ):
        """Flow layer updating the auxiliary part of a state.

        Args:
            auxiliary_shape (tuple[int, ...]): shape of auxilaries
            num_pos (int, optional): number of position DoF. Defaults to 3.
            num_heads (int, optional): number of transformer heads. Defaults to 4.
            num_dims (int, optional): node dimension within the transformer stack. Defaults to 64.
            num_hidden (int, optional): hidden dim of transformer. Defaults to 64.
            num_blocks (int, optional): number of transformer blocks. Defaults to 1.
            key (KeyArray): PRNGKey for param initialization
        """
        chain = key_chain(key)
        self.symmetrizer = QuatEncoder(num_dims, key=next(chain))
        self.auxiliary_shape = auxiliary_shape
        self.num_low_rank = num_low_rank
        self.low_rank_regularizer = low_rank_regularizer
        self.transform = transform
        num_out = (2 + 2 * num_low_rank) * auxiliary_shape[-1]
        self.net = MLPMixer(
            num_inp=num_dims + num_pos,  # * 2,
            num_out=num_out,
            key=next(chain),
            **kwargs,
        )
        self.seq_len = kwargs["seq_len"]

    def params(self, input: RigidWithAuxiliary):
        """Compute the parameters for the affine transform

        Args:
            input (State): current state

        Returns:
            tuple[Array, Array]: the parameters (shift, scale) of the affine transform
        """
        pos = input.pos
        feats = jnp.concatenate([pos, self.symmetrizer(input.rot)], axis=-1)
        out = self.net(feats).reshape(input.aux.shape[0], -1) * IDENTITY_FACTOR

        shift_and_scale, low_rank = jnp.split(out, [2 * input.aux.shape[-1]], axis=-1)  # type: ignore
        shift, scale = jnp.split(shift_and_scale, 2, axis=-1)  # type: ignore
        shift = shift.reshape(input.aux.shape)
        scale = scale.reshape(input.aux.shape)
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
    """Flow layer updating the position part of a state"""

    symmetrizer: QuatEncoder
    net: MLPMixer
    # net: Conv
    num_low_rank: int
    low_rank_regularizer: float
    transform: str
    seq_len: int

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_dims: int,
        num_pos: int,
        num_low_rank: int,
        low_rank_regularizer: float,
        transform: str,
        *,
        key: KeyArray,
        **kwargs,
    ):
        """Flow layer updating the position part of a state.

        Args:
            auxiliary_shape (tuple[int, ...]): shape of auxilaries
            num_pos (int, optional): number of position DoF. Defaults to 3.
            num_rot (int, optional): number of quaternion DoF. Defaults to 4.
            num_heads (int, optional): number of transformer heads. Defaults to 4.
            num_dims (int, optional): node dimension within the transformer stack. Defaults to 64.
            num_hidden (int, optional): hidden dim of transformer. Defaults to 64.
            num_blocks (int, optional): number of transformer blocks. Defaults to 1.
            key (KeyArray): PRNGKey for param initialization
        """
        self.num_low_rank = num_low_rank
        chain = key_chain(key)
        self.symmetrizer = QuatEncoder(num_dims, key=next(chain))

        num_out = (2 + 2 * num_low_rank) * num_pos
        self.net = MLPMixer(
            num_inp=num_dims + auxiliary_shape[-1],
            num_out=num_out,
            key=next(chain),
            **kwargs,
        )
        self.seq_len = kwargs["seq_len"]
        self.low_rank_regularizer = low_rank_regularizer
        self.transform = transform

    def params(self, input: RigidWithAuxiliary):  # -> tuple[Array, Array]:
        """Compute the parameters for the affine transform

        Args:
            input (State): current state

        Returns:
            tuple[Array, Array]: the parameters (shift, scale) of the affine transform
        """
        aux = input.aux
        if len(aux.shape) == 1:
            aux = jnp.tile(aux[None], (input.pos.shape[0], 1))

        feats = jnp.concatenate([aux, self.symmetrizer(input.rot)], axis=-1)
        out = self.net(feats).reshape(input.pos.shape[0], -1) * IDENTITY_FACTOR

        shift_and_scale, low_rank = jnp.split(out, [2 * input.pos.shape[-1]], axis=-1)  # type: ignore
        shift, scale = jnp.split(shift_and_scale, 2, axis=-1)  # type: ignore
        shift = shift.reshape(input.pos.shape)
        scale = scale.reshape(input.pos.shape)
        return shift, scale

    def forward(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Forward transform"""
        shift, scale = self.params(input)
        pipe = Affine(shift, scale)
        pos, ldj = unpack(pipe.forward(input.pos))
        return Transformed(lenses.bind(input).pos.set(pos), ldj)

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        """Inverse transform"""
        shift, scale = self.params(input)
        pipe = Affine(shift, scale)
        pos, ldj = unpack(pipe.inverse(input.pos))
        return Transformed(lenses.bind(input).pos.set(pos), ldj)


class EuclideanToRigidTransform(equinox.Module):

    mean: jnp.ndarray

    def forward(self, input: DataWithAuxiliary) -> Transformed[RigidWithAuxiliary]:
        rigid, ldj = unpack(VectorizedTransform(RigidTransform()).forward(input.pos))
        pos = rigid.pos - self.mean
        return Transformed(
            RigidWithAuxiliary(rigid.rot, pos, rigid.ics, input.aux, input.box),
            ldj,
        )

    def inverse(self, input: RigidWithAuxiliary) -> Transformed[DataWithAuxiliary]:

        pos = input.pos + self.mean
        rigid = jax.vmap(RigidRepresentation)(input.rot, pos, input.ics)
        pos, ldj = unpack(VectorizedTransform(RigidTransform()).inverse(rigid))

        sign = jnp.sign(input.rot[:, (0,)])
        return Transformed(
            DataWithAuxiliary(pos, input.aux, sign, input.box, None),
            ldj,
        )


def _coupling(
    key: KeyArray,
    auxiliary_shape: tuple[int, ...],
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
        aux_block = [
            AuxUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.auxiliary_update),
                key=next(chain),
            )
        ]
        pos_block = [
            PosUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.position_update),
                key=next(chain),
            ),
        ]

        if specs.act_norm:
            pos_block += [ActNorm(lenses.lens.pos)]
            aux_block += [ActNorm(lenses.lens.aux)]

        sub_block = Pipe(
            [
                *aux_block,
                *pos_block,
                QuatUpdate(
                    auxiliary_shape=auxiliary_shape,
                    **asdict(specs.quaternion_update),
                    key=next(chain),
                ),
            ]
        )
        blocks.append(sub_block)
    return Pipe(blocks)


def build_flow(
    key: KeyArray,
    auxiliary_shape: tuple[int, ...],
    specs: FlowSpecification,
    base: OpenMMDensity,
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
        blocks.append(_coupling(next(chain), auxiliary_shape, coupling))

    # couplings = LayerStackedPipe(blocks, use_scan=False)
    couplings = Pipe(blocks)
    return Pipe(
        [
            EuclideanToRigidTransform(base.data.modes),
            couplings,
            Inverted(EuclideanToRigidTransform(base.data.modes)),
        ]
    )
