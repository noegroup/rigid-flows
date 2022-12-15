from dataclasses import asdict, astuple
from math import prod
from multiprocessing.sharedctypes import Value
from turtle import forward
from typing import Any

import equinox as eqx
import jax
import lenses
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Float  # type: ignore

from flox import geom
from flox._src.flow import rigid
from flox._src.flow.impl import Moebius
from flox.flow import (
    Affine,
    DoubleMoebius,
    Pipe,
    Transform,
    Transformed,
    VectorizedTransform,
)
from flox.util import key_chain, unpack

from .data import AugmentedData
from .nn import Dense, QuatEncoder
from .specs import (
    CouplingSpecification,
    FlowSpecification,
    PreprocessingSpecification,
)
from .system import SimulationBox

KeyArray = jnp.ndarray | jax.random.PRNGKeyArray


Scalar = Float[Array, ""] | float
Vector3 = Float[Array, "... 3"]
Quaternion = Float[Array, "... 4"]
Auxiliary = Float[Array, f"... AUX"]

AtomRepresentation = Float[Array, "... MOL 4 3"]


@pytree_dataclass(frozen=True)
class InternalCoordinates:
    d_OH1: Scalar = jnp.array(0.09572)
    d_OH2: Scalar = jnp.array(0.09572)
    a_HOH: Scalar = jnp.array(104.52 * jnp.pi / 180)
    d_OM: Scalar = jnp.array(0.0125)
    a_OM: Scalar = jnp.array(52.259937 * jnp.pi / 180)


@pytree_dataclass(frozen=True)
class RigidRepresentation:
    rot: Quaternion
    pos: Vector3
    ics: InternalCoordinates = InternalCoordinates()


def to_rigid(pos: AtomRepresentation) -> Transformed[RigidRepresentation]:
    q, p, *_ = rigid.from_euclidean(pos[:3])
    ldj = rigid.from_euclidean_log_jacobian(pos[:3])
    return Transformed(RigidRepresentation(q, p), ldj)


def from_rigid(rp: RigidRepresentation) -> Transformed[AtomRepresentation]:
    r_OM = rp.ics.d_OM * jnp.array(
        [jnp.sin(rp.ics.a_OM), 0.0, jnp.cos(rp.ics.a_OM)]
    )
    r_OM = geom.qrot3d(rp.rot, r_OM)
    pos = rigid.to_euclidean(rp.rot, rp.pos, *astuple(rp.ics)[:3])
    ldj = rigid.to_euclidean_log_jacobian(rp.rot, rp.pos, *astuple(rp.ics)[:3])
    pos = jnp.concatenate([pos, (pos[0] + r_OM)[None]], axis=0)
    return Transformed(pos, ldj)


class RigidTransform(Transform[AtomRepresentation, RigidRepresentation]):
    def forward(
        self, inp: AtomRepresentation
    ) -> Transformed[RigidRepresentation]:
        return to_rigid(inp)

    def inverse(
        self, inp: RigidRepresentation
    ) -> Transformed[AtomRepresentation]:
        return from_rigid(inp)


@pytree_dataclass
class State:
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


class QuatUpdate(eqx.Module):
    """Flow layer updating the quaternion part of a state"""

    net: Dense

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
        self.net = Dense(
            num_inp=auxiliary_shape[-1] + num_pos,
            num_out=4 * 4 + 4,  # num_rot,
            key=key,
            **kwargs,
        )

    def params(self, input: State):
        """Compute the parameters for the double moebius transform

        Args:
            input (State): current state

        Returns:
            Array: the parameter (reflection) of the double moebius transform
        """
        aux = input.aux
        pos = input.pos - jnp.mean(input.pos, axis=(0, 1))
        if len(aux.shape) == 1:
            aux = jnp.tile(aux[None], (pos.shape[0], 1))
        feats = jnp.concatenate([aux, input.pos], axis=-1)
        out = self.net(feats) * 1e-1

        mat, reflection = jnp.split(out, (16,), -1)  # type: ignore

        reflection = reflection.reshape(input.rot.shape)
        reflection = jax.vmap(lambda x: x / (1 + geom.norm(x)) * 0.9999)(
            reflection
        )

        return mat, reflection

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        mat, reflection = self.params(input)
        new, ldj = unpack(
            Pipe(
                [
                    VectorizedTransform(QuaternionAffine(mat)),
                    VectorizedTransform(DoubleMoebius(reflection)),
                ]
            ).forward(input.rot)
        )
        return Transformed(lenses.bind(input).rot.set(new), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        mat, reflection = self.params(input)
        new, ldj = unpack(
            Pipe(
                [
                    VectorizedTransform(QuaternionAffine(mat)),
                    VectorizedTransform(DoubleMoebius(reflection)),
                ]
            ).inverse(input.rot)
        )
        return Transformed(lenses.bind(input).rot.set(new), ldj)


class AuxUpdate(eqx.Module):
    """Flow layer updating the auxiliary part of a state"""

    symmetrizer: QuatEncoder
    net: Dense | eqx.nn.Sequential
    auxiliary_shape: tuple[int, ...]

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int = 3,
        num_dims: int = 64,
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
        self.net = Dense(
            num_inp=num_dims + num_pos,
            num_out=2 * auxiliary_shape[-1],
            key=next(chain),
            reduce_output=(len(self.auxiliary_shape) == 1),
            **kwargs,
        )

    def params(self, input: State) -> tuple[Array, Array]:
        """Compute the parameters for the affine transform

        Args:
            input (State): current state

        Returns:
            tuple[Array, Array]: the parameters (shift, scale) of the affine transform
        """
        pos = input.pos - jnp.mean(input.pos, axis=(0, 1))
        feats = jnp.concatenate([pos, self.symmetrizer(input.rot)], axis=-1)
        out = self.net(feats) * 1e-1
        out = out.reshape(input.aux.shape[0], -1)
        shift, scale = jnp.split(out, 2, axis=-1)
        shift = shift.reshape(self.auxiliary_shape)
        scale = scale.reshape(self.auxiliary_shape)
        return shift, scale

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        shift, pre_scale = self.params(input)

        keep = jax.nn.sigmoid(pre_scale)
        replace = jax.nn.sigmoid(-pre_scale)

        ldj = jax.nn.log_sigmoid(pre_scale).sum()

        new = input.aux * keep + shift * replace

        return Transformed(lenses.bind(input).aux.set(new), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        shift, pre_scale = self.params(input)

        keep = jax.nn.sigmoid(pre_scale)
        replace = jax.nn.sigmoid(-pre_scale)

        ldj = jax.nn.log_sigmoid(-pre_scale).sum()

        new = (input.aux - shift * replace) / keep

        return Transformed(lenses.bind(input).aux.set(new), ldj)


class ActNorm(eqx.Module):

    pos_mean: Array | None = None
    pos_log_std: Array | None = None
    aux_mean: Array | None = None
    aux_log_std: Array | None = None

    def forward(self, input: State):
        assert self.pos_mean is not None
        assert self.pos_log_std is not None
        assert self.aux_mean is not None
        assert self.aux_log_std is not None
        pos = (input.pos - self.pos_mean) * jnp.exp(-self.pos_log_std)
        aux = (input.aux - self.aux_mean) * jnp.exp(-self.aux_log_std)
        output = input
        output = lenses.bind(output).pos.set(pos)
        output = lenses.bind(output).aux.set(aux)
        ldj = -jnp.sum(self.pos_log_std) + jnp.sum(self.aux_log_std)
        return Transformed(output, ldj)

    def inverse(self, input: State):
        assert self.pos_mean is not None
        assert self.pos_log_std is not None
        assert self.aux_mean is not None
        assert self.aux_log_std is not None
        pos = input.pos * jnp.exp(self.pos_log_std) + self.pos_mean
        aux = input.aux * jnp.exp(self.pos_log_std) + self.aux_mean
        output = input
        output = lenses.bind(output).pos.set(pos)
        output = lenses.bind(output).aux.set(aux)
        ldj = jnp.sum(self.pos_log_std) + jnp.sum(self.aux_log_std)
        return Transformed(output, ldj)

    @staticmethod
    def initialize(batch: State):
        return ActNorm(
            pos_mean=jnp.mean(batch.pos, axis=0),
            aux_mean=jnp.mean(batch.aux, axis=0),
            pos_log_std=jnp.log(
                jnp.clip(jnp.std(batch.pos, axis=0), 0.01, 100.0)
            ),
            aux_log_std=jnp.log(
                jnp.clip(jnp.std(batch.aux, axis=0), 0.01, 100.0)
            ),
        )


def initialize_actnorm(flow: Pipe, batch: Any):
    layers = []
    for layer in flow.transforms:
        if isinstance(layer, Pipe):
            layer = initialize_actnorm(layer, batch)

        if isinstance(layer, ActNorm):
            layer = ActNorm.initialize(batch)
        else:
            batch, _ = unpack(jax.vmap(layer.forward)(batch))
        layers.append(layer)
    return Pipe(layers)


class PosUpdate(eqx.Module):
    """Flow layer updating the position part of a state"""

    symmetrizer: QuatEncoder
    net: Dense

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_dims: int = 64,
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
        chain = key_chain(key)
        self.symmetrizer = QuatEncoder(num_dims, key=next(chain))
        self.net = Dense(
            num_inp=num_dims + auxiliary_shape[-1],
            num_out=2 * 3,
            key=next(chain),
            **kwargs,
        )

    def params(self, input: State):  # -> tuple[Array, Array]:
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
        out = self.net(feats) * 1e-1
        out = out.reshape(input.pos.shape[0], -1)
        shift, scale = jnp.split(out, 2, axis=-1)
        return shift, scale

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        shift, pre_scale = self.params(input)

        keep = jax.nn.sigmoid(pre_scale)
        replace = jax.nn.sigmoid(-pre_scale)

        ldj = jax.nn.log_sigmoid(pre_scale).sum()

        new = input.pos * keep + shift * replace

        return Transformed(lenses.bind(input).pos.set(new), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        shift, pre_scale = self.params(input)

        keep = jax.nn.sigmoid(pre_scale)
        replace = jax.nn.sigmoid(-pre_scale)

        ldj = jax.nn.log_sigmoid(-pre_scale).sum()

        new = (input.pos - shift * replace) / keep

        return Transformed(lenses.bind(input).pos.set(new), ldj)


class PositionEncoder(eqx.Module):
    """Encodes initial positions within the PBC box into
    displacement vectors relative to centers
    which are predicted from auxiliaries
    """

    symmetrizer: QuatEncoder
    net: Dense

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int = 3,
        num_dims: int = 64,
        *,
        key: KeyArray,
        **kwargs,
    ):
        """Encodes initial positions within the PBC box into
        displacement vectors relative to centers
        which are predicted from auxiliaries

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
        chain = key_chain(key)
        self.symmetrizer = QuatEncoder(num_dims, key=next(chain))
        self.net = Dense(
            num_inp=auxiliary_shape[-1] + num_dims,
            num_out=num_pos,
            key=next(chain),
            **kwargs,
        )

    def params(self, input: State) -> Array:
        """Compute the parameters (centers) given the input state

        Args:
            input (State): current state

        Returns:
            Array: the centers relative to which displacements are computed
        """
        aux = input.aux
        if len(aux.shape) == 1:
            aux = jnp.tile(aux[None], (input.pos.shape[0], 1))
        feats = jnp.concatenate([aux, self.symmetrizer(input.rot)], axis=-1)
        out = self.net(feats) * 1e-1
        center = out.reshape(input.pos.shape)
        return center

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        center = self.params(input)
        ldj = jnp.zeros(())
        diff = input.pos - center
        return Transformed(lenses.bind(input).pos.set(diff), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        center = self.params(input)
        ldj = jnp.zeros(())
        pos = input.pos + center
        return Transformed(lenses.bind(input).pos.set(pos), ldj)


@pytree_dataclass(frozen=True)
class InitialTransform:
    """Initial transform, transforming data into a state."""

    def forward(self, input: AugmentedData) -> Transformed[State]:
        rigid, ldj = unpack(
            VectorizedTransform(RigidTransform()).forward(input.pos)
        )
        rigid = lenses.bind(rigid).rot.set(rigid.rot * input.sign)
        pos = rigid.pos + input.com
        state = State(rigid.rot, pos, rigid.ics, input.aux, input.box)
        return Transformed(state, ldj)

    def inverse(self, input: State) -> Transformed[AugmentedData]:
        rigid = jax.vmap(RigidRepresentation)(input.rot, input.pos)
        pos, ldj = unpack(VectorizedTransform(RigidTransform()).inverse(rigid))
        sign = jnp.sign(input.rot[:, (0,)])

        com = jnp.mean(pos, axis=(0, 1))
        pos = pos - com[None, None]
        data = AugmentedData(pos, com, input.aux, sign, input.box)
        return Transformed(data, ldj)


def _preprocess(
    key: KeyArray,
    auxiliary_shape: tuple[int, ...],
    specs: PreprocessingSpecification,
) -> Pipe[AugmentedData, State]:
    """The initial blocks handing:

     - mapping augmented data into a state
     - predicting position centers from the auxilaries
     - mapping positions into displacements relative to position centers

    Args:
        key (KeyArray): PRNG key
        num_aux (int): number of auxilaries

    Returns:
        Pipe[AugmentedData, State]: the initial transform
    """
    chain = key_chain(key)
    return Pipe[AugmentedData, State](
        [
            InitialTransform(),
            AuxUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.auxiliary_update),
                key=next(chain),
            ),
            PositionEncoder(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.displacement_encoder),
                key=next(chain),
            ),
            ActNorm(),
        ]
    )


def _coupling(
    key: KeyArray,
    auxiliary_shape: tuple[int, ...],
    specs: CouplingSpecification,
) -> Pipe[State, State]:
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
            pos_block += [ActNorm()]
            aux_block += [ActNorm()]

        blocks += [
            *aux_block,
            *pos_block,
            QuatUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.quaternion_update),
                key=next(chain),
            ),
        ]
    return Pipe[State, State](blocks)


def build_flow(
    key: KeyArray,
    auxiliary_shape: tuple[int, ...],
    specs: FlowSpecification,
) -> Pipe[AugmentedData, State]:
    """Creates the final flow composed of:

     - a preprocessing transformation
     - multiple coupling blokcks

    Args:
        key (KeyArray): PRNG key
        num_aux (int): number of auxilaries
        num_blocks (int, optional): number of coupling blocks. Defaults to 2.

    Returns:
        Pipe[AugmentedData, State]: the final flow
    """
    chain = key_chain(key)
    blocks: list[Transform[Any, Any]] = [
        _preprocess(next(chain), auxiliary_shape, specs.preprocessing)
    ]
    for coupling in specs.couplings:
        blocks.append(_coupling(next(chain), auxiliary_shape, coupling))
    return Pipe[AugmentedData, State](blocks)
