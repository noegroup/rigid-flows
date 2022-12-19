from cmath import exp
from dataclasses import asdict, astuple
from itertools import accumulate
from math import prod
from turtle import forward
from typing import Any

import equinox as eqx
import jax
import lenses
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Float

from flox import geom
from flox._src.flow import rigid
from flox._src.flow.api import C
from flox._src.flow.impl import Affine
from flox.flow import (
    DoubleMoebius,
    Pipe,
    Transform,
    Transformed,
    VectorizedTransform,
)
from flox.util import key_chain, unpack

from .data import AugmentedData
from .lowrank import LowRankFlow  # type: ignore
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


class ActNorm(eqx.Module):

    lens: lenses.ui.UnboundLens
    mean: Array | None = None
    log_std: Array | None = None

    def forward(self, input: State):
        assert self.mean is not None
        assert self.log_std is not None
        val = self.lens.get()(input)
        scale = jax.nn.softplus(self.log_std) + 1e-6

        val = (val - self.mean) / scale

        ldj = -jnp.log(scale).sum()

        output = self.lens.set(val)(input)
        return Transformed(output, ldj)

    def inverse(self, input: State):
        assert self.mean is not None
        assert self.log_std is not None
        val = self.lens.get()(input)
        scale = jax.nn.softplus(self.log_std) + 1e-6

        val = val * scale + self.mean

        ldj = jnp.log(scale).sum()

        output = self.lens.set(val)(input)
        return Transformed(output, ldj)

    def initialize(self, batch: State):
        val = self.lens.get()(batch)
        std = 1e-6 + jnp.std(val, axis=0)
        log_std = jnp.log(jnp.exp(std) - 1)
        return ActNorm(
            self.lens,
            jnp.mean(val, axis=0),
            log_std,
        )


def initialize_actnorm(flow: Pipe, batch: Any):
    layers = []
    for layer in flow.transforms:
        if isinstance(layer, Pipe):
            layer = initialize_actnorm(layer, batch)

        if isinstance(layer, ActNorm):
            layer = layer.initialize(batch)
        else:
            batch, _ = unpack(jax.vmap(layer.forward)(batch))
        layers.append(layer)
    return Pipe(layers)


@pytree_dataclass(frozen=True)
class GatedShift:
    new: Array
    mix: Array

    def forward(self, input: Array):
        mix = jax.nn.sigmoid(self.mix)
        ldj = jax.nn.log_sigmoid(self.mix).sum()
        output = input * mix + self.new * (1.0 - mix)
        return Transformed(output, ldj)

    def inverse(self, input: Array):
        mix = jax.nn.sigmoid(self.mix)
        ldj = -jax.nn.log_sigmoid(self.mix).sum()
        output = (input - self.new * (1.0 - mix)) / self.mix
        return Transformed(output, ldj)


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
    num_low_rank: int
    low_rank_regularizer: float
    transform: str

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
        self.net = Dense(
            num_inp=num_dims + num_pos,
            num_out=(2 + 2 * self.num_low_rank) * auxiliary_shape[-1],
            key=next(chain),
            reduce_output=(len(self.auxiliary_shape) == 1),
            **kwargs,
        )

    def params(self, input: State):
        """Compute the parameters for the affine transform

        Args:
            input (State): current state

        Returns:
            tuple[Array, Array]: the parameters (shift, scale) of the affine transform
        """
        pos = input.pos - jnp.mean(input.pos, axis=(0, 1))
        feats = jnp.concatenate([pos, self.symmetrizer(input.rot)], axis=-1)
        out = self.net(feats).reshape(-1)

        splits = tuple(
            accumulate([prod(input.aux.shape), prod(input.aux.shape)])
        )

        new, mix, uvs = jnp.split(out, splits, axis=-1)  # type: ignore

        new = new.reshape(input.aux.shape)
        mix = mix.reshape(input.aux.shape)
        mix = mix * 1e-1
        u, v = uvs.reshape(2, -1)
        u = u / jnp.sqrt(prod(u.shape)) * 1e-1
        v = v / jnp.sqrt(prod(v.shape)) * 1e-1
        return new, mix, u, v

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        new, mix, u, v = self.params(input)
        match self.transform:
            case "gated":
                transform = GatedShift(new, mix)
            case "affine":
                transform = Affine(new, mix)
            case _:
                raise ValueError(f"unknown transform {self.transform}")
        pipe = Pipe(
            [
                LowRankFlow(u, v, self.low_rank_regularizer),
                transform,
            ]
        )
        aux, ldj = unpack(pipe.forward(input.aux))
        return Transformed(lenses.bind(input).aux.set(aux), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        new, mix, u, v = self.params(input)
        match self.transform:
            case "gated":
                transform = GatedShift(new, mix)
            case "affine":
                transform = Affine(new, mix)
            case _:
                raise ValueError(f"unknown transform {self.transform}")
        pipe = Pipe(
            [
                LowRankFlow(u, v, self.low_rank_regularizer),
                transform,
            ]
        )
        aux, ldj = unpack(pipe.inverse(input.aux))
        return Transformed(lenses.bind(input).aux.set(aux), ldj)


class PosUpdate(eqx.Module):
    """Flow layer updating the position part of a state"""

    symmetrizer: QuatEncoder
    net: Dense
    num_low_rank: int
    low_rank_regularizer: float
    transform: str

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
        self.net = Dense(
            num_inp=num_dims + auxiliary_shape[-1],
            num_out=(2 + 2 * num_low_rank) * num_pos,
            key=next(chain),
            **kwargs,
        )
        self.low_rank_regularizer = low_rank_regularizer
        self.transform = transform

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
        out = self.net(feats)
        out = out.reshape(-1)

        splits = tuple(
            accumulate([prod(input.pos.shape), prod(input.pos.shape)])
        )

        new, mix, uvs = jnp.split(out, splits, axis=-1)  # type: ignore

        new = new.reshape(input.pos.shape)
        mix = mix.reshape(input.pos.shape)
        mix = mix * 1e-1
        u, v = uvs.reshape(2, -1)
        u = u / jnp.sqrt(prod(u.shape)) * 1e-1
        v = v / jnp.sqrt(prod(v.shape)) * 1e-1
        return new, mix, u, v

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        new, mix, u, v = self.params(input)
        match self.transform:
            case "gated":
                transform = GatedShift(new, mix)
            case "affine":
                transform = Affine(new, mix)
            case _:
                raise ValueError(f"unknown transform {self.transform}")
        pipe = Pipe(
            [
                # LowRankFlow(u, v, self.low_rank_regularizer),
                transform,
            ]
        )
        pos, ldj = unpack(pipe.forward(input.pos))
        return Transformed(lenses.bind(input).pos.set(pos), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        new, mix, u, v = self.params(input)
        match self.transform:
            case "gated":
                transform = GatedShift(new, mix)
            case "affine":
                transform = Affine(new, mix)
            case _:
                raise ValueError(f"unknown transform {self.transform}")
        pipe = Pipe(
            [
                # LowRankFlow(u, v, self.low_rank_regularizer),
                transform,
            ]
        )
        pos, ldj = unpack(pipe.inverse(input.pos))
        return Transformed(lenses.bind(input).pos.set(pos), ldj)


class PositionEncoder(eqx.Module):
    """Encodes initial positions within the PBC box into
    displacement vectors relative to centers
    which are predicted from auxiliaries
    """

    net: Dense

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int,
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
        self.net = Dense(
            num_inp=auxiliary_shape[-1],
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
        feats = jnp.concatenate([aux], axis=-1)
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
) -> Transform[AugmentedData, State]:
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
    aux_block = [
        AuxUpdate(
            auxiliary_shape=auxiliary_shape,
            **asdict(specs.auxiliary_update),
            key=next(chain),
        )
    ]
    pos_block = [
        PositionEncoder(
            auxiliary_shape=auxiliary_shape,
            **asdict(specs.position_encoder),
            key=next(chain),
        )
    ]
    if specs.act_norm:
        pos_block += [ActNorm(lenses.lens.pos)]
        aux_block += [ActNorm(lenses.lens.aux)]
    return Pipe(
        [
            InitialTransform(),
            *aux_block,
            *pos_block,
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
            pos_block += [ActNorm(lenses.lens.pos)]
            aux_block += [ActNorm(lenses.lens.aux)]

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
