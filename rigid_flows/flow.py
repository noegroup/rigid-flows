from dataclasses import asdict, astuple
from math import prod
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
from .nn import QuatEncoder, TransformerStack
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

    net: TransformerStack

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int = 3,
        num_heads: int = 4,
        num_dims: int = 64,
        num_hidden: int = 64,
        num_blocks: int = 1,
        *,
        key: KeyArray,
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
        self.net = TransformerStack(
            num_inp=auxiliary_shape[-1] + num_pos * 2,
            num_out=4 * 4 + 4,  # num_rot,
            num_heads=num_heads,
            num_dims=num_dims,
            num_hidden=num_hidden,
            num_blocks=num_blocks,
            key=key,
        )

    def params(self, input: State):
        """Compute the parameters for the double moebius transform

        Args:
            input (State): current state

        Returns:
            Array: the parameter (reflection) of the double moebius transform
        """
        aux = input.aux
        if len(aux.shape) == 1:
            aux = jnp.tile(aux[None], (input.pos.shape[0], 1))
        # feats = jnp.concatenate([aux, input.pos], axis=-1)
        feats = jnp.concatenate([aux, input.pos[..., 0], input.pos[..., 1]])
        out = self.net(feats) * 1e-2

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
    net: TransformerStack | eqx.nn.Sequential
    auxiliary_shape: tuple[int, ...]

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int = 3,
        num_heads: int = 4,
        num_dims: int = 64,
        num_hidden: int = 64,
        num_blocks: int = 1,
        *,
        key: KeyArray,
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
        self.net = TransformerStack(
            num_inp=num_dims + num_pos * 2,
            num_out=2 * auxiliary_shape[-1],
            num_heads=num_heads,
            num_dims=num_dims,
            num_hidden=num_hidden,
            num_blocks=num_blocks,
            key=next(chain),
            reduce_output=(len(self.auxiliary_shape) == 1),
        )

    def params(self, input: State) -> tuple[Array, Array]:
        """Compute the parameters for the affine transform

        Args:
            input (State): current state

        Returns:
            tuple[Array, Array]: the parameters (shift, scale) of the affine transform
        """
        # feats = jnp.concatenate(
        #     [input.pos, self.symmetrizer(input.rot)], axis=-1
        # )
        feats = jnp.concatenate(
            [input.pos[..., 0], input.pos[..., 1], self.symmetrizer(input.rot)],
            axis=-1,
        )
        out = self.net(feats)
        out = out.reshape(input.aux.shape[0], -1)
        shift, scale = jnp.split(out, 2, axis=-1)
        shift = shift.reshape(self.auxiliary_shape)
        scale = scale.reshape(self.auxiliary_shape)
        scale = scale * 1e-2
        return shift, scale

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        shift, scale = self.params(input)
        new, ldj = unpack(Affine(shift, scale).forward(input.aux))
        ldj = jnp.sum(ldj)
        return Transformed(lenses.bind(input).aux.set(new), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        shift, scale = self.params(input)
        new, ldj = unpack(Affine(shift, scale).inverse(input.aux))
        ldj = jnp.sum(ldj)
        return Transformed(lenses.bind(input).aux.set(new), ldj)


class PosUpdate(eqx.Module):
    """Flow layer updating the position part of a state"""

    symmetrizer: QuatEncoder
    net: TransformerStack

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_heads: int = 4,
        num_dims: int = 64,
        num_hidden: int = 64,
        num_blocks: int = 1,
        *,
        key: KeyArray,
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
        self.net = TransformerStack(
            num_inp=num_dims + auxiliary_shape[-1],
            # num_out=12,
            num_out=2 * 3,
            num_heads=num_heads,
            num_dims=num_dims,
            num_hidden=num_hidden,
            num_blocks=num_blocks,
            key=next(chain),
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
        out = self.net(feats) * 1e-2
        out = out.reshape(input.pos.shape)
        norm = norm = jnp.sqrt(1e-12 + jnp.square(out)).sum(
            axis=-1, keepdims=True
        )
        reflection = out / (1.0 + norm) * 0.99
        return reflection
        # out = out.reshape(input.pos.shape[0], -1)
        # out = out * 1e-2
        # return out * 1e-2
        shift, scale = jnp.split(out, 2, axis=-1)
        scale = scale
        return shift, scale

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        reflection = self.params(input)
        new, ldj = unpack(
            VectorizedTransform(
                VectorizedTransform(Moebius(reflection))
            ).forward(input.pos)
        )
        return Transformed(lenses.bind(input).pos.set(new), ldj)
        # shift, scale = self.params(input)
        # new, ldj = unpack(Affine(shift, scale).forward(input.pos))
        # ldj = jnp.sum(ldj)
        # params = self.params(input)
        # new, ldj = unpack(
        #     VectorizedTransform(FullAffine(params)).forward(input.pos)
        #     # VectorizedTransform(FullAffine(params)).forward(input.pos)
        # )
        # return Transformed(lenses.bind(input).pos.set(new), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        reflection = self.params(input)
        new, ldj = unpack(
            VectorizedTransform(
                VectorizedTransform(Moebius(reflection))
            ).inverse(input.pos)
        )
        return Transformed(lenses.bind(input).pos.set(new), ldj)
        # shift, scale = self.params(input)
        # new, ldj = unpack(Affine(shift, scale).inverse(input.pos))
        # ldj = jnp.sum(ldj)
        # params = self.params(input)
        # new, ldj = unpack(
        #     VectorizedTransform(FullAffine(params)).inverse(input.pos)
        #     # VectorizedTransform(FullAffine(params)).inverse(input.pos)
        # )
        # return Transformed(lenses.bind(input).pos.set(new), ldj)


# def log_cosh(x):
#     return jax.nn.logsumexp(jnp.stack([x, -x], axis=0), axis=0) + jnp.log(0.5)


# def tanh_transform(
#     x,
# ):
#     y = jnp.tanh(x)
#     ldj = -2 * log_cosh(x)
#     return y, jnp.sum(ldj)


# def atanh_transform(
#     x,
# ):
#     x = jnp.clip(x, -1.0 + 1e-4, 1.0 - 1e-4)
#     y = jnp.arctanh(x)
#     ldj = jnp.log(1.0 / (1.0 - jnp.square(x)))
#     # ldj = 2 * log_cosh(y)
#     return y, jnp.sum(ldj)


# @pytree_dataclass(frozen=True)
# class AtanhTransform:
#     def forward(self, input: Array) -> Transformed[Array]:
#         return Transformed(*atanh_transform(input))

#     def inverse(self, input: Array) -> Transformed[Array]:
# return Transformed(*tanh_transform(input))


class DisplacementEncoder(eqx.Module):
    """Encodes initial positions within the PBC box into
    displacement vectors relative to centers
    which are predicted from auxiliaries
    """

    symmetrizer: QuatEncoder
    net: TransformerStack

    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        num_pos: int = 3,
        num_heads: int = 4,
        num_dims: int = 64,
        num_hidden: int = 64,
        num_blocks: int = 1,
        *,
        key: KeyArray,
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
        self.net = TransformerStack(
            num_inp=auxiliary_shape[-1] + num_dims,
            # num_out=num_pos,
            num_out=1,
            num_heads=num_heads,
            num_dims=num_dims,
            num_hidden=num_hidden,
            key=next(chain),
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
        out = jnp.tanh(out) * jnp.pi
        center = jax.vmap(jax.vmap(geom.rotmat2d))(out)
        # center = out.reshape(input.pos.shape[:-1])
        # center = jax.nn.sigmoid(center) * input.box.size
        return center

    def forward(self, input: State) -> Transformed[State]:
        """Forward transform"""
        center = self.params(input)
        pos = jnp.einsum("...i, ...ij -> ...j", input.pos, center)
        ldj = jnp.zeros(())
        return Transformed(lenses.bind(input).pos.set(pos), ldj)
        # diff = geom.Torus(input.box.size).tangent(center, input.pos - center)
        # diff = diff / input.box.size * 2
        # diff, ldj = unpack(VectorizedTransform(AtanhTransform()).forward(diff))
        # return Transformed(lenses.bind(input).pos.set(diff), ldj)

    def inverse(self, input: State) -> Transformed[State]:
        """Inverse transform"""
        center = self.params(input)
        center = self.params(input)
        pos = jnp.einsum("...i, ...ji -> ...j", input.pos, center)
        ldj = jnp.zeros(())
        return Transformed(lenses.bind(input).pos.set(pos), ldj)
        # diff, ldj = unpack(
        #     VectorizedTransform(AtanhTransform()).inverse(input.pos)
        # )
        # diff = diff * input.box.size / 2
        # pos = geom.Torus(input.box.size).shift(center, diff)
        # return Transformed(lenses.bind(input).pos.set(pos), ldj)


@pytree_dataclass(frozen=True)
class InitialTransform:
    """Initial transform, transforming data into a state."""

    def forward(self, input: AugmentedData) -> Transformed[State]:
        rigid, ldj = unpack(
            VectorizedTransform(RigidTransform()).forward(input.pos)
        )
        rigid = lenses.bind(rigid).rot.set(rigid.rot * input.sign)
        pos = (rigid.pos % input.box.size) / input.box.size
        pos = rigid.pos * (2 * jnp.pi) - jnp.pi
        pos = jnp.stack(
            [
                jnp.cos(pos),
                jnp.sin(pos),
            ],
            axis=-1,
        )

        state = State(rigid.rot, pos, rigid.ics, input.aux, input.box)
        return Transformed(state, ldj)

    def inverse(self, input: State) -> Transformed[AugmentedData]:
        pos = jnp.arctan2(input.pos[..., 1], input.pos[..., 0])
        pos = (pos + jnp.pi) / (2 * jnp.pi)
        pos = pos * input.box.size
        rigid = jax.vmap(RigidRepresentation)(input.rot, pos)
        pos, ldj = unpack(VectorizedTransform(RigidTransform()).inverse(rigid))
        sign = jnp.sign(input.rot[:, (0,)])
        data = AugmentedData(pos, input.aux, sign, input.box)
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
            DisplacementEncoder(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.displacement_encoder),
                key=next(chain),
            ),
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
        blocks += [
            AuxUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.auxiliary_update),
                key=next(chain),
            ),
            QuatUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.quaternion_update),
                key=next(chain),
            ),
            PosUpdate(
                auxiliary_shape=auxiliary_shape,
                **asdict(specs.position_update),
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
