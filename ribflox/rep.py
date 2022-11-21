from dataclasses import astuple
from typing import Generic, TypeVar

import jax
import jax.numpy as jnp
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Float  # type: ignore

import flox
from flox._src.flow import rigid
from flox.flow import Transform, Transformed

Scalar = Float[Array, ""] | float
Vector3 = Float[Array, "... 3"]
Quaternion = Float[Array, "... 4"]
Auxiliary = Float[Array, f"... AUX"]

BoxSize = Vector3

__all__ = [
    "InternalCoordinates",
    "RigidRepresentation",
    "AtomRepresentation",
    "RigidTransform",
]


@pytree_dataclass(frozen=True)
class InternalCoordinates:
    d_OH1: Scalar = 0.095712
    d_OH2: Scalar = 0.095712
    a_HOH: Scalar = 104.52 * jnp.pi / 180
    d_OM: Scalar = 0.0125
    a_OM: Scalar = 52.259937 * jnp.pi / 180


@pytree_dataclass(frozen=True)
class RigidRepresentation:
    rot: Quaternion
    pos: Vector3
    ics: InternalCoordinates = InternalCoordinates()


AtomRepresentation = Float[Array, "... MOL 4 3"]

Representation = TypeVar(
    "Representation", RigidRepresentation, AtomRepresentation
)


@pytree_dataclass(frozen=True)
class State(Generic[Representation]):
    mol: Representation
    aux: Auxiliary
    box: BoxSize


def to_rigid(pos: AtomRepresentation) -> Transformed[RigidRepresentation]:
    q, p, *_ = rigid.from_euclidean(pos[:3])
    ldj = rigid.from_euclidean_log_jacobian(pos[:3])
    return Transformed(RigidRepresentation(q, p), ldj)


def from_rigid(rp: RigidRepresentation) -> Transformed[AtomRepresentation]:
    r_OM = rp.ics.d_OM * jnp.array(
        [jnp.sin(rp.ics.a_OM), 0.0, jnp.cos(rp.ics.a_OM)]
    )
    r_OM = flox.geom.qrot3d(rp.rot, r_OM)
    pos = rigid.to_euclidean(rp.rot, rp.pos, *astuple(rp.ics)[:3])
    ldj = rigid.to_euclidean_log_jacobian(rp.rot, rp.pos, *astuple(rp.ics)[:3])
    pos = jnp.concatenate([pos, (pos[0] + r_OM)[None]], axis=0)
    return Transformed(pos, ldj)


class RigidTransform(Transform[AtomRepresentation, RigidRepresentation]):
    def forward(
        self, inp: AtomRepresentation
    ) -> Transformed[RigidRepresentation]:
        return jax.vmap(to_rigid)(inp)

    def inverse(
        self, inp: RigidRepresentation
    ) -> Transformed[AtomRepresentation]:
        return jax.vmap(from_rigid)(inp)
