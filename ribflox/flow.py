from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Generic, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import lenses
from jax_dataclasses import pytree_dataclass
from jaxtyping import Array, Int  # type: ignore
from lenses import lens

import flox
import flox.geom as geom
from flox._src.geom.manifold import VectorN
from flox._src.nn.modules import dense
from flox.flow import (
    Affine,
    ConvexPotential,
    LensedTransform,
    Transform,
    Transformed,
    VectorizedTransform,
)
from flox.util import Lens, key_chain

from .nn import PosEncoder, RotEncoder
from .rep import RigidRepresentation, State

POS_DIM = 3
QUAT_DIM = 4


__all__ = ["pos_step", "aux_step", "rot_step", "full_step", "WrappedLens"]

KeyArray = jax.random.PRNGKeyArray | jnp.ndarray

S = TypeVar("S")
T = TypeVar("T")

A = TypeVar("A")
B = TypeVar("B")

X = TypeVar("X")
Y = TypeVar("Y")

Getter = Callable[[S], A]
Setter = Callable[[S, A], S]
Modifier = Callable[[A], B]

Decoder = Callable[[S, A], Transform[X, Y]]
Combiner = Callable[[Iterable[A]], A]


class ParameterSource(eqx.Module, Generic[S, A, B]):

    getter: Getter[S, A]  # TODO: more legible getter definition?
    modify: Modifier[A, B] | None = None

    def __call__(self, input: S) -> B:
        out = self.getter(input)
        if self.modify is not None:
            out = self.modify(out)
        else:
            out = cast(B, out)
        return out


class TransformFactory(eqx.Module, Generic[S, A, X, Y]):

    sources: tuple[Getter[S, A], ...]
    combine: Combiner[A]
    decoder: Decoder[S, A, X, Y]

    def __call__(self, input: S) -> Transform[X, Y]:
        combined = self.combine([source(input) for source in self.sources])
        return self.decoder(input, combined)


class GeneralStep(eqx.Module, Generic[S, T, A, B, X, Y]):

    lens_fwd: Lens[S, T, X, Y]
    lens_bwd: Lens[T, S, Y, X]
    fact_fwd: TransformFactory[S, A, X, Y]
    fact_bwd: TransformFactory[T, B, Y, X]

    def forward(self, input: S) -> Transformed[T]:
        return LensedTransform(
            self.fact_fwd(input), self.lens_fwd, self.lens_bwd
        ).forward(input)

    def inverse(self, input: T) -> Transformed[S]:
        return LensedTransform(
            self.fact_bwd(input), self.lens_bwd, self.lens_fwd
        ).forward(input)


class SimpleStep(eqx.Module, Generic[S, A, B, X]):

    lens: Lens[S, S, X, X]
    fact: TransformFactory[S, A, X, X]

    def forward(self, input: S) -> Transformed[S]:
        return LensedTransform(self.fact(input), self.lens, self.lens).forward(
            input
        )

    def inverse(self, input: S) -> Transformed[S]:
        return LensedTransform(self.fact(input), self.lens, self.lens).inverse(
            input
        )


class WrappedLens(eqx.Module):

    lens: lenses.ui.UnboundLens

    def inject(self, state, new):
        return self.lens.set(new)(state)

    def project(self, state):
        return self.lens.get()(state)


class LensGetter(eqx.Module):
    lens: lenses.ui.UnboundLens

    def __call__(self, target):
        return self.lens.get()(target)


def compute_splits(
    tot: int,
    inp: int,
) -> Int[Array, ""]:
    assert (tot - 2) % (inp + 2) == 0
    nctrl = (tot - 2) // (inp + 2)
    splits = [
        nctrl * (0 + inp),
        nctrl * (1 + inp) + 1,
    ]
    return cast(Array, splits)


def compute_total_dim(nctrl: int, inp: int) -> int:
    return nctrl * inp + 2 * (nctrl + 1)


@pytree_dataclass(frozen=True)
class ManifoldShift(Transform[VectorN, VectorN]):
    torus: flox.geom.Torus
    shift: VectorN

    def forward(self, inp: VectorN):
        return Transformed(
            self.torus.shift(inp, self.shift), jnp.zeros(())
        )

    def inverse(self, inp: VectorN):
        return Transformed(
            self.torus.shift(inp, -self.shift), jnp.zeros(())
        )


class TorusShiftDecoder(eqx.Module):

    net: Callable[[Array], Array]
    out_shape: tuple[int, ...]
    num_mol: int

    def __init__(
        self,
        num_mol: int,
        num_inp: int,
        num_out: int,
        hidden: tuple[int, ...],
        activation: Callable[..., Any],
        *,
        key: KeyArray
    ):
        self.net = dense(
            units=(num_inp, *hidden, num_mol * num_out),
            activation=activation,
            key=key,
        )
        self.out_shape = (num_mol, num_out)
        self.num_mol = num_mol

    def __call__(
        self, state: State, latent: Array
    ) -> VectorizedTransform[VectorN, VectorN]:
        shift = self.net(latent).reshape(self.out_shape)
        box = jnp.tile(state.box[None], reps=(self.num_mol, 1))
        return VectorizedTransform(ManifoldShift(geom.Torus(box), shift))


class AffineDecoder(eqx.Module):

    net: Callable[[Array], Array]
    out_shape: tuple[int, ...]

    def __init__(
        self,
        num_mol: int,
        num_inp: int,
        num_out: int,
        hidden: tuple[int, ...],
        activation: Callable[..., Any],
        *,
        key: KeyArray
    ):
        POS_DIM = 3
        self.net = dense(
            units=(num_inp, *hidden, num_mol * num_out * 2),
            activation=activation,
            key=key,
        )
        self.out_shape = (2, num_mol, num_out)

    def __call__(self, state: State, latent: Array):
        (shift,), (scale,) = jnp.split(
            self.net(latent).reshape(self.out_shape), 2, 0
        )
        return VectorizedTransform(Affine(shift, scale))


class ConvexPotentialGradientDecoder(eqx.Module):

    net: Callable[[Array], Array]
    dim: int
    out_shape: tuple[int, ...]

    def __init__(
        self,
        num_mol: int,
        num_inp: int,
        num_out: int,
        num_ctrl_pts: int,
        hidden: tuple[int, ...],
        activation: Callable[..., Any],
        *,
        key: KeyArray
    ):
        self.dim = num_out
        num_params = compute_total_dim(num_ctrl_pts, num_out)
        self.net = dense(
            units=(num_inp, *hidden, num_mol * num_params),
            activation=activation,
            key=key,
        )
        self.out_shape = (num_mol, num_params)

    def __call__(self, state: State[RigidRepresentation], latent: Array):
        params = self.net(latent)
        params = params.reshape(self.out_shape)
        ctrlpts, weights, bias = jnp.split(
            params, compute_splits(params.shape[-1], self.dim), axis=-1
        )
        ctrlpts = ctrlpts.reshape(
            *state.mol.rot.shape[:-1], -1, state.mol.rot.shape[-1]
        )
        return VectorizedTransform(ConvexPotential(ctrlpts, weights, bias))


def rot_step(
    key: KeyArray,
    num_mol: int,
    num_auxiliaries: int,
    num_latent: int,
    num_hidden: tuple[int, ...],
    num_ctrl_pts: int,
) -> Transform[State[RigidRepresentation], State[RigidRepresentation]]:
    chain = key_chain(key)
    return SimpleStep(
        lens=WrappedLens(lens.mol.rot),
        fact=TransformFactory(
            sources=[
                ParameterSource(
                    getter=LensGetter(
                        lens.Tuple(lens.box.Parts()[0], lens.mol.pos.Parts()[0])
                    ),
                    modify=PosEncoder(
                        num_mol=num_mol,
                        num_out=num_latent,
                        hidden=num_hidden,
                        activation=jax.nn.silu,
                        key=next(chain),
                    ),
                ),
                ParameterSource(
                    getter=LensGetter(lens.aux),
                    modify=eqx.nn.Sequential(
                        [
                            eqx.nn.Lambda(partial(jnp.reshape, newshape=(-1,))),
                            dense(
                                (
                                    num_mol * num_auxiliaries,
                                    *num_hidden,
                                    num_latent,
                                ),
                                jax.nn.silu,
                                key=next(chain),
                            ),
                        ]
                    ),
                ),
            ],
            combine=jnp.concatenate,
            decoder=ConvexPotentialGradientDecoder(
                num_mol=num_mol,
                num_inp=2 * num_latent,
                num_out=4,
                num_ctrl_pts=num_ctrl_pts,
                hidden=num_hidden,
                activation=jax.nn.silu,
                key=next(chain),
            ),
        ),
    )


def pos_step(
    key: KeyArray,
    num_mol: int,
    num_auxiliaries: int,
    num_latent: int,
    num_hidden: tuple[int, ...],
) -> Transform[State[RigidRepresentation], State[RigidRepresentation]]:
    chain = key_chain(key)
    return SimpleStep(
        lens=WrappedLens(lens.mol.pos),
        fact=TransformFactory(
            sources=[
                ParameterSource(
                    getter=LensGetter(lens.mol.rot),
                    modify=eqx.nn.Sequential(
                        [
                            eqx.nn.Lambda(partial(jnp.reshape, newshape=(-1,))),
                            RotEncoder(
                                dim=QUAT_DIM,
                                num_mol=num_mol,
                                num_out=num_latent,
                                num_lat=num_latent,
                                hidden=num_hidden,
                                activation=jax.nn.silu,
                                key=next(chain),
                            ),
                        ]
                    ),
                ),
                ParameterSource(
                    getter=LensGetter(lens.aux),
                    modify=eqx.nn.Sequential(
                        [
                            eqx.nn.Lambda(partial(jnp.reshape, newshape=(-1,))),
                            dense(
                                (
                                    num_mol * num_auxiliaries,
                                    *num_hidden,
                                    num_latent,
                                ),
                                jax.nn.silu,
                                key=next(chain),
                            ),
                        ]
                    ),
                ),
            ],
            combine=jnp.concatenate,
            decoder=TorusShiftDecoder(
                num_mol=num_mol,
                num_inp=2 * num_latent,
                num_out=POS_DIM,
                hidden=num_hidden,
                activation=jax.nn.silu,
                key=next(chain),
            ),
        ),
    )


def aux_step(
    key: KeyArray,
    num_mol: int,
    num_auxiliaries: int,
    num_latent: int,
    num_hidden: tuple[int, ...],
) -> Transform[State[RigidRepresentation], State[RigidRepresentation]]:
    chain = key_chain(key)
    return SimpleStep(
        lens=WrappedLens(lens.aux),
        fact=TransformFactory(
            sources=[
                ParameterSource(
                    getter=LensGetter(lens.mol.rot),
                    modify=eqx.nn.Sequential(
                        [
                            eqx.nn.Lambda(partial(jnp.reshape, newshape=(-1,))),
                            RotEncoder(
                                dim=QUAT_DIM,
                                num_mol=num_mol,
                                num_out=num_latent,
                                num_lat=num_latent,
                                hidden=num_hidden,
                                activation=jax.nn.silu,
                                key=next(chain),
                            ),
                        ]
                    ),
                ),
                ParameterSource(
                    getter=LensGetter(
                        lens.Tuple(lens.box.Parts()[0], lens.mol.pos.Parts()[0])
                    ),
                    modify=PosEncoder(
                        num_mol=num_mol,
                        num_out=num_latent,
                        hidden=num_hidden,
                        activation=jax.nn.silu,
                        key=next(chain),
                    ),
                ),
            ],
            combine=jnp.concatenate,
            decoder=AffineDecoder(
                num_mol=num_mol,
                num_inp=2 * num_latent,
                num_out=num_auxiliaries,
                hidden=num_hidden,
                activation=jax.nn.silu,
                key=next(chain),
            ),
        ),
    )


def full_step(
    key: KeyArray,
    num_mol: int,
    num_auxiliaries: int,
    num_latent: int,
    num_hidden: tuple[int, ...],
    num_ctrl_pts: int,
):
    chain = key_chain(key)
    return flox.flow.Pipe[
        State[RigidRepresentation], State[RigidRepresentation]
    ](
        [
            pos_step(
                next(chain), num_mol, num_auxiliaries, num_latent, num_hidden
            ),
            aux_step(
                next(chain), num_mol, num_auxiliaries, num_latent, num_hidden
            ),
            rot_step(
                next(chain),
                num_mol,
                num_auxiliaries,
                num_latent,
                num_hidden,
                num_ctrl_pts,
            ),
        ]
    )
