from collections.abc import Callable
from functools import partial
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from lenses import lens

from flox._src.nn.modules import dense
from flox.util import key_chain

KeyArray = jax.random.PRNGKeyArray | jnp.ndarray

POS_DIM = 3

__all__ = ["RotEncoder", "PosEncoder"]


class RotEncoder(eqx.Module):

    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    def __init__(
        self,
        dim: int,
        num_mol: int,
        num_out: int,
        num_lat: int,
        hidden: tuple[int, ...],
        activation: Callable[..., Any],
        *,
        key: KeyArray
    ):
        chain = key_chain(key)

        self.encoder = dense(
            units=(num_mol * dim, *hidden, num_lat),
            activation=activation,
            key=key,
        )
        self.decoder = dense(
            units=(num_lat, *hidden, num_out), activation=activation, key=key
        )

    def __call__(self, quat, key: KeyArray | None = None):
        # make flip invariant
        ypos = self.encoder(quat)
        yneg = self.encoder(-quat)

        # make sure gradients flow
        y = jnp.stack([ypos, yneg], axis=0)
        weight = jax.nn.softmax(y, axis=0)
        z = (weight * y).sum(0)
        return self.decoder(z)


class PosEncoder(eqx.Module):

    encoder: eqx.nn.Sequential

    def __init__(
        self,
        num_mol: int,
        num_out: int,
        hidden: tuple[int, ...],
        activation: Callable[..., Any],
        *,
        key: KeyArray
    ):
        self.encoder = dense(
            units=(2 * num_mol * POS_DIM, *hidden, num_out),
            activation=activation,
            key=key,
        )

    def __call__(self, box_and_pos, key: KeyArray | None = None):
        box, pos = box_and_pos
        return self.encoder(
            jnp.concatenate(
                [
                    jnp.sin(pos / box * 2 * jnp.pi),
                    jnp.cos(pos / box * 2 * jnp.pi),
                ],
                axis=-1,
            ).reshape(-1)
        )
