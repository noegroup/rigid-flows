from functools import partial
from itertools import accumulate
from math import sqrt

import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float

from flox.util import key_chain

KeyArray = jnp.ndarray | jax.random.PRNGKeyArray


class QuatEncoder(eqx.Module):
    """Encodes a quaternion into a flip-invariant representation."""

    encoder: eqx.nn.Linear

    def __init__(self, num_out: int, *, key: KeyArray):
        """Encodes a quaternion into a flip-invariant representation.

        Args:
            num_out (int): number of dimensions of output representation.
            key (KeyArray): PRNG Key for layer initialization
        """
        self.encoder = eqx.nn.Linear(4, num_out + 1, key=key)

    def __call__(
        self, quat: Float[Array, "... num_mols 4"]
    ) -> Float[Array, "... num_mols 4"]:
        inp = jnp.stack([quat, -quat])
        out = jax.vmap(jax.vmap(self.encoder))(inp)
        weight = jax.nn.softmax(out[..., 0], axis=0)
        return (weight[..., None] * out[..., 1:]).sum(axis=0)


class QuatEncoderSquare(eqx.Module):

    encoder: eqx.nn.Linear
    num_out: int
    num_channels: int
    norm: eqx.nn.LayerNorm

    def __init__(self, num_channels: int, num_out: int, *, key: KeyArray):
        self.encoder = eqx.nn.Linear(4, num_out * num_channels, key=key)
        self.num_out = num_out
        self.num_channels = num_channels
        self.norm = eqx.nn.LayerNorm(self.num_out)

    def __call__(self, quat):
        feat = jax.vmap(self.encoder)(quat).reshape(
            quat.shape[0], self.num_out, self.num_channels
        )
        out = (feat * feat).sum(axis=-1)
        return jax.vmap(self.norm)(out)


def modified_square_plus(x, a=0.2, b=1.0):
    return ((1.0 - a) * x + jnp.sqrt(jnp.square(x) + b) / jnp.sqrt(1 + b)) / (
        2 - a
    )


ACTIVATION_FUNCTIONS = {
    "silu": jax.nn.silu,
    "tanh": jax.nn.tanh,
    "mish": lambda x: x * jax.nn.tanh(jax.nn.softplus(x)),
    "gelu": jax.nn.gelu,
    "siren": lambda x: jnp.sin(x),
    "softplus": jax.nn.softplus,
    "squareplus": modified_square_plus,
}


class Transformer(eqx.Module):

    params: eqx.nn.Linear
    norm: eqx.nn.LayerNorm
    num_heads: int
    splits: tuple[int]

    def __init__(
        self,
        seq_len: int,
        num_inp: int,
        num_out: int,
        num_hidden: int,
        activation: str,
        num_blocks: int = 0,
        reduce_output: bool = False,
        *,
        key: KeyArray,
    ):
        chain = key_chain(key)
        self.num_heads = 8
        self.params = eqx.nn.Linear(
            num_inp,
            (self.num_heads * (2 * num_hidden + num_out)),
            key=next(chain),
        )
        self.norm = eqx.nn.LayerNorm(num_out, elementwise_affine=True)

        self.splits = tuple(accumulate([num_hidden, num_hidden]))

    def __call__(self, input, *args, **kwargs):

        out = jax.vmap(self.params)(input).reshape(
            input.shape[0], self.num_heads, -1
        )
        keys, queries, values = jnp.split(out, self.splits, axis=-1)  # type: ignore
        keys = keys / sqrt(keys.shape[-1])
        logits = jnp.einsum("ihk, jhk -> ijh", keys, queries)
        attention = jax.nn.softmax(logits, axis=1)
        output = jnp.einsum("ijh, jhe -> ie", attention, values)
        return jax.vmap(self.norm)(output)


class LayerStacked(eqx.Module):

    layers: eqx.Module

    def __init__(self, layers: list[eqx.Module]):
        params, static = eqx.partition(layers, eqx.is_array)  # type: ignore
        params = jax.tree_map(
            lambda *args: jnp.stack(args),
            *params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )
        self.layers = eqx.combine(params, static[0])  # type: ignore

    def __call__(self, x):

        params, static = eqx.partition(self.layers, eqx.is_array)  # type: ignore

        def body(x, param):
            fn = eqx.combine(param, static)
            y = fn(x)  # type: ignore
            return y, None

        y, _ = jax.lax.scan(body, x, params)
        return y


class AttentiveLayerStacked(eqx.Module):

    layers: eqx.Module
    query: eqx.nn.Linear
    keys: eqx.nn.Linear
    norm: eqx.nn.LayerNorm

    def __init__(
        self, inp_dim: int, hidden: int, layers: list[eqx.Module], *, key
    ):
        chain = key_chain(key)
        params, static = eqx.partition(layers, eqx.is_array)  # type: ignore
        params = jax.tree_map(
            lambda *args: jnp.stack(args),
            *params,
            is_leaf=lambda x: isinstance(x, jnp.ndarray),
        )
        self.layers = eqx.combine(params, static[0])  # type: ignore
        self.query = eqx.nn.Linear(inp_dim, hidden, key=next(chain))
        self.keys = eqx.nn.Linear(inp_dim, hidden, key=next(chain))
        self.norm = eqx.nn.LayerNorm(inp_dim, elementwise_affine=True)

    def __call__(self, x):

        x = self.norm(x)

        params, static = eqx.partition(self.layers, eqx.is_array)  # type: ignore

        def body(x, param):
            fn = eqx.combine(param, static)
            y = fn(x)  # type: ignore
            return y, y

        _, ys = jax.lax.scan(body, x, params)

        query = jax.vmap(self.query)(x)
        keys = jax.vmap(jax.vmap(self.keys))(ys)

        logits = jnp.einsum("nk, ink -> ni", query, keys)
        attention = jax.nn.softmax(logits, axis=0)

        output = jnp.einsum("ni, ink -> nk", attention, ys)
        return output


class MLPMixerLayer(eqx.Module):

    token_mixer: eqx.nn.Sequential
    dim_mixer: eqx.nn.Sequential
    token_norm: eqx.nn.LayerNorm
    dim_norm: eqx.nn.LayerNorm

    def __init__(
        self,
        seq_len: int,
        num_inp: int,
        expansion_factor: float,
        activation,
        *,
        key,
    ) -> None:
        chain = key_chain(key)
        num_hidden = int(num_inp * expansion_factor)
        self.dim_mixer = eqx.nn.Sequential(
            [
                eqx.nn.Linear(num_inp, num_hidden, key=next(chain)),
                eqx.nn.Lambda(activation),
                eqx.nn.Linear(num_hidden, num_inp, key=next(chain)),
            ]
        )
        self.dim_norm = eqx.nn.LayerNorm(num_inp, elementwise_affine=True)

        self.token_mixer = eqx.nn.Sequential(
            [
                eqx.nn.Linear(seq_len, num_hidden, key=next(chain)),
                eqx.nn.Lambda(activation),
                eqx.nn.Linear(num_hidden, seq_len, key=next(chain)),
            ]
        )
        self.token_norm = eqx.nn.LayerNorm(num_inp, elementwise_affine=True)

    def __call__(self, input):
        input = input + jax.vmap(self.dim_mixer, in_axes=0)(
            jax.vmap(self.dim_norm)(input)
        )
        input = input + jax.vmap(self.token_mixer, in_axes=1, out_axes=1)(
            jax.vmap(self.token_norm)(input)
        )
        return input


class MLPMixer(eqx.Module):

    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    mixers: LayerStacked | AttentiveLayerStacked
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        seq_len: int,
        num_inp: int,
        num_out: int,
        expansion_factor: float,
        activation: str,
        num_blocks: int = 0,
        *,
        key: KeyArray,
    ):
        chain = key_chain(key)
        num_hidden = int(num_inp * expansion_factor)
        self.encoder = eqx.nn.Linear(num_inp, num_hidden, key=next(chain))
        self.mixers = LayerStacked(
            [
                MLPMixerLayer(
                    seq_len,
                    num_hidden,
                    expansion_factor,
                    ACTIVATION_FUNCTIONS[activation],
                    key=next(chain),
                )
                for _ in range(num_blocks)
            ],
        )
        self.decoder = eqx.nn.Linear(num_hidden, num_out, key=next(chain))
        self.norm = eqx.nn.LayerNorm(num_hidden, elementwise_affine=True)

    def __call__(self, input):
        hidden = jax.vmap(self.encoder)(input)
        hidden = self.mixers(hidden)
        hidden = self.norm(hidden)
        out = jax.vmap(self.decoder)(hidden)
        return out


class RotationTransformerLayer(eqx.Module):

    keys: eqx.nn.Linear
    queries: eqx.nn.Linear
    features: eqx.nn.Sequential

    def __init__(
        self,
        num_inp: int,
        num_channels: int,
        features: eqx.nn.Sequential,
        *,
        key,
    ):
        chain = key_chain(key)
        self.keys = eqx.nn.Linear(num_inp, num_channels, key=next(chain))
        self.queries = eqx.nn.Linear(num_inp, num_channels, key=next(chain))
        self.features = features

    def __call__(
        self,
        rotations,
        features,
    ) -> None:
        rotations = jnp.concatenate([rotations, -rotations], axis=0)
        keys = jax.vmap(self.keys)(rotations)
        queries = jax.vmap(self.queries)(rotations)

        logits = jnp.einsum("ik, jk -> ij", queries, keys)
        attention = jax.nn.softmax(logits, axis=0)

        out = self.features(features)
        output = jnp.einsum("ij, jk -> k", attention, out)

        return output


class PositionTransformerLayer(eqx.Module):

    keys: eqx.nn.Linear
    queries: eqx.nn.Linear
    features: eqx.nn.Sequential

    def __init__(
        self,
        num_inp: int,
        num_channels: int,
        features: eqx.nn.Sequential,
        *,
        key,
    ):
        chain = key_chain(key)
        self.keys = eqx.nn.Linear(num_inp, num_channels, key=next(chain))
        self.queries = eqx.nn.Linear(num_inp, num_channels, key=next(chain))
        self.features = features

    def __call__(
        self,
        displacements,
        features,
    ):
        keys = jax.vmap(self.keys)(displacements)
        queries = jax.vmap(self.queries)(displacements)

        logits = jnp.einsum("ik, jk -> ij", queries, keys)
        attention = jax.nn.softmax(logits, axis=0)

        out = self.features(features)
        output = jnp.einsum("ij, jk -> k", attention, out)

        return output


class ResNetLayer(eqx.Module):

    residual: eqx.nn.Sequential
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        num_inp: int,
        expansion_factor: float,
        activation,
        *,
        key,
    ) -> None:
        chain = key_chain(key)
        num_hidden = int(num_inp * expansion_factor)
        self.residual = eqx.nn.Sequential(
            [
                eqx.nn.Linear(num_inp, num_hidden, key=next(chain)),
                eqx.nn.Lambda(activation),
                eqx.nn.Linear(num_hidden, num_inp, key=next(chain)),
            ]
        )
        self.norm = eqx.nn.LayerNorm(num_inp)

    def __call__(self, input: Array):
        return input + self.residual(self.norm(input))


class ResNet(eqx.Module):

    encoder: eqx.nn.Linear
    decoder: eqx.nn.Linear
    residuals: LayerStacked | AttentiveLayerStacked
    norm: eqx.nn.LayerNorm

    def __init__(
        self,
        seq_len: int,
        num_inp: int,
        num_out: int,
        expansion_factor: float,
        activation: str,
        num_blocks: int = 0,
        *,
        key: KeyArray,
    ):
        chain = key_chain(key)

        num_hidden = int(num_inp * expansion_factor)
        self.encoder = eqx.nn.Linear(num_inp, num_hidden, key=next(chain))
        self.residuals = LayerStacked(
            # seq_len * num_hidden,
            # 32,
            [
                ResNetLayer(
                    seq_len * num_hidden,
                    expansion_factor,
                    ACTIVATION_FUNCTIONS[activation],
                    key=next(chain),
                )
                for _ in range(num_blocks)
            ],
            # key=next(chain),
        )
        self.decoder = eqx.nn.Linear(num_hidden, num_out, key=next(chain))
        self.norm = eqx.nn.LayerNorm(num_hidden, elementwise_affine=True)

    def __call__(self, input):
        hidden = jax.vmap(self.encoder)(input)
        hidden = self.residuals(hidden.reshape(-1))
        hidden = jax.vmap(self.norm)(hidden.reshape(input.shape[0], -1))
        out = jax.vmap(self.decoder)(hidden)
        return out
