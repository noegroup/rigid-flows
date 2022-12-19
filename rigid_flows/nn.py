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


def modified_square_plus(x, a=0.2, b=1.0):
    return ((1.0 - a) * x + jnp.sqrt(jnp.square(x) + b) / jnp.sqrt(1 + b)) / (
        2 - a
    )


ACTIVATION_FUNCTIONS = {
    "silu": jax.nn.silu,
    "tanh": jax.nn.tanh,
    "mish": lambda x: x * jax.nn.tanh(jax.nn.softplus(x)),
    "gelu": jax.nn.gelu,
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
        # self.queries = eqx.nn.Linear(
        #     num_inp, (self.num_heads * num_hidden), key=next(chain)
        # )
        # self.values = eqx.nn.Linear(
        #     num_inp, (self.num_heads * num_out), key=next(chain)
        # )
        self.norm = eqx.nn.LayerNorm(num_out, elementwise_affine=True)

        self.splits = tuple(accumulate([num_hidden, num_hidden]))

    def __call__(self, input, *args, **kwargs):

        out = jax.vmap(self.params)(input).reshape(
            input.shape[0], self.num_heads, -1
        )
        keys, queries, values = jnp.split(out, self.splits, axis=-1)  # type: ignore
        keys = keys / sqrt(keys.shape[-1])
        # keys = jax.vmap(self.keys)(input).reshape(
        #     input.shape[0], self.num_heads, -1
        # )
        # keys = keys / sqrt(keys.shape[-1])
        # queries = jax.vmap(self.queries)(input).reshape(
        #     input.shape[0], self.num_heads, -1
        # )
        # values = jax.vmap(self.values)(input).reshape(
        #     input.shape[0], self.num_heads, -1
        # )
        logits = jnp.einsum("ihk, jhk -> ijh", keys, queries)
        attention = jax.nn.softmax(logits, axis=1)
        output = jnp.einsum("ijh, jhe -> ie", attention, values)
        return jax.vmap(self.norm)(output)




class LayerStacked(eqx.Module):

    layers: eqx.Module

    def __init__(self, layers: list[eqx.Module]):
        self.layers = jax.tree_map(jnp.stack, layers, is_leaf=eqx.is_array)

    def __call__(self, x):

        params, static = eqx.partition(self.layers, eqx.is_array)

        def body(x, param):
            fn = eqx.combine(param, static)
            y = fn(x)
            return y, None

        y, _ = jax.lax.scan(body, x, params)
        return y



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
        *,
        key,
    ) -> None:
        chain = key_chain(key)
        num_hidden = int(num_inp * expansion_factor)
        self.dim_mixer = eqx.nn.Sequential(
            [
                eqx.nn.Linear(num_inp, num_hidden, key=next(chain)),
                eqx.nn.Lambda(jax.nn.gelu),
                eqx.nn.Linear(num_hidden, num_inp, key=next(chain)),
            ]
        )
        self.dim_norm = eqx.nn.LayerNorm(num_inp, elementwise_affine=True)

        self.token_mixer = eqx.nn.Sequential(
            [
                eqx.nn.Linear(seq_len, num_hidden, key=next(chain)),
                eqx.nn.Lambda(jax.nn.gelu),
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
    mixers: LayerStacked#list[MLPMixerLayer]
    norm: eqx.nn.LayerNorm

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
        self.encoder = eqx.nn.Linear(num_inp, num_hidden, key=next(chain))
        self.mixers = LayerStacked([
            MLPMixerLayer(seq_len, num_hidden, 2, key=next(chain))
            for _ in range(num_blocks)
        ])

        self.decoder = eqx.nn.Linear(num_hidden, num_out, key=next(chain))
        self.norm = eqx.nn.LayerNorm(num_hidden, elementwise_affine=True)

    def __call__(self, input):
        hidden = jax.vmap(self.encoder)(input)
        hidden =self.mixers(hidden)
        hidden = self.norm(hidden)
        out = jax.vmap(self.decoder)(hidden)
        return out


class Dense(eqx.Module):
    """Stack of transformer layers.

    DISCLAIMER: right now only implements a simple dense net!!!!
    """

    encoder: eqx.nn.Sequential
    decoder: eqx.nn.Sequential

    inner: tuple[eqx.nn.Sequential]

    reduce_output: bool
    seq_len: int

    # use simple dense net for now as transformers don't work (yet)
    # foo: eqx.nn.Sequential

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
        self.seq_len = seq_len
        self.encoder = eqx.nn.Sequential(
            [
                # eqx.nn.Linear(seq_len * num_inp, num_hidden, key=next(chain)),
                eqx.nn.Linear(num_inp, num_hidden, key=next(chain)),
                eqx.nn.LayerNorm(num_hidden, elementwise_affine=True),
            ]
        )
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"unknown activation {activation}")

        self.inner = tuple(
            eqx.nn.Sequential(
                [
                    Transformer(
                        seq_len,
                        num_hidden,
                        num_hidden,
                        0,
                        "",
                        0,
                        key=next(chain),
                    )
                    # eqx.nn.Linear(num_hidden, num_hidden, key=next(chain)),
                    # eqx.nn.LayerNorm(num_hidden, elementwise_affine=True),
                    # eqx.nn.Lambda(ACTIVATION_FUNCTIONS[activation]),
                ]
            )
            for _ in range(num_blocks)
        )
        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(num_hidden, elementwise_affine=True),
                eqx.nn.Linear(num_hidden, num_out, key=next(chain)),
            ]
        )
        self.reduce_output = reduce_output

    def __call__(
        self, input: Float[Array, "... seq_len node_dim"]
    ) -> Float[Array, "... seq_len node_dim"]:

        input = jax.vmap(self.encoder)(input)
        # input = self.encoder(input.reshape(-1))
        for res in self.inner:
            input = input + res(input)
        output = jax.vmap(self.decoder)(input)
        # output = self.decoder(input).reshape(input.shape[0], -1)
        # if self.reduce_output:
        #     output = output.sum(axis=0)
        # else:
        #     output = output.reshape(self.seq_len, -1)
        return output
