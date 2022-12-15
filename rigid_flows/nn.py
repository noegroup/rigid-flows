import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float  # type: ignore

from flox._src.nn.modules import dense
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


class Transformer(eqx.Module):
    """Standard impl of a transformer according to the `attention is all you need` paper."""

    attention_1: eqx.nn.MultiheadAttention
    attention_2: eqx.nn.MultiheadAttention

    norm_1: eqx.nn.LayerNorm
    norm_2: eqx.nn.LayerNorm
    norm_3: eqx.nn.LayerNorm

    dense: eqx.nn.Sequential

    def __init__(
        self, num_heads: int, num_dims: int, num_hidden: int, *, key: KeyArray
    ):
        """Standard impl of a transformer according to the `attention is all you need` paper."

        Args:
            num_heads (int): number of transformer heads
            num_dims (int): node dimensionality
            num_hidden (int): hidden dimension of final dense layer
            key (KeyArray): PRNG Key for layer initialization
        """

        chain = key_chain(key)

        self.attention_1 = eqx.nn.MultiheadAttention(
            num_heads,
            num_dims * num_heads,
            use_key_bias=True,
            use_query_bias=False,
            use_output_bias=True,
            key=next(chain),
        )

        self.attention_2 = eqx.nn.MultiheadAttention(
            num_heads,
            num_dims * num_heads,
            use_key_bias=True,
            use_query_bias=False,
            use_output_bias=True,
            key=next(chain),
        )

        self.norm_1 = eqx.nn.LayerNorm(
            shape=(num_dims * num_heads), elementwise_affine=True
        )

        self.norm_2 = eqx.nn.LayerNorm(
            shape=(num_dims * num_heads), elementwise_affine=True
        )

        self.norm_3 = eqx.nn.LayerNorm(
            shape=(num_dims * num_heads), elementwise_affine=True
        )

        self.dense = dense(
            (num_dims * num_heads, num_hidden, num_dims * num_heads),
            jax.nn.silu,
            key=next(chain),
        )

    def __call__(
        self, input: Float[Array, "... seq_len node_dim"]
    ) -> Float[Array, "... seq_len node_dim"]:

        input += self.attention_1(input, input, input)
        input = jax.vmap(self.norm_1)(input)

        input += self.attention_2(input, input, input)
        input = jax.vmap(self.norm_2)(input)

        input += jax.vmap(self.dense)(input)
        input = jax.vmap(self.norm_3)(input)

        return input


class TransformerStack(eqx.Module):
    """Stack of transformer layers.

    DISCLAIMER: right now only implements a simple dense net!!!!
    """

    # encoder: eqx.nn.Linear
    # decoder: eqx.nn.Linear

    # transformers: tuple[Transformer]

    reduce_output: bool

    # use simple dense net for now as transformers don't work (yet)
    foo: eqx.nn.Sequential

    def __init__(
        self,
        num_inp: int,
        num_out: int,
        num_heads: int,
        num_dims: int,
        num_hidden: int,
        num_blocks: int = 0,
        reduce_output: bool = False,
        *,
        key: KeyArray
    ):
        chain = key_chain(key)
        # self.encoder = eqx.nn.Linear(
        #     num_inp, num_heads * num_dims, key=next(chain)
        # )
        # self.transformers = tuple(
        #     Transformer(num_heads, num_dims, num_hidden, key=next(chain))
        #     for _ in range(num_blocks)
        # )
        # self.decoder = eqx.nn.Linear(
        #     num_heads * num_dims, num_out, key=next(chain)
        # )

        self.foo = eqx.nn.Sequential(
            [
                eqx.nn.Linear(num_inp * 16, num_hidden, key=next(chain)),
                eqx.nn.LayerNorm((num_hidden,), elementwise_affine=True),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(num_hidden, num_hidden, key=next(chain)),
                eqx.nn.LayerNorm((num_hidden,), elementwise_affine=True),
                eqx.nn.Lambda(jax.nn.silu),
                eqx.nn.Linear(num_hidden, num_out * 16, key=next(chain)),
            ]
        )
        self.reduce_output = reduce_output

    def __call__(
        self, input: Float[Array, "... seq_len node_dim"]
    ) -> Float[Array, "... seq_len node_dim"]:
        out = self.foo(input.reshape(-1))
        if self.reduce_output:
            out = out.sum(axis=0)
        else:
            out = out.reshape(16, -1)
        return out
        input = jax.vmap(self.encoder)(input)
        for transformer in self.transformers:
            input = transformer(input)

        return jax.vmap(self.decoder)(input)
