import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Float
from numpy import reshape  # type: ignore

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


from flox._src import bulk


class Conv(eqx.Module):

    encoder_kernel: Array
    inner_kernel: Array
    decoder_kernel: Array

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

        num_inner = 1
        window_size = 3
        self.encoder_kernel = jax.random.normal(
            next(chain), shape=(window_size,) * 3 + (32, num_inp)
        )
        self.inner_kernel = jax.random.normal(
            next(chain), shape=(num_inner,) + (window_size,) * 3 + (32, 32)
        )
        self.decoder_kernel = jax.random.normal(
            next(chain), shape=(window_size,) * 3 + (num_out, 32)
        )

    def __call__(self, pos: Array, input: Array):
        resolution = (8, 8, 8)
        # raise ValueError(input.shape)
        # raise ValueError(pos.shape, input.shape)
        idxs = jax.vmap(bulk.lattice_indices, in_axes=(0, None, None))(
            pos, bulk.neighbors(pos.shape[-1]), resolution
        )
        weights = jax.vmap(bulk.lattice_weights, in_axes=(0, 0, None))(
            pos, idxs, resolution
        )
        img = bulk.scatter(input, weights, idxs, resolution)
        out = bulk.conv_nd(img, self.encoder_kernel)
        for kernel in self.inner_kernel:
            res = out
            out = bulk.conv_nd(out, kernel)
            out = res + jax.nn.silu(out)
        out = bulk.conv_nd(out, self.decoder_kernel)

        out = bulk.gather(out, idxs, bulk.lattice.AggregationMethod.Nothing)
        out = out.sum(axis=1)

        out = out.reshape(input.shape[0], -1)
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
                eqx.nn.Linear(seq_len * num_inp, num_hidden, key=next(chain)),
                eqx.nn.LayerNorm(num_hidden, elementwise_affine=True),
            ]
        )
        if activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"unknown activation {activation}")

        self.inner = tuple(
            eqx.nn.Sequential(
                [
                    eqx.nn.Linear(num_hidden, num_hidden, key=next(chain)),
                    eqx.nn.LayerNorm(num_hidden, elementwise_affine=True),
                    eqx.nn.Lambda(ACTIVATION_FUNCTIONS[activation]),
                ]
            )
            for _ in range(num_blocks)
        )
        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(num_hidden, elementwise_affine=True),
                eqx.nn.Linear(num_hidden, seq_len * num_out, key=next(chain)),
            ]
        )
        self.reduce_output = reduce_output

    def __call__(
        self, input: Float[Array, "... seq_len node_dim"]
    ) -> Float[Array, "... seq_len node_dim"]:
        input = self.encoder(input.reshape(-1))
        for res in self.inner:
            input = input + res(input)
        output = self.decoder(input)
        if self.reduce_output:
            output = output.sum(axis=0)
        else:
            output = output.reshape(self.seq_len, -1)
        return output
