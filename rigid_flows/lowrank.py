from math import prod, sqrt

import jax
import jax.numpy as jnp
from jax import Array, float0
from jax_dataclasses import pytree_dataclass

from flox.flow import Transformed

__all__ = ["LowRankFlow"]


def low_rank_matmul(us, vs, x, *, residual: bool):
    assert us.shape == vs.shape
    if residual:
        res = x
    else:
        res = jnp.zeros_like(x)
    return res + jnp.einsum("ki, kj, ...j -> ...i", us, vs, x)


def invert_and_ldj_stable(u, v, U_inv, V_inv, regularizer: float):
    u_inv = low_rank_matmul(U_inv, V_inv, u, residual=True)

    # guarantee invertibility
    v = v * jax.nn.softplus(u_inv @ v)

    ldj = jnp.log1p(u_inv @ v)
    v_inv = low_rank_matmul(V_inv, U_inv, v, residual=True)
    u_inv = -u_inv / (1 + u_inv @ v)
    return v, u_inv, v_inv, ldj


def invert_uv_stable(U, V, regularizer: float):
    k, n = U.shape
    U_inv = jnp.zeros_like(U)
    V_inv = jnp.zeros_like(U)
    ldj = 0

    def body(carry, args):
        i, u, v = args
        ldj, U_inv, V_inv = carry
        v, u_inv, v_inv, ldj_new = invert_and_ldj_stable(
            u, v, U_inv, V_inv, regularizer
        )
        ldj += ldj_new
        U_inv += jnp.eye(k)[i][:, None] * u_inv[None]
        V_inv += jnp.eye(k)[i][:, None] * v_inv[None]
        return (ldj, U_inv, V_inv), v

    idxs = jnp.arange(k)
    (ldj, U_inv, V_inv), V = jax.lax.scan(
        body, (ldj, U_inv, V_inv), (idxs, U, V)
    )

    return V, U_inv, V_inv, ldj


def low_rank_mat_mul_backward(res, grads):
    x, U, V, U_inv, V_inv = res
    grad_y, grad_ldj = grads
    grad_x = low_rank_matmul(V, U, grad_y, residual=True)
    grad_u = jnp.einsum(
        "j, ki, i -> kj", grad_y, V, x
    ) + grad_ldj * low_rank_matmul(V_inv, U_inv, V, residual=True)
    grad_v = jnp.einsum(
        "i, ki, j -> kj", grad_y, U, x
    ) + grad_ldj * low_rank_matmul(U_inv, V_inv, U, residual=True)
    return grad_x, grad_u, grad_v, None, None, None


def low_rank_mat_mul_forward(x, us, vs, us_inv, vs_inv, ldj):
    y = low_rank_matmul(us, vs, x, residual=True)
    return (y, ldj), (x, us, vs, us_inv, vs_inv)


@jax.custom_vjp
def apply_low_rank_mat_mul(x, us, vs, us_inv, vs_inv, ldj):
    return low_rank_mat_mul_forward(x, us, vs, us_inv, vs_inv, ldj)[0]


apply_low_rank_mat_mul.defvjp(
    low_rank_mat_mul_forward, low_rank_mat_mul_backward
)


@pytree_dataclass(frozen=True)
class LowRankFlow:

    us: Array
    vs: Array
    regularizer: float

    def forward(self, input: Array):
        shape = input.shape
        input = input.reshape(-1)

        us = self.us.reshape(-1, input.shape[-1])
        us = us / sqrt(prod(us.shape))
        vs = self.vs.reshape(-1, input.shape[-1])
        vs = us / sqrt(prod(us.shape))

        vs, us_inv, vs_inv, ldj = invert_uv_stable(us, vs, self.regularizer)

        output, ldj = apply_low_rank_mat_mul(input, us, vs, us_inv, vs_inv, ldj)
        output = output.reshape(shape)

        return Transformed(output, ldj)

    def inverse(self, input: Array):
        shape = input.shape
        input = input.reshape(-1)

        us = self.us.reshape(-1, input.shape[-1])
        us = us / sqrt(prod(us.shape))
        vs = self.vs.reshape(-1, input.shape[-1])
        vs = us / sqrt(prod(us.shape))

        vs, us_inv, vs_inv, ldj = invert_uv_stable(us, vs, self.regularizer)
        us, vs, us_inv, vs_inv = us_inv, vs_inv, us, vs
        ldj = -ldj

        output, ldj = apply_low_rank_mat_mul(input, us, vs, us_inv, vs_inv, ldj)
        output = output.reshape(shape)

        return Transformed(output, ldj)
