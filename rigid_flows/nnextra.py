import equinox as eqx
import jax
import jax.numpy as jnp
from flox.util import key_chain

from .system import QUATERNION_DIM, SPATIAL_DIM


class PositionConditionerBlock(eqx.Module):

    nodes_kq: eqx.nn.Linear
    aux_kq: eqx.nn.Linear | None
    rot_kq: eqx.nn.Linear
    values: eqx.nn.Linear
    num_heads: int

    def __init__(
        self,
        node_dim: int,
        num_aux: int | None,
        num_heads: int,
        num_channels: int,
        key,
    ) -> None:
        chain = key_chain(key)
        self.num_heads = num_heads

        self.nodes_kq = eqx.nn.Linear(
            node_dim, num_heads * num_channels * 2, key=next(chain)
        )
        if num_aux is not None:
            self.aux_kq = eqx.nn.Linear(
                num_aux, num_heads * num_channels * 2, key=next(chain)
            )
        else:
            self.aux_kq = None
        self.rot_kq = eqx.nn.Linear(
            QUATERNION_DIM,
            num_heads * num_channels * 2,
            use_bias=False,
            key=next(chain),
        )
        n_types = 2 if num_aux is None else 3
        self.values = eqx.nn.Linear(
            node_dim, node_dim * num_heads * n_types, key=next(chain)
        )

    def __call__(self, nodes, aux, rot):
        seq_len = nodes.shape[0]  # same as node_dim
        n_types = 2 if aux is None else 3

        val = jax.vmap(self.values)(nodes).reshape(
            seq_len, n_types * self.num_heads, -1
        )

        nodes_k, nodes_q = jnp.split(
            jax.vmap(self.nodes_kq)(nodes).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )
        if aux is not None:
            aux_k, aux_q = jnp.split(
                jax.vmap(self.aux_kq)(aux).reshape(seq_len, self.num_heads, -1),
                2,
                -1,
            )
        rot_k, rot_q = jnp.split(
            jax.vmap(self.rot_kq)(rot).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )

        if aux is not None:
            k = jnp.concatenate([nodes_k, aux_k, rot_k], axis=-2)
            q = jnp.concatenate([nodes_q, aux_q, rot_q], axis=-2)
        else:
            k = jnp.concatenate([nodes_k, rot_k], axis=-2)
            q = jnp.concatenate([nodes_q, rot_q], axis=-2)

        logits = jnp.einsum("ihc, jhc -> ijh", k, q)
        logits = logits / jnp.sqrt(self.num_heads)

        att = jax.nn.softmax(logits, axis=-2)
        out = jnp.einsum("ijh, jhd -> id", att, val)

        return out


class PosConditioner(eqx.Module):

    blocks: list[tuple[PositionConditionerBlock, eqx.nn.LayerNorm]]
    decoder: eqx.nn.Linear

    def __init__(
        self, seq_len, out, num_aux, num_heads, num_channels, num_blocks, *, key
    ) -> None:
        chain = key_chain(key)
        self.blocks = [
            (
                PositionConditionerBlock(
                    seq_len, num_aux, num_heads, num_channels, key=next(chain)
                ),
                # eqx.nn.MultiheadAttention(num_heads, seq_len*num_channels, ???, key=key),
                eqx.nn.LayerNorm((seq_len,), elementwise_affine=True),
            )
            for _ in range(num_blocks)
        ]

        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm((seq_len,), elementwise_affine=True),
                eqx.nn.Linear(seq_len, out, use_bias=True, key=next(chain)),
            ]
        )

    def __call__(self, aux, rot):
        seq_len = rot.shape[0]
        nodes = jnp.eye(seq_len)
        for block, norm in self.blocks:
            nodes = nodes + jax.vmap(norm)(block(nodes, aux, rot))
        return jax.vmap(self.decoder)(nodes)


class AuxiliaryConditionerBlock(eqx.Module):

    nodes_kq: eqx.nn.Linear
    pos_kq: eqx.nn.Linear
    rot_kq: eqx.nn.Linear
    values: eqx.nn.Linear
    num_heads: int

    def __init__(
        self, node_dim: int, num_heads: int, num_channels: int, *, key
    ) -> None:
        chain = key_chain(key)
        self.num_heads = num_heads
        self.nodes_kq = eqx.nn.Linear(
            node_dim, num_heads * num_channels * 2, key=next(chain)
        )
        self.pos_kq = eqx.nn.Linear(
            2 * SPATIAL_DIM, num_heads * num_channels * 2, key=next(chain)
        )
        self.rot_kq = eqx.nn.Linear(
            QUATERNION_DIM,
            num_heads * num_channels * 2,
            use_bias=False,
            key=next(chain),
        )
        self.values = eqx.nn.Linear(node_dim, node_dim * num_heads * 3, key=next(chain))

    def __call__(self, nodes, pos, rot):
        seq_len = nodes.shape[0]

        val = jax.vmap(self.values)(nodes).reshape(seq_len, 3 * self.num_heads, -1)

        nodes_k, nodes_q = jnp.split(
            jax.vmap(self.nodes_kq)(nodes).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )
        pos_k, pos_q = jnp.split(
            jax.vmap(self.pos_kq)(pos).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )

        rot_k, rot_q = jnp.split(
            jax.vmap(self.rot_kq)(rot).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )

        k = jnp.concatenate([nodes_k, pos_k, rot_k], axis=-2)
        q = jnp.concatenate([nodes_q, pos_q, rot_q], axis=-2)

        logits = jnp.einsum("ihc, jhc -> ijh", k, q)
        logits = logits / jnp.sqrt(self.num_heads)
        att = jax.nn.softmax(logits, axis=-2)
        out = jnp.einsum("ijh, jhd -> id", att, val)

        return out


class AuxConditioner(eqx.Module):

    blocks: list[tuple[AuxiliaryConditionerBlock, eqx.nn.LayerNorm]]
    decoder: eqx.nn.Linear

    def __init__(self, seq_len, out, num_heads, num_channels, num_blocks, *, key) -> None:
        chain = key_chain(key)
        self.blocks = [
            (
                AuxiliaryConditionerBlock(
                    seq_len, num_heads, num_channels, key=next(chain)
                ),
                eqx.nn.LayerNorm((seq_len,), elementwise_affine=True),
            )
            for _ in range(num_blocks)
        ]
        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm((seq_len,), elementwise_affine=True),
                eqx.nn.Linear(seq_len, out, use_bias=True, key=next(chain)),
            ]
        )

    def __call__(self, pos, rot):
        seq_len = pos.shape[0]
        nodes = jnp.eye(seq_len)
        for block, norm in self.blocks:
            nodes = nodes + jax.vmap(norm)(block(nodes, pos, rot))
        return jax.vmap(self.decoder)(nodes)


class RotationConditionerBlock(eqx.Module):

    nodes_kq: eqx.nn.Linear
    pos_kq: eqx.nn.Linear
    aux_kq: eqx.nn.Linear | None
    values: eqx.nn.Linear
    num_heads: int

    def __init__(
        self, node_dim: int, num_aux: int | None, num_heads: int, num_channels: int, key
    ) -> None:
        chain = key_chain(key)
        self.num_heads = num_heads
        self.nodes_kq = eqx.nn.Linear(
            node_dim, num_heads * num_channels * 2, key=next(chain)
        )
        self.pos_kq = eqx.nn.Linear(
            2 * SPATIAL_DIM, num_heads * num_channels * 2, key=next(chain)
        )
        if num_aux is not None:
            self.aux_kq = eqx.nn.Linear(
                num_aux,
                num_heads * num_channels * 2,
                use_bias=False,
                key=next(chain),
            )
        else:
            self.aux_kq = None
        n_types = 2 if num_aux is None else 3
        self.values = eqx.nn.Linear(
            node_dim, node_dim * num_heads * n_types, key=next(chain)
        )

    def __call__(self, nodes, pos, aux):
        seq_len = nodes.shape[0]
        n_types = 2 if aux is None else 3

        val = jax.vmap(self.values)(nodes).reshape(
            seq_len, n_types * self.num_heads, -1
        )

        nodes_k, nodes_q = jnp.split(
            jax.vmap(self.nodes_kq)(nodes).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )
        pos_k, pos_q = jnp.split(
            jax.vmap(self.pos_kq)(pos).reshape(seq_len, self.num_heads, -1),
            2,
            -1,
        )

        if aux is not None:
            aux_k, aux_q = jnp.split(
                jax.vmap(self.aux_kq)(aux).reshape(seq_len, self.num_heads, -1),
                2,
                -1,
            )

            k = jnp.concatenate([nodes_k, pos_k, aux_k], axis=-2)
            q = jnp.concatenate([nodes_q, pos_q, aux_q], axis=-2)
        else:
            k = jnp.concatenate([nodes_k, pos_k], axis=-2)
            q = jnp.concatenate([nodes_q, pos_q], axis=-2)

        logits = jnp.einsum("ihc, jhc -> ijh", k, q)
        logits = logits / jnp.sqrt(self.num_heads)

        att = jax.nn.softmax(logits, axis=-2)
        out = jnp.einsum("ijh, jhd -> id", att, val)

        return out


class RotConditioner(eqx.Module):

    blocks: list[tuple[RotationConditionerBlock, eqx.nn.LayerNorm]]
    decoder: eqx.nn.Sequential

    def __init__(
        self, seq_len, out, num_aux, num_heads, num_channels, num_blocks, key
    ) -> None:
        chain = key_chain(key)
        self.blocks = [
            (
                RotationConditionerBlock(
                    seq_len, num_aux, num_heads, num_channels, key=next(chain)
                ),
                eqx.nn.LayerNorm((seq_len,), elementwise_affine=True),
            )
            for _ in range(num_blocks)
        ]
        self.decoder = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm((seq_len,), elementwise_affine=True),
                eqx.nn.Linear(seq_len, out, use_bias=True, key=next(chain)),
            ]
        )

    def __call__(self, pos, aux):
        seq_len = pos.shape[0]
        nodes = jnp.eye(seq_len)
        for block, norm in self.blocks:
            nodes = nodes + jax.vmap(norm)(block(nodes, pos, aux))
        return jax.vmap(self.decoder)(nodes)
