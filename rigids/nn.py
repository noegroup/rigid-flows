from collections.abc import Callable
from typing import Any

import equinox as eqx

from flox.util import key_chain

KeyArray = jax.random.PRNGKeyArray | jnp.ndarray


def dense(
    units, activation, *, key: KeyArray
) -> eqx.nn.Sequential:
    chain = key_chain(key)
    num_layers = len(units) - 1
    layers = []
    for i, (inp, out) in enumerate(zip(units[:-1], units[1:])):
        layers.append(eqx.nn.Linear(inp, out, use_bias=True, key=next(chain)))
        if i < num_layers - 1:
            if activation is not None:
                if isinstance(activation, eqx.Module):
                    layers.append(activation)
                else:
                    layers.append(eqx.nn.Lambda(activation))
    return eqx.nn.Sequential(layers)