import contextlib
from functools import partial, wraps

import equinox as eqx
import jax
from equinox.module import Static


@contextlib.contextmanager
def jit_and_cleanup_cache(fn):
    @partial(
        jax.jit,
        static_argnames=(
            "static",
            "static_tree_def",
        ),
    )
    def jitted_call(args, kwargs, static, static_tree_def):
        args_static, kwargs_static = jax.tree_util.tree_unflatten(
            static_tree_def, static
        )
        args = eqx.combine(args, args_static)
        kwargs = eqx.combine(kwargs, kwargs_static)
        out = fn(*args, **kwargs)
        dynamic_out, static_out = eqx.partition(out, eqx.is_array)  # type: ignore
        return dynamic_out, Static(static_out)

    @wraps(fn)
    def wrapper(*args, **kwargs):
        args, args_static = eqx.partition(args, eqx.is_array)  # type: ignore
        kwargs, kwargs_static = eqx.partition(kwargs, eqx.is_array)  # type: ignore

        static = (args_static, kwargs_static)
        static, static_tree_def = jax.tree_util.tree_flatten(static)
        static = tuple(static)

        dynamic_out, static_out = jitted_call(
            args, kwargs, static, static_tree_def
        )
        return eqx.combine(dynamic_out, static_out.value)

    yield wrapper
    jitted_call._clear_cache()  # type: ignore
