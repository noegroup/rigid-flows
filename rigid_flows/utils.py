import contextlib
from collections.abc import Callable, Generator
from functools import partial, wraps
from typing import Any, Mapping, ParamSpec, TypeVar, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from equinox.module import Static

R = TypeVar("R")
P = ParamSpec("P")


@contextlib.contextmanager
def jit_and_cleanup_cache(
    fn: Callable[P, R]
) -> Generator[Callable[P, R], None, None]:
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
        kwargs = cast(Mapping, eqx.combine(kwargs, kwargs_static))
        out = fn(*args, **kwargs)  # type: ignore
        dynamic_out, static_out = eqx.partition(out, eqx.is_array)  # type: ignore
        return dynamic_out, Static(static_out)

    @wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        args, args_static = eqx.partition(args, eqx.is_array)  # type: ignore
        kwargs, kwargs_static = eqx.partition(kwargs, eqx.is_array)  # type: ignore

        static = (args_static, kwargs_static)
        static, static_tree_def = jax.tree_util.tree_flatten(static)
        static = tuple(static)

        dynamic_out, static_out = jitted_call(
            args, kwargs, static, static_tree_def
        )
        return cast(R, eqx.combine(dynamic_out, static_out.value))

    yield wrapper
    jitted_call.clear_cache()  # type: ignore


def scanned_vmap(
    fn,
    batch_size: int,
    in_axes: Any = 0,
    out_axes: Any = 0,
    *vmap_args,
    **vmap_kwargs,
):
    """Batched version of vmap avoiding OOM errors for GPU-resources available to mortals.

       This function considers the vmap dimension(s) of the input arguments to be of size

            num_batches * batch_size + leftover

        where

            num_batches = size // batch_size
            leftover = num_batches * batch_size - num_batches

        It then splits the input into two chunks of size num_batchs * batch_size and leftover

            - the first one is vmapped iteratively using jax.lax.scan
            - the second one is vmapped in the usual way

        Finally it combines both outputs into one that is coherent with jax.vmap's output.

    Args:
        fn: vmapped function
        batch_size: maximum size of a batch that is processed in parallel
        in_axes: see vmap doc
        out_axes: see vmap doc
        vmap_args: args passed to vmap
        vmap_kwargs: kwargs passed to vmap
    Returns:
        PyTree: see jax.vmap
    """

    @wraps(fn)
    def wrapper(*args):

        # make sure we have a full tree with in_axes
        args_tree = jax.tree_util.tree_structure(args)
        in_axes_ = in_axes
        if jax.tree_util.tree_structure(in_axes_) != args_tree:
            in_axes_ = jax.tree_util.tree_unflatten(
                args_tree, (in_axes_,) * args_tree.num_leaves
            )

        # filter args tree to contain only args that are vmapped
        mapped_args = eqx.filter(
            args, jax.tree_map(lambda x: x is not None, in_axes_)
        )

        # compute the number of elements in the super batch
        num_inputs = jax.tree_util.tree_flatten(
            jax.tree_map(lambda arg, ax: arg.shape[ax], mapped_args, in_axes_)
        )[0][0]
        num_batches = num_inputs // batch_size

        def slice_into_batches(arg, ax, num_batches, batch_size):
            """slices the super batch into chunks of size [..., num_batches, batch_size, ...]"""
            if ax is None:
                return None
            num_scanned = num_batches * batch_size
            shape = arg.shape
            scanned_slice = (slice(None),) * ax + (
                slice(None, num_scanned, None),
            )
            scanned_arg = arg[scanned_slice]
            scanned_arg = scanned_arg.reshape(
                *shape[:ax], num_batches, batch_size, *shape[ax + 1 :]
            )
            scanned_arg = jnp.swapaxes(scanned_arg, 0, ax)  # type: ignore
            return scanned_arg

        def slice_off_leftover(arg, ax, num_batches, batch_size):
            """slices off the leftover batch that does not need scanning over"""
            num_scanned = num_batches * batch_size
            leftover_slice = (slice(None),) * ax + (
                slice(num_scanned, None, None),
            )
            leftover_arg = arg[leftover_slice]
            return leftover_arg

        # arguments that are scanned over
        scanned_args = jax.tree_map(
            partial(
                slice_into_batches,
                num_batches=num_batches,
                batch_size=batch_size,
            ),
            mapped_args,
            in_axes_,
        )
        # arguments that are processed as usual
        leftover_args = jax.tree_map(
            partial(
                slice_off_leftover,
                num_batches=num_batches,
                batch_size=batch_size,
            ),
            mapped_args,
            in_axes_,
        )

        def scan_body(_, scan_args):
            merged = eqx.combine(scan_args, args)
            out = jax.vmap(
                fn,
                in_axes=in_axes_,
                out_axes=out_axes,
                *vmap_args,
                **vmap_kwargs,
            )(*merged)
            return _, out

        # result of scanning over batches
        _, out_scanned = jax.lax.scan(scan_body, init=None, xs=scanned_args)

        # usual result
        out_leftover = jax.vmap(
            fn, in_axes=in_axes_, out_axes=out_axes, *vmap_args, **vmap_kwargs
        )(*eqx.combine(leftover_args, args))

        def merge_outputs(out_scanned, out_leftover, ax):
            """merges scanned and leftover outputs"""
            if ax is None:
                return None
            out_scanned = jnp.swapaxes(out_scanned, 0, ax)
            shape = out_scanned.shape
            out_scanned = out_scanned.reshape(*shape[:ax], -1, *shape[ax + 2 :])
            out = jnp.concatenate([out_scanned, out_leftover], axis=ax)
            return out

        out_tree = jax.tree_util.tree_structure(out_leftover)
        out_axes_ = out_axes
        if jax.tree_util.tree_structure(out_axes_) != out_tree:
            out_axes_ = jax.tree_util.tree_unflatten(
                out_tree, (out_axes_,) * out_tree.num_leaves
            )

        out = jax.tree_map(
            merge_outputs,
            out_scanned,
            out_leftover,
            out_axes_,
        )
        return out

    return wrapper
