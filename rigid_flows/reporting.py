import contextlib
import io
import itertools as it
import logging
import os
import shutil
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar, cast

import equinox as eqx
import jax
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf  # type: ignore
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from flox._src.flow.api import Inverted, Transform, Transformed, bind
from flox._src.flow.sampling import Sampler
from flox._src.util.jax import key_chain

from .data import AugmentedData
from .density import BaseDensity, KeyArray, TargetDensity
from .flow import InitialTransform, State
from .system import OpenMMEnergyModel, SimulationBox
from .utils import jit_and_cleanup_cache

logger = logging.getLogger("rigid-flows")


def _compute_quat_contour_levels(
    qi: Array,
    qj: Array,
    threshold: float = 1e-5,
    num_levels: int = 5,
    num_bins: int = 50,
) -> tuple[Array, Array]:
    h, *_ = jnp.histogram2d(
        qi,
        qj,
        density=True,
        bins=(jnp.linspace(-1, 1, num_bins + 1), jnp.linspace(-1, 1, num_bins + 1)),  # type: ignore
    )
    h = jnp.log(threshold + h)
    levels = jnp.linspace(h[h > jnp.log(1e-4)].min(), h.max(), num_levels)
    return h, levels


def _plot_quat_contour_lines(
    q: Array,
    quat_idx: int,
    dim_i: int,
    dim_j: int,
    colors: str,
    num_bins: int = 50,
):
    gx, gy = jnp.meshgrid(
        jnp.linspace(-1, 1, num_bins), jnp.linspace(-1, 1, num_bins)
    )
    h, levels = _compute_quat_contour_levels(
        q[:, quat_idx, dim_i], q[:, quat_idx, dim_j], num_bins=num_bins
    )
    return plt.contourf(
        gx,
        gy,
        h,
        levels,
        colors=colors,
        alpha=0.5,
        extend="neither",
        antialiased=False,
    )


def plot_quaternions(
    data: Array, samples: Array, prior: Array, quat_idxs: tuple[int, ...]
):
    labels = ["w", "x", "y", "z"]
    n = 0

    legend_elements = [
        Patch(facecolor="black", edgecolor="black", label="samples"),
        Patch(facecolor="red", edgecolor="red", label="data"),
        Patch(facecolor="blue", edgecolor="blue", label="base"),
    ]

    plt.suptitle(
        f"2D projections of quaternions {', '.join(map(str, quat_idxs))}"
    )

    fig = plt.figure(figsize=(6 * 2, len(quat_idxs) * 2 + 0.8))
    for i, (j, k) in it.product(quat_idxs, it.combinations(range(4), 2)):
        n = n + 1
        plt.subplot(len(quat_idxs), 6, n)

        _plot_quat_contour_lines(data, i, j, k, "red")

        _plot_quat_contour_lines(prior, i, j, k, "blue")

        _plot_quat_contour_lines(samples, i, j, k, "black")

        if n % 6 == 1:
            plt.ylabel(f"$q_{{{i}}}$")

        if (6 * len(quat_idxs) - n) < 6:
            plt.xlabel(f"{labels[j]}-{labels[k]} - proj")

        plt.xticks([])
        plt.yticks([])
        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.tight_layout()

    plt.legend(handles=legend_elements)
    return fig


def _compute_oxy_contour_levels(
    xi: Array,
    xj: Array,
    threshold: float = 1e-5,
    num_levels: int = 5,
    num_bins: int = 50,
) -> tuple[Array, ...]:
    h, *bins = jnp.histogram2d(xi, xj, density=True, bins=num_bins + 1)
    h = jnp.log(threshold + h)
    levels = jnp.linspace(h[h > jnp.log(1e-4)].min(), h.max(), num_levels)
    return h, levels, *bins


def _plot_oxy_contour_lines(
    p: Array, box: SimulationBox, dim_i: int, dim_j: int, colors: str
):
    num_bins = 50
    gx, gy = jnp.meshgrid(
        jnp.linspace(box.min[dim_i], box.max[dim_i], num_bins + 1),
        jnp.linspace(box.min[dim_j], box.max[dim_j], num_bins + 1),
    )
    h, levels, *_ = _compute_oxy_contour_levels(
        p[:, :, dim_i].reshape(-1) % box.size[dim_i],
        p[:, :, dim_j].reshape(-1) % box.size[dim_j],
    )
    return plt.contourf(
        gx,
        gy,
        h,
        levels,
        colors=colors,
        alpha=0.3,
        extend="neither",
        antialiased=False,
    )


def plot_oxygen_positions(samples: Array, data: Array, box: SimulationBox):
    labels = ("x", "y", "z")

    legend_elements = [
        Patch(facecolor="black", edgecolor="black", label="samples"),
        Patch(facecolor="red", edgecolor="red", label="data"),
    ]

    fig = plt.figure(figsize=(3 * 3, 1 * 3))
    plt.suptitle("projection of oxygen positions")
    for k, (i, j) in enumerate(it.combinations(range(3), 2), start=1):

        plt.subplot(1, 3, k)
        _plot_oxy_contour_lines(samples, box, i, j, "black")
        _plot_oxy_contour_lines(data, box, i, j, "red")
        plt.legend()
        plt.ylabel(labels[j])
        plt.xlabel(labels[i])
    plt.legend(handles=legend_elements)
    return fig


def write_figure_to_tensorboard(label: str, fig: Figure, num_iter: int):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    # plt.savefig(buf, format='png')
    buf.seek(0)

    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    summary_op = tf.summary.image(label, image, num_iter)
    return buf


def compute_energies(
    pos: np.ndarray, box: SimulationBox, model: OpenMMEnergyModel
):
    return model.compute_energies_and_forces(
        pos.reshape(pos.shape[0], -1, 3), np.diag(box.size)
    )


def plot_energy_histogram(
    energies_data: np.ndarray,
    energies_model: np.ndarray,
    weights: np.ndarray | None = None,
    num_stds=15.0,
    num_bins=100,
):
    assert len(energies_data.shape) == 1
    assert len(energies_model.shape) == 1
    assert weights is None or len(weights.shape) == 1

    data_mean = np.mean(energies_data)
    data_std = np.std(energies_data)
    bins = tuple(
        x
        for x in np.linspace(
            data_mean - num_stds * data_std,
            data_mean + num_stds * data_std,
            num_bins,
        )
    )
    fig = plt.figure()
    plt.title(f"distribution of energies")
    plt.hist(
        np.array(energies_data),
        bins=bins,
        histtype="step",
        density=True,
        label="data",
    )
    plt.hist(
        np.array(energies_model),
        bins=bins,
        histtype="step",
        density=True,
        label="model",
    )
    if weights is not None:
        plt.hist(
            np.array(energies_model),
            bins=bins,
            histtype="step",
            weights=np.array(weights),
            density=True,
            label="model + weights",
        )
    plt.xlabel(f"u(x)")
    plt.legend()
    return fig


def compute_stable_weights(diff):
    logZ = np.logaddexp.reduce(diff)
    weights = np.exp(diff - logZ)
    return weights


@pytree_dataclass(frozen=True)
class SamplingStatistics:
    weights: np.ndarray

    omm_energies: np.ndarray
    aux_energies: np.ndarray
    model_energies: np.ndarray

    ess: float
    num: int


T = TypeVar("T")


def batched_sampler(
    sample: Sampler[T], num_samples_tot: int, num_samples_per_batch: int
) -> Callable[[KeyArray], Transformed[T] | None]:
    def accum(accum: Transformed[T], new: Transformed[T]) -> Transformed[T]:
        return jax.tree_map(lambda a, b: jnp.concatenate([a, b]), accum, new)

    def sample_batched(key: KeyArray) -> Transformed[T] | None:
        chain = key_chain(key)
        samples = None
        num_left = num_samples_tot
        while num_left > 0:
            batch_size = min(num_samples_per_batch, num_left)
            batch = jax.vmap(sample)(
                jax.random.split(
                    next(chain),
                    batch_size,
                )
            )
            num_left -= batch_size
            if samples is None:
                samples = batch
            else:
                samples = accum(samples, batch)
        return samples

    return sample_batched


def sample_from_model(
    key: KeyArray, base: BaseDensity, flow: Transform[AugmentedData, State]
) -> Transformed[AugmentedData]:
    latent = base.sample(key)
    sample = bind(latent, Inverted(flow))
    return sample


def sample_from_target(
    key: KeyArray,
    target: TargetDensity,
) -> Transformed[AugmentedData]:
    return target.sample(key)


def sample_from_base(key: KeyArray, base: BaseDensity) -> Transformed[State]:
    return base.sample(key)


def compute_sampling_statistics(
    samples: Transformed[AugmentedData],
    target: TargetDensity,
) -> SamplingStatistics:
    aux_energies = -target.aux_model.log_prob(samples.obj.aux)
    aux_energies = aux_energies.reshape(aux_energies.shape[0], -1).sum(axis=-1)

    sample_positions = np.array(samples.obj.pos).reshape(
        samples.obj.pos.shape[0], -1, 3
    )
    box = np.diag(samples.obj.box.size[0])
    omm_energies, _ = target.model.compute_energies_and_forces(
        sample_positions, box
    )
    target_energies = omm_energies + aux_energies
    model_energies = np.array(samples.ldj)

    weights = compute_stable_weights(model_energies - target_energies)

    ess = np.square(np.sum(weights)) / np.sum(np.square(weights))

    return SamplingStatistics(
        weights=np.array(weights),
        omm_energies=np.array(omm_energies),
        aux_energies=np.array(aux_energies),
        model_energies=np.array(model_energies),
        ess=float(ess),
        num=len(omm_energies),
    )


def save_summary(path: str, data: Any):
    np.savez(path, **asdict(data))


@pytree_dataclass(frozen=True)
class ReportingSpecifications:
    num_samples: int
    num_samples_per_batch: int
    plot_quaternions: tuple[int, ...] | None
    plot_oxygens: bool
    plot_energy_histograms: bool
    report_ess: bool
    save_model: bool
    save_samples: bool
    save_statistics: bool


@pytree_dataclass(frozen=True)
class Reporter:

    base: BaseDensity
    target: TargetDensity
    run_dir: str
    specs: ReportingSpecifications
    scope: str | None

    def report_model(
        self,
        key: KeyArray,
        flow: Transform[AugmentedData, State],
        num_iter: int,
    ):
        return report_model(
            key,
            flow,
            self.base,
            self.target,
            num_iter,
            self.run_dir,
            self.scope if self.scope else "",
            self.specs,
        )

    def with_scope(self, scope) -> "Reporter":
        return Reporter(
            self.base,
            self.target,
            self.run_dir,
            self.specs,
            self.scope + "/" + scope if self.scope else scope,
        )


def report_model(
    key: KeyArray,
    flow: Transform[AugmentedData, State],
    base: BaseDensity,
    target: TargetDensity,
    num_iter: int,
    run_dir: str,
    scope: str,
    specs: ReportingSpecifications,
):
    chain = key_chain(key)

    logger.info("preparing report")

    logger.info("sampling from data")
    with jit_and_cleanup_cache(
        batched_sampler(
            partial(sample_from_target, target=target),
            specs.num_samples,
            specs.num_samples_per_batch,
        )
    ) as sample:
        data_samples = sample(next(chain))
    assert data_samples is not None

    logger.info("sampling from prior")
    with jit_and_cleanup_cache(
        batched_sampler(
            partial(sample_from_base, base=base),
            specs.num_samples,
            specs.num_samples_per_batch,
        )
    ) as sample:
        prior_samples = sample(next(chain))
    assert prior_samples is not None

    logger.info("sampling from model")
    with jit_and_cleanup_cache(
        batched_sampler(
            partial(sample_from_model, base=base, flow=flow),
            specs.num_samples,
            specs.num_samples_per_batch,
        )
    ) as sample:
        model_samples = sample(next(chain))
    assert model_samples is not None

    stats = compute_sampling_statistics(model_samples, target)

    artifact_path = f"{run_dir}/{scope}"
    Path(artifact_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving artifacts to {artifact_path}")

    # save model
    if specs.save_model:
        path = f"{artifact_path}/model.eqx"
        logger.info(f"Saving model to {path}")
        eqx.tree_serialise_leaves(path, flow)

    # save model samples
    if specs.save_samples:
        path = f"{artifact_path}/samples.npz"
        logger.info(f"Saving samples to {path}")
        save_summary(path, model_samples)

    # save sample statistics
    if specs.save_statistics:
        path = f"{artifact_path}/stats.npz"
        save_summary(path, stats)
        logger.info(f"Saving statistics to {path}")

    # plot quaternion histograms
    if specs.plot_quaternions is not None:
        data_quats = jax.vmap(InitialTransform().forward)(
            data_samples.obj
        ).obj.rot
        model_quats = jax.vmap(InitialTransform().forward)(
            model_samples.obj
        ).obj.rot
        prior_quats = prior_samples.obj.rot
        logger.info(f"plotting quaternions")
        fig = plot_quaternions(
            data_quats, model_quats, prior_quats, specs.plot_quaternions
        )
        write_figure_to_tensorboard(f"{scope}/plots/quaternions", fig, num_iter)

    # plot oxygen histograms
    if specs.plot_oxygens:
        data_pos = jax.vmap(InitialTransform().forward)(
            data_samples.obj
        ).obj.pos
        model_pos = jax.vmap(InitialTransform().forward)(
            model_samples.obj
        ).obj.pos
        logger.info(f"plotting oxygens")
        fig = plot_oxygen_positions(model_pos, data_pos, target.box)
        write_figure_to_tensorboard(f"{scope}/plots/oxygens", fig, num_iter)

    # report ESS
    if specs.report_ess:
        logger.info(f"reporting ESS = {stats.ess}")
        tf.summary.scalar(f"{scope}/metrics/ess", stats.ess, num_iter)

    # plot energy histograms
    if specs.plot_energy_histograms:
        logger.info(f"plotting energy histogram")
        fig = plot_energy_histogram(
            np.array(data_samples.ldj),
            stats.model_energies,
            stats.weights,
        )
        write_figure_to_tensorboard(f"{scope}/plots/energies", fig, num_iter)