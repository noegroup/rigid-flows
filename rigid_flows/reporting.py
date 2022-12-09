import io
import itertools as it
import json
import logging
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any, Callable, TypeVar

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
from flox._src.util.misc import unpack

from .data import AugmentedData
from .density import BaseDensity, KeyArray, TargetDensity
from .flow import InitialTransform, State
from .specs import ReportingSpecifications
from .system import OpenMMEnergyModel, SimulationBox
from .utils import jit_and_cleanup_cache, scanned_vmap


def pretty_json(hp):
    json_hp = json.dumps(hp, indent=2)
    return "".join("\t" + line for line in json_hp.splitlines(True))


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
    h_filtered = h[h > jnp.log(1e-4)]
    levels = jnp.linspace(h_filtered.min(), h_filtered.max(), num_levels)
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
    h_filtered = h[h > jnp.log(1e-4)]
    levels = jnp.linspace(h_filtered.min(), h_filtered.max(), num_levels)
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

        ax = plt.subplot(1, 3, k)
        _plot_oxy_contour_lines(samples, box, i, j, "black")
        _plot_oxy_contour_lines(data, box, i, j, "red")
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
    label: str,
    energies_data: np.ndarray,
    energies_model: np.ndarray,
    weights: np.ndarray | None = None,
    num_stds=5.0,
    num_bins=100,
):
    assert len(energies_data.shape) == 1
    assert len(energies_model.shape) == 1
    assert weights is None or len(weights.shape) == 1

    data_mean = np.mean(energies_data)
    data_std = np.std(energies_data)
    model_min = np.min(energies_model)

    min_val = data_mean - num_stds * data_std
    max_val = np.maximum(
        model_min + num_stds * data_std, data_mean + num_stds * data_std
    )
    if np.isnan(max_val) or np.isinf(max_val) or max_val > 1e6:
        max_val = data_mean + num_stds * data_std

    bins = tuple(
        x
        for x in np.linspace(
            min_val,
            max_val,
            num_bins,
        )
    )
    fig = plt.figure()
    plt.title(f"{label}")
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
    out = target.sample(key)
    return out


def sample_from_base(key: KeyArray, base: BaseDensity) -> Transformed[State]:
    return base.sample(key)


def compute_model_likelihood(
    samples: Transformed[AugmentedData],
    flow: Transform[AugmentedData, State],
    base: BaseDensity,
):
    latent, ldj = unpack(flow.forward(samples.obj))
    return base.potential(latent) - ldj


def compute_sample_energies(
    samples: Transformed[AugmentedData],
    target: TargetDensity,
):
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
    return omm_energies, aux_energies


def compute_sampling_statistics(
    samples: Transformed[AugmentedData],
    target: TargetDensity,
) -> SamplingStatistics:

    omm_energies, aux_energies = compute_sample_energies(samples, target)

    model_energies = np.array(samples.ldj)

    target_energies = omm_energies + aux_energies
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
    np.savez_compressed(path, **asdict(data))


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
    tot_iter: int,
    run_dir: str,
    scope: str,
    specs: ReportingSpecifications,
):
    chain = key_chain(key)

    logger = logging.getLogger("main")

    logging.info("preparing report")

    logging.info("sampling from data")
    with jit_and_cleanup_cache(
        scanned_vmap(
            partial(sample_from_target, target=target),
            specs.num_samples_per_batch,
        )
    ) as sample:
        data_samples = sample(jax.random.split(next(chain), specs.num_samples))
    assert data_samples is not None

    logging.info("sampling from prior")
    with jit_and_cleanup_cache(
        scanned_vmap(
            partial(sample_from_base, base=base),
            specs.num_samples_per_batch,
        )
    ) as sample:
        prior_samples = sample(jax.random.split(next(chain), specs.num_samples))
    assert prior_samples is not None

    logging.info("sampling from model")
    with jit_and_cleanup_cache(
        scanned_vmap(
            partial(sample_from_model, base=base, flow=flow),
            specs.num_samples_per_batch,
        )
    ) as sample:
        model_samples = sample(jax.random.split(next(chain), specs.num_samples))
    assert model_samples is not None

    stats = compute_sampling_statistics(model_samples, target)

    artifact_path = f"{run_dir}/{scope}"
    Path(artifact_path).mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving artifacts to {artifact_path}")

    # save model
    if specs.save_model:
        path = f"{artifact_path}/model.eqx"
        logging.info(f"Saving model to {path}")
        eqx.tree_serialise_leaves(path, flow)

    # save model samples
    if specs.save_samples:
        path = f"{artifact_path}/samples.npz"
        logging.info(f"Saving samples to {path}")
        save_summary(path, model_samples)

    # save sample statistics
    if specs.save_statistics:
        path = f"{artifact_path}/stats.npz"
        save_summary(path, stats)
        logging.info(f"Saving statistics to {path}")

    # plot quaternion histograms
    if specs.plot_quaternions is not None:
        data_quats = jax.vmap(InitialTransform().forward)(
            data_samples.obj
        ).obj.rot
        model_quats = jax.vmap(InitialTransform().forward)(
            model_samples.obj
        ).obj.rot
        prior_quats = prior_samples.obj.rot
        logging.info(f"plotting quaternions")
        fig = plot_quaternions(
            data_quats, model_quats, prior_quats, specs.plot_quaternions
        )
        write_figure_to_tensorboard(f"{scope}/plots/quaternions", fig, tot_iter)

    # plot oxygen histograms
    if specs.plot_oxygens:
        data_pos = jax.vmap(InitialTransform().forward)(
            data_samples.obj
        ).obj.pos
        model_pos = jax.vmap(InitialTransform().forward)(
            model_samples.obj
        ).obj.pos
        logging.info(f"plotting oxygens")
        fig = plot_oxygen_positions(model_pos, data_pos, target.box)
        write_figure_to_tensorboard(f"{scope}/plots/oxygens", fig, tot_iter)

    # report ESS
    if specs.report_ess:
        logging.info(f"reporting ESS = {stats.ess}")
        tf.summary.scalar(f"/metrics/ess", stats.ess, tot_iter)
        log_weights = jnp.log(stats.weights)
        log_weights = log_weights[
            (~jnp.isnan(log_weights)) & (~jnp.isinf(log_weights))
        ]
        tf.summary.histogram("/metrics/log_weights", log_weights, step=tot_iter)

    # report NLL
    if specs.report_likelihood:
        logging.info(f"reporting likelihood")
        with jit_and_cleanup_cache(
            scanned_vmap(
                partial(compute_model_likelihood, base=base, flow=flow),
                specs.num_samples_per_batch,
            )
        ) as eval_likelihood:
            data_likelihood = eval_likelihood(data_samples)
            model_likelihood = eval_likelihood(model_samples)
            tf.summary.scalar(
                f"/metrics/likelihood/data", jnp.mean(data_likelihood), tot_iter
            )
            tf.summary.scalar(
                f"/metrics/likelihood/model",
                jnp.mean(model_likelihood),
                tot_iter,
            )
            tf.summary.histogram(
                "/metrics/energies/model/data", data_likelihood, step=tot_iter
            )
            tf.summary.histogram(
                "metrics/energies/model/model", model_likelihood, step=tot_iter
            )

    # plot energy histograms
    if specs.plot_energy_histograms:
        logging.info(f"plotting energy histogram")
        omm_energies, aux_energies = compute_sample_energies(
            data_samples, target
        )

        fig = plot_energy_histogram(
            "OpenMM",
            omm_energies,
            stats.omm_energies,
            stats.weights,
        )
        tf.summary.histogram(
            f"/metrics/energies/open_mm/data",
            omm_energies,
            step=tot_iter,
        )
        tf.summary.histogram(
            f"/metrics/energies/open_mm/model",
            stats.omm_energies,
            step=tot_iter,
        )
        write_figure_to_tensorboard(
            f"{scope}/plots/energies/open_mm", fig, tot_iter
        )

        fig = plot_energy_histogram(
            "OpenMM + Auxiliaries",
            omm_energies + aux_energies,
            stats.omm_energies + stats.aux_energies,
            stats.weights,
        )
        tf.summary.histogram(
            f"/metrics/energies/open_mm+aux/data",
            omm_energies + aux_energies,
            step=tot_iter,
        )
        tf.summary.histogram(
            f"/metrics/energies/open_mm+aux/model",
            stats.omm_energies + stats.aux_energies,
            step=tot_iter,
        )
        write_figure_to_tensorboard(
            f"{scope}/plots/energies/total", fig, tot_iter
        )

        fig = plot_energy_histogram(
            "Auxiliaries",
            aux_energies,
            stats.aux_energies,
            stats.weights,
        )
        tf.summary.histogram(
            f"/metrics/energies/aux/data", aux_energies, step=tot_iter
        )
        tf.summary.histogram(
            f"/metrics/energies/aux/model",
            stats.aux_energies,
            step=tot_iter,
        )
        write_figure_to_tensorboard(
            f"{scope}/plots/energies/auxiliary", fig, tot_iter
        )
