from functools import partial
from itertools import accumulate
from math import prod
from typing import Callable, Iterable, cast

import equinox as eqx
import jax
import lenses
import optax
import tensorflow as tf  # type: ignore
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from optax import (
    GradientTransformation,
    OptState,
    huber_loss,
    safe_root_mean_squares,
)
from tqdm import tqdm

from flox._src.flow.potential import Potential, PushforwardPotential
from flox._src.flow.sampling import PushforwardSampler, Sampler
from flox.flow import PullbackSampler, Transform
from flox.util import key_chain, unpack

from .data import AugmentedData
from .density import BaseDensity, DensityModel, TargetDensity
from .flow import State
from .reporting import Reporter
from .specs import SystemSpecification, TrainingSpecification
from .system import OpenMMEnergyModel, wrap_openmm_model
from .utils import jit_and_cleanup_cache

KeyArray = Array | jax.random.PRNGKeyArray


Flow = Transform[AugmentedData, State]


def get_scheduler(specs: TrainingSpecification):
    learning_rates = jnp.power(
        10.0,
        jnp.linspace(
            jnp.log10(specs.init_learning_rate),
            jnp.log10(specs.target_learning_rate),
            specs.num_epochs + 1,
        ),
    )
    alphas = learning_rates[1:] / learning_rates[:-1]
    alphas = jnp.concatenate([alphas, jnp.ones((1,))])
    scheduler = optax.join_schedules(
        tuple(
            optax.cosine_decay_schedule(
                learning_rate, specs.num_iters_per_epoch, alpha=alpha
            )
            for learning_rate, alpha in zip(learning_rates, alphas, strict=True)
        ),
        tuple(accumulate((specs.num_iters_per_epoch,) * (specs.num_epochs))),
    )
    return scheduler


LossFun = Callable[[Transform[AugmentedData, State]], Array]
LossFunFactory = Callable[[KeyArray], LossFun]
TotLossFun = Callable[[Transform[AugmentedData, State]], tuple[Array, dict]]


def force_matching_loss_fn(
    key: KeyArray,
    base: Potential[State],
    source: Sampler[AugmentedData],
    omm_energy_model: OpenMMEnergyModel,
    num_samples: int,
    perturbation_noise: float,
    ignore_charge_site: bool,
) -> LossFun:
    chain = key_chain(key)
    keys = jax.random.split(next(chain), num_samples)
    samples = jax.vmap(source)(keys).obj
    if perturbation_noise > 0:
        samples = lenses.bind(samples).pos.set(
            samples.pos
            + perturbation_noise
            * jax.random.normal(next(chain), samples.pos.shape)
        )
    _, omm_forces = wrap_openmm_model(omm_energy_model)[1](
        samples.pos, None, True
    )

    num_atoms = prod(samples.pos.shape[1:])
    if ignore_charge_site:
        mask = (jnp.arange(num_atoms) % 4) != 3
        mask = jnp.tile(mask[None], (num_samples, 1))
    else:
        mask = jnp.ones((num_samples, num_atoms))

    def evaluate(flow: Transform[AugmentedData, State]) -> Array:
        flow_grads = jax.vmap(jax.grad(PushforwardPotential(base, flow)))(
            samples
        )
        mse = 0
        mse += jnp.mean(
            jnp.square(
                -(
                    flow_grads.pos.reshape(num_samples, -1)
                    - omm_forces.reshape(num_samples, -1)
                )
                * mask
            )
        )
        mse += jnp.mean(jnp.square(flow_grads.aux - samples.aux))
        mse += jnp.mean(jnp.square(flow_grads.com - samples.com))
        return mse

    return evaluate


def kullback_leiber_divergence_fn(
    key: KeyArray,
    base: DensityModel[State],
    target: DensityModel[AugmentedData],
    num_samples: int,
    reverse: bool,
) -> LossFun:
    keys = jax.random.split(key, num_samples)

    def evaluate(flow: Transform[AugmentedData, State]) -> Array:
        if not reverse:
            out = jax.vmap(PushforwardSampler(target.sample, flow))(keys)
            return jnp.mean(jax.vmap(base.potential)(out.obj) - out.ldj)
        else:
            out = jax.vmap(PullbackSampler(base.sample, flow))(keys)
            return jnp.mean(jax.vmap(target.potential)(out.obj) - out.ldj)

    return evaluate


def var_grad_loss_fn(
    key: KeyArray,
    sampler: Sampler[AugmentedData],
    base: Potential[State],
    target: Potential[AugmentedData],
    num_samples: int,
) -> LossFun:

    keys = jax.random.split(key, num_samples)

    samples = jax.lax.stop_gradient(jax.vmap(sampler)(keys).obj)

    def evaluate(flow: Transform[AugmentedData, State]) -> Array:
        flow_energies = jax.vmap(PushforwardPotential(base, flow))(samples)
        target_energies = jax.vmap(target)(samples)
        return jnp.var(flow_energies - target_energies)

    return evaluate


@pytree_dataclass(frozen=True)
class Loss:
    label: str
    loss_fn: LossFun
    weight: float


def total_loss_fn(losses: Iterable[Loss]) -> TotLossFun:
    def evaluate(flow: Transform[AugmentedData, State]) -> tuple[Array, dict]:
        report = {}
        tot_loss = jnp.zeros(())
        for loss in losses:
            loss_value = loss.loss_fn(flow)
            tot_loss += loss.weight * loss_value
            report[loss.label] = loss_value
        return tot_loss, report

    return evaluate


def update_fn(
    optim: GradientTransformation,
    loss_fn: TotLossFun,
):
    def update(
        flow: Transform[AugmentedData, State],
        opt_state: OptState,
    ):
        (loss, report), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            flow
        )
        params = eqx.filter(flow, eqx.is_array)
        updates, opt_state = optim.update(grad, opt_state, params)  # type: ignore
        flow = cast(
            Transform[AugmentedData, State], eqx.apply_updates(flow, updates)
        )
        return loss, flow, opt_state, report

    return update


def train_fn(
    flow: Transform[AugmentedData, State],
    base: BaseDensity,
    target: TargetDensity,
    specs: TrainingSpecification,
):
    params = eqx.filter(flow, eqx.is_array)
    optim = optax.adam(get_scheduler(specs))
    optim = optax.apply_if_finite(optim, specs.apply_if_finite_trials)
    if specs.use_grad_clipping:
        optim = optax.adaptive_grad_clip(specs.grad_clipping_ratio)

    opt_state = optim.init(params)  # type: ignore

    def init_losses(
        key: KeyArray,
        flow: Transform[AugmentedData, State],
    ) -> TotLossFun:
        chain = key_chain(key)
        partial_loss_fns = []
        if specs.weight_fe > 0:
            partial_loss_fns.append(
                Loss(
                    "reverse_kl",
                    kullback_leiber_divergence_fn(
                        key=next(chain),
                        base=base,
                        target=target,
                        num_samples=specs.num_samples,
                        reverse=True,
                    ),
                    specs.weight_fe,
                )
            )
        if specs.weight_nll > 0:
            partial_loss_fns.append(
                Loss(
                    "neg_log_likelihood",
                    kullback_leiber_divergence_fn(
                        key=next(chain),
                        base=base,
                        target=target,
                        num_samples=specs.num_samples,
                        reverse=False,
                    ),
                    specs.weight_nll,
                )
            )
        if specs.weight_fm_model > 0:
            partial_loss_fns.append(
                Loss(
                    "force_matching_model_samples",
                    force_matching_loss_fn(
                        key=next(chain),
                        base=base.potential,
                        source=PullbackSampler(base.sample, flow),
                        omm_energy_model=target.model,
                        num_samples=specs.num_samples,
                        perturbation_noise=specs.fm_model_perturbation_noise,
                        ignore_charge_site=specs.fm_ignore_charge_site,
                    ),
                    specs.weight_nll,
                )
            )
        if specs.weight_fm_target > 0:
            partial_loss_fns.append(
                Loss(
                    "force_matching_target_samples",
                    force_matching_loss_fn(
                        key=next(chain),
                        base=base.potential,
                        source=target.sample,
                        omm_energy_model=target.model,
                        num_samples=specs.num_samples,
                        perturbation_noise=specs.fm_target_perturbation_noise,
                        ignore_charge_site=specs.fm_ignore_charge_site,
                    ),
                    specs.weight_nll,
                )
            )
        if specs.weight_vg_model > 0:
            partial_loss_fns.append(
                Loss(
                    "var_grad_model_samples",
                    var_grad_loss_fn(
                        key=next(chain),
                        sampler=PullbackSampler(base.sample, flow),
                        target=target.potential,
                        base=base.potential,
                        num_samples=specs.num_samples,
                    ),
                    specs.weight_vg_model,
                )
            )
        if specs.weight_vg_target > 0:
            partial_loss_fns.append(
                Loss(
                    "var_grad_target_samples",
                    var_grad_loss_fn(
                        key=next(chain),
                        sampler=target.sample,
                        target=target.potential,
                        base=base.potential,
                        num_samples=specs.num_samples,
                    ),
                    specs.weight_vg_model,
                )
            )
        return total_loss_fn(partial_loss_fns)

    def train_step(
        key: KeyArray,
        flow: Transform[AugmentedData, State],
        opt_state: OptState,
    ):
        loss_fn = init_losses(key, flow)
        return update_fn(optim, loss_fn)(flow, opt_state)

    return opt_state, train_step


def run_training_stage(
    key: KeyArray,
    base: BaseDensity,
    target: TargetDensity,
    flow: Flow,
    training_specs: TrainingSpecification,
    system_specs: SystemSpecification,
    reporter: Reporter,
    tot_iter: int,
):

    chain = key_chain(key)
    scheduler = get_scheduler(training_specs)

    opt_state, train_step = train_fn(flow, base, target, training_specs)

    with jit_and_cleanup_cache(train_step) as step:
        for num_epoch in range(training_specs.num_epochs):

            target.model.set_softcore_cutoff(
                system_specs.softcore_cutoff,
                system_specs.softcore_potential,
                system_specs.softcore_slope,
            )

            epoch_reporter = reporter.with_scope(f"epoch_{num_epoch}")
            pbar = tqdm(
                range(training_specs.num_iters_per_epoch),
                desc=f"Epoch: {num_epoch}",
            )

            for _ in pbar:
                loss, flow, opt_state, report = step(
                    next(chain), flow, opt_state
                )
                tf.summary.scalar(
                    f"loss/total/{reporter.scope}", loss, tot_iter
                )
                for name, val in report.items():
                    tf.summary.scalar(f"loss/{name}", val, tot_iter)

                tf.summary.scalar(
                    f"learning_rate/{reporter.scope}",
                    scheduler(tot_iter),
                    tot_iter,
                )
                tot_iter += 1

            target.model.set_softcore_cutoff(None)

            epoch_reporter.report_model(next(chain), flow, tot_iter)
    return flow
