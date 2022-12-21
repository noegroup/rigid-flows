from functools import partial
from itertools import accumulate
from typing import cast

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

from flox.flow import PullbackSampler, Transform
from flox.util import key_chain, unpack

from .data import AugmentedData
from .density import BaseDensity, DensityModel, TargetDensity
from .flow import State
from .reporting import Reporter
from .specs import SystemSpecification, TrainingSpecification
from .utils import jit_and_cleanup_cache

KeyArray = Array | jax.random.PRNGKeyArray


Flow = Transform[AugmentedData, State]


def negative_log_likelihood(
    inp: AugmentedData,
    base: DensityModel,
    flow: Transform[AugmentedData, State],
):
    out, ldj = unpack(flow.forward(inp))
    return jnp.sum(base.potential(out) - ldj)


def flow_force(
    inp: AugmentedData,
    base: DensityModel,
    flow: Transform[AugmentedData, State],
):
    return -jax.grad(negative_log_likelihood)(inp, base, flow).pos


def force_matching_loss(
    inp: AugmentedData,
    base: DensityModel,
    flow: Transform[AugmentedData, State],
    aggregation: str,
):
    assert inp.force is not None
    diff = flow_force(inp, base, flow).reshape(-1) - inp.force.reshape(-1)
    if aggregation == "mse":
        return 0.5 * jnp.square(diff).sum()
    elif aggregation == "rmse":
        return safe_root_mean_squares(diff, 1e-12).sum()
    elif aggregation == "mae":
        return jnp.abs(diff).sum()
    elif aggregation == "huber":
        return huber_loss(diff).sum()
    else:
        raise NotImplementedError(f"{aggregation}")


def free_energy_loss(
    inp: State,
    target: DensityModel,
    flow: Transform[AugmentedData, State],
):
    out, ldj = unpack(flow.inverse(inp))
    return jnp.sum(target.potential(out) - ldj)


def energy_difference(
    inp: AugmentedData,
    base: DensityModel,
    target: DensityModel,
    flow: Transform[AugmentedData, State],
):
    model_energy = negative_log_likelihood(inp, base, flow)
    target_energy = target.potential(inp)
    return target_energy - model_energy


def per_sample_loss(
    key: KeyArray,
    base: DensityModel,
    target: TargetDensity,
    flow: Transform[AugmentedData, State],
    weight_nll: float,
    weight_fm_target: float,
    weight_fm_model: float,
    weight_fe: float,
    weight_vg_target: float,
    weight_vg_model: float,
    fm_aggregation: str | None,
):
    chain = key_chain(key)

    total_loss = 0
    num_losses = 0
    losses = {}
    var_grad_losses = {}
    if weight_nll > 0:
        inp, _ = unpack(target.sample(next(chain)))
        nll_loss = negative_log_likelihood(inp, base, flow)
        losses["nll"] = nll_loss
        total_loss += weight_nll * nll_loss

    if weight_fe > 0:
        inp, _ = unpack(base.sample(next(chain)))
        kl_loss = free_energy_loss(inp, target, flow)
        losses["kl"] = kl_loss
        total_loss += weight_fe * kl_loss

    if weight_fm_target > 0:
        assert fm_aggregation is not None
        inp, _ = unpack(target.sample(next(chain)))
        inp = jax.lax.stop_gradient(inp)
        if inp.force is None:
            _, force = target.model.compute_energies_and_forces(inp.pos, None)
            inp = lenses.bind(inp).force.set(force)
        fm_loss = force_matching_loss(inp, base, flow, fm_aggregation)
        losses["fm_target"] = fm_loss
        total_loss += weight_fm_target * fm_loss
    if weight_fm_model > 0:
        assert fm_aggregation is not None
        inp, _ = unpack(PullbackSampler(base.sample, flow)(next(chain)))

        inp = jax.lax.stop_gradient(inp)
        _, force = target.model.compute_energies_and_forces(inp.pos, None)
        inp = lenses.bind(inp).force.set(force)
        fm_loss = force_matching_loss(inp, base, flow, fm_aggregation)
        losses["fm_model"] = fm_loss
        total_loss += weight_fm_model * fm_loss

    if weight_vg_target > 0:
        inp, _ = unpack(target.sample(next(chain)))
        var_grad_losses["target"] = energy_difference(inp, base, target, flow)
    if weight_vg_model > 0:
        inp, _ = unpack(PullbackSampler(base.sample, flow)(next(chain)))
        inp = jax.lax.stop_gradient(inp)
        var_grad_losses["model"] = energy_difference(inp, base, target, flow)

    return total_loss, losses, var_grad_losses


def batch_loss(
    key: KeyArray,
    base: DensityModel,
    target: TargetDensity,
    flow: Transform[AugmentedData, State],
    weight_nll: float,
    weight_fm_target: float,
    weight_fm_model: float,
    weight_fe: float,
    weight_vg_model: float,
    weight_vg_target: float,
    fm_aggregation: str | None,
    num_samples: int,
):
    total_loss, losses, var_grad_losses = jax.vmap(
        partial(
            per_sample_loss,
            base=base,
            target=target,
            flow=flow,
            weight_nll=weight_nll,
            weight_fm_target=weight_fm_target,
            weight_fm_model=weight_fm_model,
            weight_fe=weight_fe,
            weight_vg_model=weight_vg_model,
            weight_vg_target=weight_vg_target,
            fm_aggregation=fm_aggregation,
        ),
        axis_name="batch",
    )(jax.random.split(key, num_samples))
    losses = jax.tree_map(jnp.mean, losses)
    total_loss_agg = jnp.mean(total_loss)

    for weight, loss_type in zip(
        (weight_vg_model, weight_vg_target), ("model", "target")
    ):
        if weight > 0.0 and loss_type in var_grad_losses:
            var_grad_loss = var_grad_losses[loss_type]
            var_grad_loss_agg = 0.5 * jnp.var(var_grad_loss)
            total_loss_agg += weight * var_grad_loss_agg
            losses["vargrad_" + loss_type] = var_grad_loss_agg
    return total_loss_agg, losses


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


@pytree_dataclass(frozen=True)
class Trainer:
    optim: GradientTransformation
    base: DensityModel
    target: TargetDensity
    weight_nll: float
    weight_fm_target: float
    weight_fm_model: float
    weight_fe: float
    weight_vg_model: float
    weight_vg_target: float
    fm_aggregation: str | None
    num_samples: int

    def init(
        self,
        key: KeyArray,
        flow: Flow,
    ):
        params, static = eqx.partition(flow, eqx.is_array)  # type: ignore
        opt_state = self.optim.init(params)
        return opt_state

    # jax.value_and_grad()
    def step(
        self,
        key: KeyArray,
        flow: Flow,
        opt_state: OptState,
    ):
        (loss, losses), grad = eqx.filter_value_and_grad(
            lambda flow: batch_loss(
                key=key,
                flow=flow,
                base=self.base,
                target=self.target,
                num_samples=self.num_samples,
                weight_nll=self.weight_nll,
                weight_fm_model=self.weight_fm_model,
                weight_fm_target=self.weight_fm_target,
                weight_fe=self.weight_fe,
                weight_vg_model=self.weight_vg_model,
                weight_vg_target=self.weight_vg_target,
                fm_aggregation=self.fm_aggregation,
            ),
            has_aux=True,
        )(flow)
        updates, opt_state = self.optim.update(grad, opt_state)
        flow = cast(Flow, eqx.apply_updates(flow, updates))
        return loss, flow, opt_state, losses

    @staticmethod
    def from_specs(
        base: DensityModel, target: TargetDensity, specs: TrainingSpecification
    ):
        optim = optax.adam(get_scheduler(specs))
        optim = optax.apply_if_finite(optim, 10)
        if specs.use_grad_clipping:
            optim = optax.adaptive_grad_clip(specs.grad_clipping_ratio)
        return Trainer(
            optim,
            base,
            target,
            specs.weight_nll,
            specs.weight_fm_model,
            specs.weight_fm_target,
            specs.weight_fe,
            specs.weight_vg_model,
            specs.weight_vg_target,
            specs.fm_aggregation,
            specs.num_samples,
        )


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
    trainer = Trainer.from_specs(base, target, training_specs)

    opt_state = trainer.init(next(chain), flow)

    with jit_and_cleanup_cache(trainer.step) as step:
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
                loss, flow, opt_state, other_losses = step(
                    next(chain), flow, opt_state
                )
                tf.summary.scalar(
                    f"loss/total/{reporter.scope}", loss, tot_iter
                )
                for name, val in other_losses.items():
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
