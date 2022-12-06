from functools import partial
from logging import Logger
from typing import cast

import equinox as eqx
import jax
import optax
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from optax import (
    GradientTransformation,
    OptState,
    huber_loss,
    safe_root_mean_squares,
)

from flox.flow import Transform
from flox.util import key_chain, unpack

from .data import AugmentedData
from .density import DensityModel
from .flow import State
from .reporting import Reporter

KeyArray = Array | jax.random.PRNGKeyArray


Flow = Transform[AugmentedData, State]


def negative_log_likelihood(
    inp: AugmentedData,
    base: DensityModel,
    flow: Transform[AugmentedData, State],
):
    out, ldj = unpack(flow.forward(inp))
    return base.potential(out) - ldj


def flow_force(
    inp: AugmentedData,
    base: DensityModel,
    flow: Transform[AugmentedData, State],
):
    return -jax.grad(negative_log_likelihood)(inp, base, flow)


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
    return target.potential(out) - ldj


def per_sample_loss(
    key: KeyArray,
    base: DensityModel,
    target: DensityModel,
    flow: Transform[AugmentedData, State],
    weight_nll: float,
    weight_fm: float,
    weight_fe: float,
    fm_aggregation: str,
):
    loss = 0.0
    chain = key_chain(key)
    if weight_nll > 0:
        inp, _ = unpack(target.sample(next(chain)))
        loss += weight_nll * negative_log_likelihood(inp, base, flow)
    if weight_fm > 0:
        inp, _ = unpack(target.sample(next(chain)))
        loss += weight_fm * force_matching_loss(inp, base, flow, fm_aggregation)
    if weight_fe > 0:
        inp, _ = unpack(base.sample(next(chain)))
        loss += weight_fe * free_energy_loss(inp, target, flow)
    return loss


def batch_loss(
    key: KeyArray,
    base: DensityModel,
    target: DensityModel,
    flow: Transform[AugmentedData, State],
    weight_nll: float,
    weight_fm: float,
    weight_fe: float,
    fm_aggregation: str,
    num_samples: int,
):
    return jnp.mean(
        jax.vmap(
            partial(
                per_sample_loss,
                base=base,
                target=target,
                flow=flow,
                weight_nll=weight_nll,
                weight_fm=weight_fm,
                weight_fe=weight_fe,
                fm_aggregation=fm_aggregation,
            )
        )(jax.random.split(key, num_samples))
    )


@pytree_dataclass(frozen=True)
class TrainingSpecification:
    num_epochs: int
    num_iters_per_epoch: int
    init_learning_rate: float
    target_learning_rate: float
    weight_nll: float
    weight_fm: float
    weight_fe: float
    fm_aggregation: str
    num_samples: int


def get_scheduler(specs: TrainingSpecification):
    learning_rates = jnp.power(
        10.0,
        jnp.linspace(
            jnp.log10(specs.init_learning_rate),
            jnp.log10(specs.target_learning_rate),
            specs.num_epochs,
        ),
    )
    alphas = learning_rates[1:] / learning_rates[:-1]
    alphas = jnp.concatenate([alphas, jnp.ones((1,))])
    scheduler = optax.join_schedules(
        tuple(
            optax.cosine_decay_schedule(
                learning_rate, specs.num_iters_per_epoch, alpha=alpha
            )
            for learning_rate, alpha in zip(learning_rates, alphas)
        ),
        (specs.num_iters_per_epoch,) * (specs.num_epochs - 1),
    )
    return scheduler


@pytree_dataclass(frozen=True)
class Trainer:
    optim: GradientTransformation
    base: DensityModel
    target: DensityModel
    weight_nll: float
    weight_fm: float
    weight_fe: float
    fm_aggregation: str
    num_samples: int

    def init(
        self,
        key: KeyArray,
        flow: Flow,
    ):
        params, static = eqx.partition(flow, eqx.is_array)  # type: ignore
        opt_state = self.optim.init(params)
        return opt_state

    def step(
        self,
        key: KeyArray,
        flow: Flow,
        opt_state: OptState,
    ):
        loss, grad = eqx.filter_value_and_grad(
            lambda key: batch_loss(
                key=key,
                flow=flow,
                base=self.base,
                target=self.target,
                num_samples=self.num_samples,
                weight_nll=self.weight_nll,
                weight_fm=self.weight_fm,
                weight_fe=self.weight_fe,
                fm_aggregation=self.fm_aggregation,
            )
        )(key)
        updates, opt_state = self.optim.update(grad, opt_state)
        flow = cast(Flow, eqx.apply_updates(flow, updates))
        return loss, flow, opt_state

    @staticmethod
    def from_specs(
        base: DensityModel, target: DensityModel, specs: TrainingSpecification
    ):
        return Trainer(
            optax.adam(get_scheduler(specs)),
            base,
            target,
            specs.weight_nll,
            specs.weight_fm,
            specs.weight_fe,
            specs.fm_aggregation,
            specs.num_samples,
        )


def run_training_stage(
    key: KeyArray,
    base: DensityModel,
    target: DensityModel,
    flow: Flow,
    specs: TrainingSpecification,
    reporter: Reporter,
):

    chain = key_chain(key)
    scheduler = get_scheduler(specs)
    trainer = Trainer.from_specs(base, target, specs)

    opt_state = eqx.filter_jit(trainer.init)(next(chain), flow)
    step = eqx.filter_jit(trainer.step)

    for num_iter in range(specs.num_iters_per_epoch):
        loss, flow, opt_state = step(next(chain), flow, opt_state)
        reporter.write_scalar("loss", loss, num_iter)
        reporter.write_scalar("learning_rate", scheduler(num_iter), num_iter)

    return flow
