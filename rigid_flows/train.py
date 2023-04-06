from functools import partial
from itertools import accumulate
from math import prod
from termios import FF1
from typing import Callable, Iterable, cast

import equinox as eqx
import jax
import lenses
import optax
import tensorflow as tf  # type: ignore
from flox._src.flow.potential import Potential, PushforwardPotential
from flox._src.flow.sampling import PushforwardSampler, Sampler
from flox.flow import PullbackSampler, Transform
from flox.util import key_chain
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from optax import GradientTransformation, OptState
from tqdm import tqdm

from .data import DataWithAuxiliary
from .density import DensityModel, OpenMMDensity
from .flow import EuclideanToRigidTransform
from .reporting import Reporter
from .specs import SystemSpecification, TrainingSpecification
from .system import OpenMMEnergyModel, wrap_openmm_model
from .utils import jit_and_cleanup_cache

KeyArray = Array | jax.random.PRNGKeyArray
Flow = Transform[DataWithAuxiliary, DataWithAuxiliary]


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


LossFun = Callable[[Transform[DataWithAuxiliary, DataWithAuxiliary]], Array]
LossFunFactory = Callable[[KeyArray], LossFun]
TotLossFun = Callable[
    [Transform[DataWithAuxiliary, DataWithAuxiliary]], tuple[Array, dict]
]


def kullback_leiber_divergence_fn(
    key: KeyArray,
    base: DensityModel[DataWithAuxiliary],
    target: DensityModel[DataWithAuxiliary],
    num_samples: int,
    reverse: bool,
) -> LossFun:
    keys = jax.random.split(key, num_samples)

    def evaluate(flow: Transform[DataWithAuxiliary, DataWithAuxiliary]) -> Array:
        if reverse:
            out = jax.vmap(PushforwardSampler(base.sample, flow))(keys)
            return jnp.mean(jax.vmap(target.potential)(out.obj) - out.ldj)
        else:
            out = jax.vmap(PullbackSampler(target.sample, flow))(keys)
            return jnp.mean(jax.vmap(base.potential)(out.obj) - out.ldj)

    return evaluate


@pytree_dataclass(frozen=True)
class Loss:
    label: str
    loss_fn: LossFun
    weight: float


def total_loss_fn(losses: Iterable[Loss]) -> TotLossFun:
    def evaluate(
        flow: Transform[DataWithAuxiliary, DataWithAuxiliary]
    ) -> tuple[Array, dict]:
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
        flow: Transform[DataWithAuxiliary, DataWithAuxiliary],
        opt_state: OptState,
    ):
        (loss, report), grad = eqx.filter_value_and_grad(loss_fn, has_aux=True)(flow)

        grad = (
            lenses.bind(grad)
            .Recur(EuclideanToRigidTransform)
            .modify(lambda node: jax.tree_map(lambda x: jnp.zeros_like(x), node))
        )

        updates, opt_state = optim.update(grad, opt_state)  # type: ignore
        flow = cast(
            Transform[DataWithAuxiliary, DataWithAuxiliary],
            eqx.apply_updates(flow, updates),
        )
        return loss, flow, opt_state, report

    return update


def train_fn(
    flow: Transform[DataWithAuxiliary, DataWithAuxiliary],
    base: OpenMMDensity,
    target: OpenMMDensity,
    specs: TrainingSpecification,
):
    params = eqx.filter(flow, eqx.is_array)
    optim = optax.adam(get_scheduler(specs))

    opt_state = optim.init(params)  # type: ignore

    def init_losses(
        key: KeyArray,
        # flow: Transform[DataWithAuxiliary, DataWithAuxiliary],
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
        return total_loss_fn(partial_loss_fns)

    def train_step(
        key: KeyArray,
        flow: Transform[DataWithAuxiliary, DataWithAuxiliary],
        opt_state: OptState,
    ):
        loss_fn = init_losses(key)#, flow)
        return update_fn(optim, loss_fn)(flow, opt_state)

    return opt_state, train_step


def run_training_stage(
    key: KeyArray,
    base: OpenMMDensity,
    target: OpenMMDensity,
    flow: Flow,
    training_specs: TrainingSpecification,
    # system_specs: SystemSpecification,
    reporter: Reporter,
    tot_iter: int,
    loss_reporter: list | None = None,
) -> Transform[DataWithAuxiliary, DataWithAuxiliary]:

    chain = key_chain(key)
    scheduler = get_scheduler(training_specs)

    opt_state, train_step = train_fn(flow, base, target, training_specs)

    with jit_and_cleanup_cache(train_step) as step:
        for num_epoch in range(training_specs.num_epochs):

            epoch_reporter = reporter.with_scope(f"epoch_{num_epoch}")
            pbar = tqdm(
                range(training_specs.num_iters_per_epoch),
                desc=f"Epoch: {1+num_epoch}/{training_specs.num_epochs}",
            )

            for _ in pbar:
                loss, flow, opt_state, report = step(next(chain), flow, opt_state)
                pbar.set_postfix({"loss": loss})
                tf.summary.scalar(f"loss/total/{reporter.scope}", loss, tot_iter)
                if loss_reporter is not None:
                    loss_reporter.append(loss.item())
                for name, val in report.items():
                    tf.summary.scalar(f"loss/{name}", val, tot_iter)

                tf.summary.scalar(
                    f"learning_rate/{reporter.scope}",
                    scheduler(tot_iter),
                    tot_iter,
                )
                tot_iter += 1

            epoch_reporter.report_model(next(chain), flow, tot_iter)
    return flow
