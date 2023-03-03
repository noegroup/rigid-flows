#!/usr/bin/env python
# coding: utf-8

import sys
from typing import cast

import equinox as eqx
import jax
import lenses
import matplotlib.pyplot as plt
import numpy as np
from flox.flow import Pipe
from flox.util import key_chain
from jax import numpy as jnp

from rigid_flows.data import DataWithAuxiliary
from rigid_flows.density import OpenMMDensity
from rigid_flows.flow import RigidWithAuxiliary, build_flow
from rigid_flows.specs import ExperimentSpecification
from rigid_flows.utils import scanned_vmap

assert len(sys.argv) == 2, "please specify the folder path"
logdir_path = sys.argv[1]
stage = 0
epoch = 9

num_eval_samples = 10_000
num_iterations = 10
num_samples = num_eval_samples

stop_for_plots = True
override_stats = False
n_sigmas = 2  # errorbars are n_sigmas wide

print(f"+++ trainng stage {stage}, epoch {epoch} +++")
specs_path = f"{logdir_path}/config.yaml"
pretrained_model_path = f"{logdir_path}/training_stage_{stage}/epoch_{epoch}/model.eqx"
print(pretrained_model_path)

specs = ExperimentSpecification.load_from_file(specs_path)
specs = lenses.bind(specs).model.base.path.set(specs.model.base.path + "/eval_100")
selection = np.s_[-num_eval_samples:]

base = OpenMMDensity.from_specs(specs.model.use_auxiliary, specs.model.base, selection)
target = OpenMMDensity.from_specs(
    specs.model.use_auxiliary, specs.model.target, selection
)
model = base.omm_model.model

sc = model.n_molecules  # rescale free energy by number of molecules

chain = key_chain(42)


def count_params(model):
    return jax.tree_util.tree_reduce(
        lambda s, n: s + n.size if eqx.is_array(n) else s,
        model,
        jnp.zeros((), dtype=jnp.int32),
    ).item()


flow = build_flow(
    next(chain),
    specs.model.base.num_molecules,
    specs.model.use_auxiliary,
    specs.model.flow,
)
flow = cast(
    Pipe[DataWithAuxiliary, RigidWithAuxiliary],
    eqx.tree_deserialise_leaves(pretrained_model_path, flow),
)

training_data_size = (
    100_000 if specs.model.base.num_samples is None else specs.model.base.num_samples
)
print(f"tot flow parameters: {count_params(flow):_}")
print(f"MD training datapoints = {training_data_size:_}")
num_eval_samples = min(num_eval_samples, base.data.pos.shape[0])
print(f"MD eval datapoints = {num_eval_samples:_}")
print(f"batchs per epoch = {specs.train[0].num_iters_per_epoch}")
print(f"batch size = {specs.train[0].num_samples}")
print(
    f"data fraction: {specs.train[0].num_epochs*specs.train[0].num_samples*specs.train[0].num_iters_per_epoch/training_data_size}"
)


try:
    ref_file = (
        f"../data/water/DeltaF_estimates/DF-{specs.model.base}-{specs.model.target}.txt"
    )
    reference_deltaF, reference_deltaF_std = np.loadtxt(ref_file, unpack=True)
except FileNotFoundError:
    print("reference DeltaF not found")
    reference_deltaF, reference_deltaF_std = None, None


def ess(logw):
    return jnp.exp(
        2 * jax.scipy.special.logsumexp(logw) - jax.scipy.special.logsumexp(2 * logw)
    )


def plot_results(
    deltaFs,
    reference_deltaF=reference_deltaF,
    reference_deltaF_std=reference_deltaF_std,
    ESSs=None,
    num_samples=num_samples,
    std_deltaFs=None,
):
    if ESSs is not None:
        plt.plot(ESSs / num_samples * 100, "-o", label="forward")
        plt.axhline(100 / num_samples, c="k", ls=":", label="ESS=1")
        plt.ylabel("ESS %")
        plt.legend()
        plt.show(block=stop_for_plots)
        print(
            f"average ESS: {ESSs.mean() / num_samples:.2%} +/- {ESSs.std() / num_samples:.2%}"
        )

    xlim = [0, len(deltaFs)]
    n = n_sigmas  # how many sigmas for errorbar

    plt.plot(deltaFs, ".", c="green", label="LFEP")
    x = 2 * [deltaFs.mean()]
    plt.fill_between(
        xlim, x - n * deltaFs.std(), x + n * deltaFs.std(), color="green", alpha=0.3
    )
    plt.axhline(deltaFs.mean(), c="green")
    print(f"DeltaF_LFEP = {deltaFs.mean()/sc:.6f} +/- {n*deltaFs.std()/sc:.6f}")

    if reference_deltaF is not None:
        plt.axhline(reference_deltaF, c="k", ls=":", label="MBAR reference")
        x = np.array(2 * [reference_deltaF])
        plt.fill_between(
            xlim,
            x - n * reference_deltaF_std,
            x + n * reference_deltaF_std,
            color="k",
            alpha=0.1,
        )
        print(
            f"DeltaF_ref  = {reference_deltaF/sc:.6f} +/- {n*reference_deltaF_std/sc:.6f}"
        )
    plt.xlim(xlim)
    plt.legend()
    plt.show(block=stop_for_plots)


filename = f"{logdir_path}/training_stage_{stage}/epoch_{epoch}/LFEPstats"
stats_found = False
try:
    stats = np.load(filename + ".npz")
    stats_found = True
except FileNotFoundError:
    print("no LFEPstats found")

if stats_found:
    print("LFEPstats found:")
    if override_stats:
        print("    +++ they will be overwritten! +++")
    else:
        plot_results(**stats)
        sys.exit("keeping the current stats")


batch_size = 64


def jitvmap(fn, batch_size=batch_size):
    if batch_size is None:
        return jax.jit(jax.vmap(fn))
    else:
        return jax.jit(scanned_vmap(fn, batch_size))


# ## estimate LFEP with uncertainty

print(
    f"Runnig {num_iterations} iterations with {num_samples:_} samples, out of {num_eval_samples:_}"
)

N_k = np.array(2 * [num_samples])
u_kn = np.zeros((2, 2 * num_samples))

deltaFs = np.zeros(num_iterations)
std_deltaFs = np.zeros(num_iterations)
ESSs = np.zeros(num_iterations)
for i in range(num_iterations):  # again, this only works on startup!
    print("\niter:", i)
    print("sampling base...", end="\r")
    keys = jax.random.split(next(chain), num_samples)
    base_tr = jitvmap(base.sample)(keys)
    mapped_tr = jitvmap(flow.forward)(base_tr.obj)
    logw = base_tr.ldj + mapped_tr.ldj - jitvmap(target.potential)(mapped_tr.obj)
    ESSs[i] = ess(logw)
    deltaFs[i] = (jnp.log(len(logw)) - jax.scipy.special.logsumexp(logw)).item()
    print(f"DeltaF = {deltaFs[i]}, efficiency = {ESSs[i]/len(logw):.2%}")

plot_results(deltaFs, reference_deltaF, reference_deltaF_std, ESSs, num_samples)

np.savez(
    filename,
    deltaFs=deltaFs,
    std_deltaFs=std_deltaFs,
    ESSs=ESSs,
    num_samples=num_samples,
)
