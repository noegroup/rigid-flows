import argparse
import datetime
import logging
import shutil
from dataclasses import asdict
from typing import cast

import equinox as eqx
import tensorflow as tf  # type: ignore
from rigid_flows.data import AugmentedData
from rigid_flows.density import (
    BaseDensity,
    KeyArray,
    PositionPrior,
    TargetDensity,
)
from rigid_flows.flow import State, build_flow
from rigid_flows.reporting import Reporter, pretty_json
from rigid_flows.specs import ExperimentSpecification
from rigid_flows.train import run_training_stage

from flox._src.flow.api import Transform
from flox.flow import Pipe
from flox.util import key_chain


def backup_config_file(run_dir: str, specs_path: str):
    shutil.copy(specs_path, f"{run_dir}/config.yaml")


def setup_tensorboard(run_dir: str):
    local_run_dir = (
        f"{run_dir}/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    )

    writer = tf.summary.create_file_writer(local_run_dir)

    return writer, local_run_dir


def setup_model(key: KeyArray, specs: ExperimentSpecification):
    chain = key_chain(key)

    logging.info(f"Loading target density.")
    target = TargetDensity.from_specs(
        specs.model.auxiliary_shape, specs.model.target, specs.system
    )

    logging.info(f"Setting up base density.")
    assert target.data is not None
    prior = PositionPrior(target.data)
    base = BaseDensity.from_specs(
        specs.system,
        specs.model.base,
        prior,
        target.box,
        specs.model.auxiliary_shape,
    )

    logging.info(f"Setting up flow model.")
    flow = build_flow(
        next(chain), specs.model.auxiliary_shape, specs.model.flow
    )
    if specs.model.pretrained_model_path is not None:
        logging.info(
            f"Loading pre-trained model from {specs.model.pretrained_model_path}."
        )
        flow = cast(
            Pipe[AugmentedData, State],
            eqx.tree_deserialise_leaves(
                specs.model.pretrained_model_path, flow
            ),
        )

    return base, target, flow


def train(
    key: KeyArray,
    run_dir: str,
    specs: ExperimentSpecification,
    base: BaseDensity,
    target: TargetDensity,
    flow: Transform[AugmentedData, State],
    tot_iter: int,
) -> Transform[AugmentedData, State]:
    chain = key_chain(key)
    tf.summary.text("run_params", pretty_json(asdict(specs)), step=tot_iter)
    logger = logging.getLogger("main")
    logging.info(f"Starting training.")
    reporter = Reporter(base, target, run_dir, specs.reporting, scope=None)
    reporter.with_scope(f"initial").report_model(next(chain), flow, tot_iter)
    for stage, train_spec in enumerate(specs.train):
        flow = run_training_stage(
            next(chain),
            base,
            target,
            flow,
            train_spec,
            reporter.with_scope(f"training_stage_{stage}"),
            tot_iter,
        )
        tot_iter += train_spec.num_iterations
    return flow


def main():

    tf.config.experimental.set_visible_devices([], "GPU")

    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()

    writer, local_run_dir = setup_tensorboard(args.run_dir)

    logging.getLogger().setLevel(logging.INFO)

    logging.info(f"Logging tensorboard logs to {local_run_dir}.")

    logging.basicConfig(
        filename=f"logs.txt",
        filemode="w",
        encoding="utf-8",
    )

    logging.info(f"Loading specs from {args.specs}.")
    specs = ExperimentSpecification.load_from_file(args.specs)

    chain = key_chain(specs.seed)

    base, target, flow = setup_model(next(chain), specs)

    backup_config_file(local_run_dir, args.specs)

    tot_iter = specs.global_step if specs.global_step is not None else 0
    with writer.as_default():
        flow = train(
            next(chain), local_run_dir, specs, base, target, flow, tot_iter
        )


if __name__ == "__main__":
    main()
