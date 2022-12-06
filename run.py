import argparse
import datetime
import logging
import shutil
from dataclasses import asdict
from typing import cast

import equinox as eqx
import tensorflow as tf  # type: ignore
from jax_dataclasses import pytree_dataclass
from rigid_flows.config import from_yaml, pretty_json, to_yaml
from rigid_flows.data import AugmentedData
from rigid_flows.density import (
    BaseDensity,
    BaseSpecification,
    TargetDensity,
    TargetSpecification,
)
from rigid_flows.flow import FlowSpecification, State, build_flow
from rigid_flows.system import SystemSpecification
from rigid_flows.train import TrainingSpecification, run_training_stage

from flox.flow import Pipe
from flox.util import key_chain

logger = logging.getLogger("run.example")
logger.setLevel(logging.INFO)


@pytree_dataclass(frozen=True)
class ModelSpecification:
    auxiliary_shape: tuple[int, ...]
    flow: FlowSpecification
    base: BaseSpecification
    target: TargetSpecification
    pretrained_model_path: str | None


@pytree_dataclass(frozen=True)
class ExperimentSpecification:
    seed: int
    model: ModelSpecification
    system: SystemSpecification
    train: tuple[TrainingSpecification]
    run_dir: str

    @staticmethod
    def load_from_file(path: str) -> "ExperimentSpecification":
        with open(path, "r") as f:
            result = from_yaml(ExperimentSpecification, f)
        return cast(ExperimentSpecification, result)

    def save_to_file(self, path):
        with open(path, "w") as f:
            result = to_yaml(self, f)
        return result


def backup_config_file(run_dir, specs_path):
    shutil.copy(specs_path, f"{run_dir}/config.yaml")


def setup_tensorboard(specs):
    logger.info(f"Logging tensorboard logs to {specs.run_dir}.")

    log_path = f'run-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    run_dir = f"{specs.run_dir}/{log_path}"

    writer = tf.summary.create_file_writer(run_dir)

    return writer, run_dir


def setup_model(key, specs):
    chain = key_chain(key)

    logger.info(f"Loading target density.")
    target = TargetDensity.from_specs(
        specs.model.auxiliary_shape, specs.model.target, specs.system
    )

    logger.info(f"Setting up base density.")
    base = BaseDensity.from_specs(
        specs.system, specs.model.base, target.box, specs.model.auxiliary_shape
    )

    logger.info(f"Setting up flow model.")
    flow = build_flow(
        next(chain), specs.model.auxiliary_shape, specs.model.flow
    )
    if specs.model.pretrained_model_path is not None:
        logger.info(
            f"Loading pre-trained model from {specs.model.pretrained_model_path}."
        )
        flow = cast(
            Pipe[AugmentedData, State],
            eqx.tree_deserialise_leaves(
                specs.model.pretrained_model_path, flow
            ),
        )

    return base, target, flow


def train(key, run_dir, specs, base, target, flow):
    chain = key_chain(key)
    tf.summary.text("run_params", pretty_json(asdict(specs)), step=0)
    logger.info(f"Starting training.")
    for stage, train_spec in enumerate(specs.train):
        with tf.name_scope(f"stage_{stage}"):
            flow = run_training_stage(
                next(chain), base, target, flow, train_spec
            )
            model_path = f"model_stage{stage}.eqx"
            eqx.tree_serialise_leaves(f"{run_dir}/{model_path}", flow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=str, required=True)
    args = parser.parse_args()

    logger.info(f"Loading specs from {args.specs}.")
    specs = ExperimentSpecification.load_from_file(args.specs)

    chain = key_chain(specs.seed)

    base, target, flow = setup_model(next(chain), specs)

    writer, run_dir = setup_tensorboard(specs)

    backup_config_file(run_dir, args.specs)

    with writer.as_default():
        train(next(chain), run_dir, specs, base, target, flow)


if __name__ == "__main__":
    main()
