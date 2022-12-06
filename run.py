import argparse
import logging
import pprint
from typing import cast

from jax_dataclasses import pytree_dataclass
from rigid_flows.config import from_yaml, to_hparam_dict, to_yaml
from rigid_flows.density import (
    BaseDensity,
    BaseSpecification,
    TargetDensity,
    TargetSpecification,
)
from rigid_flows.flow import FlowSpecification, build_flow
from rigid_flows.reporting import Reporter
from rigid_flows.system import SystemSpecification
from rigid_flows.train import TrainingSpecification, run_training_stage
from tensorboardX import SummaryWriter

from flox.util import key_chain

logger = logging.getLogger("run.example")
logger.setLevel(logging.INFO)


@pytree_dataclass(frozen=True)
class ModelSpecification:
    auxiliary_shape: tuple[int, ...]
    flow: FlowSpecification
    base: BaseSpecification
    target: TargetSpecification


@pytree_dataclass(frozen=True)
class ExperimentSpecification:
    seed: int
    model: ModelSpecification
    system: SystemSpecification
    train: tuple[TrainingSpecification]
    logger_dir: str

    @staticmethod
    def load_from_file(path: str) -> "ExperimentSpecification":
        with open(path, "r") as f:
            result = from_yaml(ExperimentSpecification, f)
        return cast(ExperimentSpecification, result)

    def save_to_file(self, path):
        with open(path, "w") as f:
            result = to_yaml(self, f)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", type=str, required=True)
    args = parser.parse_args()

    logger.info(f"Loading specs from {args.specs}.")
    specs = ExperimentSpecification.load_from_file(args.specs)

    chain = key_chain(specs.seed)

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
    # logger.info(flow)

    pp = pprint.PrettyPrinter()

    logger.info(f"Logging tensorboard logs to {specs.logger_dir}.")
    summary_writer = SummaryWriter()

    prefix = ("train",)

    reporter = Reporter(summary_writer, prefix)
    reporter.write_text("configuration file", args.specs)

    logger.info(f"Starting training.")
    for stage, train_spec in enumerate(specs.train):
        reporter = Reporter(summary_writer, prefix + (f"stage_{stage}",))
        logger.info(f"Training {pp.pprint(train_spec)}")
        flow = run_training_stage(
            next(chain), base, target, flow, train_spec, reporter
        )

    summary_writer.close()
