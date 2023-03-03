from typing import cast

from jax_dataclasses import pytree_dataclass
from mlparams.mlparams import from_yaml, to_yaml


@pytree_dataclass(frozen=True)
class ReportingSpecifications:
    num_samples: int | None
    num_samples_per_batch: int
    plot_quaternions: tuple[int, ...] | None
    plot_oxygens: bool
    plot_energy_histograms: bool
    report_ess: bool
    report_likelihood: bool
    save_model: bool
    save_samples: bool
    save_statistics: bool


@pytree_dataclass(frozen=True)
class NodeUpdateSpecification:
    num_blocks: int
    num_heads: int
    num_channels: int


@pytree_dataclass(frozen=True)
class CouplingSpecification:
    num_repetitions: int
    auxiliary_update: NodeUpdateSpecification
    position_update: NodeUpdateSpecification
    quaternion_update: NodeUpdateSpecification


@pytree_dataclass(frozen=True)
class FlowSpecification:
    couplings: tuple[CouplingSpecification, ...]


@pytree_dataclass(frozen=True)
class TrainingSpecification:
    num_epochs: int
    num_iters_per_epoch: int
    init_learning_rate: float
    target_learning_rate: float
    weight_nll: float
    weight_fe: float

    num_samples: int

    @property
    def num_iterations(self):
        return self.num_epochs * self.num_iters_per_epoch


@pytree_dataclass(frozen=True)
class SystemSpecification:
    path: str
    num_molecules: int
    temperature: int
    ice_type: str
    water_type: str = "tip4pew"

    num_samples: int | None = None

    def __str__(self) -> str:
        string = f"ice{self.ice_type}_T{self.temperature}_N{self.num_molecules}"
        if self.water_type != "tip4pew":
            string = f"{self.water_type}_{string}"
        return string


@pytree_dataclass(frozen=True)
class ModelSpecification:
    use_auxiliary: bool
    flow: FlowSpecification
    base: SystemSpecification
    target: SystemSpecification
    pretrained_model_path: str | None


@pytree_dataclass(frozen=True)
class ExperimentSpecification:
    seed: int
    model: ModelSpecification
    train: tuple[TrainingSpecification]
    reporting: ReportingSpecifications
    global_step: int | None

    @staticmethod
    def load_from_file(path: str) -> "ExperimentSpecification":
        with open(path, "r") as f:
            result = from_yaml(ExperimentSpecification, f)
        return cast(ExperimentSpecification, result)

    def save_to_file(self, path):
        with open(path, "w") as f:
            result = to_yaml(self, f)
        return result
