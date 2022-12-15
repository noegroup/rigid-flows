from typing import cast

from jax_dataclasses import pytree_dataclass
from mlparams.mlparams import from_yaml, to_yaml


@pytree_dataclass(frozen=True)
class ReportingSpecifications:
    num_samples: int
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
class PosAndAuxUpdateSpecification:
    seq_len: int
    activation: str
    num_dims: int
    num_hidden: int
    num_blocks: int


@pytree_dataclass(frozen=True)
class QuatUpdateSpecification:
    seq_len: int
    activation: str
    num_hidden: int
    num_blocks: int


@pytree_dataclass(frozen=True)
class PreprocessingSpecification:
    auxiliary_update: PosAndAuxUpdateSpecification
    displacement_encoder: PosAndAuxUpdateSpecification


@pytree_dataclass(frozen=True)
class CouplingSpecification:
    num_repetitions: int
    auxiliary_update: PosAndAuxUpdateSpecification
    position_update: PosAndAuxUpdateSpecification
    quaternion_update: QuatUpdateSpecification
    act_norm: bool


@pytree_dataclass(frozen=True)
class FlowSpecification:
    preprocessing: PreprocessingSpecification
    couplings: tuple[CouplingSpecification, ...]


@pytree_dataclass(frozen=True)
class TargetSpecification:
    cutoff_threshold: float | None


@pytree_dataclass(frozen=True)
class BaseSpecification:
    rot_concentration: float
    pos_concentration: float


@pytree_dataclass(frozen=True)
class ModelSpecification:
    auxiliary_shape: tuple[int, ...]
    flow: FlowSpecification
    base: BaseSpecification
    target: TargetSpecification
    pretrained_model_path: str | None


@pytree_dataclass(frozen=True)
class TrainingSpecification:
    num_epochs: int
    num_iters_per_epoch: int
    init_learning_rate: float
    target_learning_rate: float
    weight_nll: float
    weight_fm: float
    weight_fe: float
    weight_vg_model: float
    weight_vg_target: float
    fm_aggregation: str | None
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
    recompute_forces: bool
    store_forces: bool
    forces_path: str | None
    fixed_box: bool

    def __str__(self) -> str:
        return f"ice{self.ice_type}_T{self.temperature}_N{self.num_molecules}"


@pytree_dataclass(frozen=True)
class ExperimentSpecification:
    seed: int
    model: ModelSpecification
    system: SystemSpecification
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
