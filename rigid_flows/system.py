import enum
import logging
from functools import partial

import jax
import numpy as np
import openmm  # type: ignore
import openmm.app  # type: ignore
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from openmm import unit  # type: ignore

from .systems.watermodel import WaterModel

logger = logging.getLogger("run.example")


@pytree_dataclass(frozen=True)
class SimulationBox:
    """
    Simulation box.

    Args:
        max: maximum corner of the box
        min: minimum corner of the box
            defaults to (0, 0, 0)
    """

    max: jnp.ndarray
    min: jnp.ndarray = jnp.zeros(3)

    @property
    def size(self):
        return self.max - self.min


@pytree_dataclass(frozen=True)
class SystemSpecification:
    path: str
    num_molecules: int
    temperature: int
    ice_type: str
    recompute_forces: bool

    def __str__(self) -> str:
        return f"ice{self.ice_type}_T{self.temperature}_N{self.num_molecules}"


class ErrorHandling(enum.Enum):
    RaiseException = "raise_exception"
    LogWarning = "log_warning"
    Nothing = "nothing"


class OpenMMEnergyModel:
    def __init__(
        self,
        model: WaterModel,
        temperature: float,
        error_handling: ErrorHandling = ErrorHandling.LogWarning,
    ):
        self.model = model
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * unit.kelvin, 1 / unit.picosecond, 1 * unit.femtosecond
        )
        for force in model.system.getForces():
            if isinstance(
                force,
                (
                    openmm.MonteCarloBarostat,
                    openmm.MonteCarloAnisotropicBarostat,
                    openmm.MonteCarloFlexibleBarostat,
                ),
            ):
                force.setDefaultTemperature(temperature)
        self.simulation = openmm.app.Simulation(
            model.topology, model.system, integrator
        )
        self.error_handling = error_handling

    def compute_energies_and_forces(
        self,
        pos: np.ndarray,
        box: np.ndarray,
    ):
        energies = np.empty(pos.shape[0], dtype=np.float32)
        forces = np.empty_like(pos, dtype=np.float32)

        assert box.shape == (3, 3), f"box.shape = {box.shape}"
        self.simulation.context.setPeriodicBoxVectors(*box)

        # iterate over batch dimension
        for i in range(len(pos)):

            energy = jnp.nan * energies[0]
            force = jnp.nan * forces[0]

            try:
                self.simulation.context.setPositions(pos[i])
                # self.simulation.context.computeVirtualSites()

                state = self.simulation.context.getState(
                    getEnergy=True, getForces=True
                )
                energy = state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
                force = state.getForces(asNumpy=True).value_in_unit(
                    unit.kilojoule_per_mole / unit.nanometer
                )
            except Exception as e:
                match self.error_handling:
                    case ErrorHandling.RaiseException:
                        raise e
                    case ErrorHandling.LogWarning:
                        logger.warning(str(e))
                    case _:
                        pass
            finally:
                energies[i] = energy
                forces[i] = force

        return energies, forces

    @staticmethod
    def from_specs(specs: SystemSpecification):
        path = f"{specs.path}/model-{specs}.json"
        logger.info(f"Loading OpenMM model specs from {path}")
        model = WaterModel.load_from_json(path)
        return OpenMMEnergyModel(model, specs.temperature)


def wrap_openmm_model(model: OpenMMEnergyModel):
    def compute_energy_and_forces(pos: Array, box: Array, has_batch_dim: bool):

        assert box.shape == (3,)
        box = jnp.diag(box)

        if not has_batch_dim:
            assert pos.shape == (model.model.n_waters, 4, 3)
            pos = jnp.expand_dims(pos, axis=0)
        else:
            assert pos.shape[1:] == (model.model.n_waters, 4, 3)

        pos_flat = pos.reshape(pos.shape[0], -1, 3)

        shape_specs = (
            jax.ShapedArray(pos_flat.shape[:1], jnp.float32),
            jax.ShapedArray(pos_flat.shape, jnp.float32),
        )

        energies, forces_flat = jax.pure_callback(
            model.compute_energies_and_forces, shape_specs, pos_flat, box
        )

        forces = forces_flat.reshape(pos.shape)

        if not has_batch_dim:
            energies = jnp.squeeze(energies, axis=0)
            forces = jnp.squeeze(forces, axis=0)

        return energies, forces

    def eval_fwd(pos: Array, box: Array, has_batch_dim: bool):
        energy, force = compute_energy_and_forces(pos, box, has_batch_dim)
        return energy, (force, box)

    def eval_bwd(res, g):
        force, box = res
        return -g * force, jnp.zeros_like(box)

    @partial(jax.custom_vjp, nondiff_argnums=(1, 2))
    def eval(pos: Array, box: Array, has_batch_dim: bool):
        return eval_fwd(pos, box, has_batch_dim)[0]

    eval.defvjp(eval_fwd, eval_bwd)

    return eval
