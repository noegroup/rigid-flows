import logging

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass

from .system import (
    ErrorHandling,
    OpenMMEnergyModel,
    SimulationBox,
    SystemSpecification,
)

logger = logging.getLogger("rigid-flows")


@pytree_dataclass(frozen=True)
class Data:
    """Raw data format."""

    pos: jnp.ndarray
    box: jnp.ndarray
    energy: jnp.ndarray
    force: jnp.ndarray | None = None

    @staticmethod
    def from_specs(
        specs: SystemSpecification,
    ):
        path = f"{specs.path}/MDtraj-{specs}.npz"
        logger.info(f"Loading data from {path}")
        raw = np.load(path)
        data = Data(*map(jnp.array, raw.values()))
        if data.box.shape[1:] == (3, 3):
            data = Data(
                data.pos, jax.vmap(jnp.diag)(data.box), data.energy, data.force
            )
        assert data.box.shape[1:] == (3,)
        return data

    def recompute_forces(self, model: OpenMMEnergyModel):
        _, forces = model.compute_energies_and_forces(
            np.array(self.pos),
            np.diag(np.array(self.box[0])),
            error_handling=ErrorHandling.RaiseException,
        )
        return Data(
            pos=self.pos,
            box=self.box,
            energy=self.energy,
            force=jnp.array(forces),
        )


@pytree_dataclass(frozen=True)
class AugmentedData:
    """Data augmented with auxilaries and quaternion signs."""

    pos: Array
    aux: Array
    sign: Array
    box: SimulationBox
    force: jnp.ndarray | None = None
