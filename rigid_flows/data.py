import logging

import jax
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass

from .system import OpenMMEnergyModel, SimulationBox, SystemSpecification

logger = logging.getLogger("rigid-flows")


@pytree_dataclass(frozen=True)
class Data:
    """Raw data format."""

    pos: jnp.ndarray
    box: jnp.ndarray
    energy: jnp.ndarray

    @staticmethod
    def from_specs(
        specs: SystemSpecification,
        omm_model: OpenMMEnergyModel,
    ):
        path = f"{specs.path}/MDtraj-{specs}.npz"
        logging.info(f"Loading data from {path}")
        raw = np.load(path)
        if raw["box"].shape[0] == 1:
            assert jnp.allclose(
                raw["box"][0], omm_model.model.box
            ), "model and MDtraj box differ"

        data = Data(
            pos=raw["pos"][:specs.num_samples].reshape(
                -1, omm_model.model.n_molecules, omm_model.model.n_sites, 3
            ),
            box=jax.vmap(jnp.diag)(raw["box"][:specs.num_samples]),
            energy=raw["ene"][:specs.num_samples],
        )

        return data


@pytree_dataclass(frozen=True)
class PreprocessedData:
    pos: jnp.ndarray
    box: jnp.ndarray
    energy: jnp.ndarray

    @staticmethod
    def from_data(data: Data) -> "PreprocessedData":
        ## remove global translation using first molecule
        pos = data.pos - data.pos[:, :1, :1]

        ## put molecules back into PBC (without breaiking them)
        shift = (pos[:, :, :1] % data.box) - pos[:, :, :1]
        pos = pos + shift

        return PreprocessedData(pos, data.box, data.energy)


@pytree_dataclass(frozen=True)
class DataWithAuxiliary:
    """Data augmented with auxilaries and quaternion signs."""

    pos: Array
    aux: Array | None
    sign: Array
    box: SimulationBox
