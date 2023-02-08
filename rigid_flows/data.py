import logging

import jax
import lenses
import numpy as np
from flox import geom
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass

from .system import ErrorHandling, OpenMMEnergyModel, SimulationBox, SystemSpecification

# from .utils import smooth_maximum

logger = logging.getLogger("rigid-flows")


def unwrap(pos: jnp.ndarray, box: jnp.ndarray):
    """ "Using as reference the first configuration"""
    return jnp.where(
        jnp.abs(pos - pos[0]) / box > 0.5,
        pos - jnp.sign(pos - pos[0]) * box,
        pos,
    )


def boxify_oxy(pos, box: jnp.ndarray):
    oxy = pos[..., 0, :]
    boxed_oxy = oxy % box
    diff = boxed_oxy - oxy
    pos = pos + diff[..., None, :]
    return pos


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
        omm_model: OpenMMEnergyModel,
        selection: slice = np.s_[:],
    ):
        path = f"{specs.path}/MDtraj-{specs}.npz"
        logging.info(f"Loading data from {path}")
        raw = np.load(path)
        if raw["box"].shape[0] == 1:
            assert jnp.allclose(
                raw["box"][0], omm_model.model.box
            ), "model and MDtraj box differ"

        data = Data(
            pos=raw["pos"][selection].reshape(
                -1, omm_model.model.n_waters, omm_model.model.n_sites, 3
            ),
            box=jax.vmap(jnp.diag)(raw["box"][selection]),
            energy=raw["ene"][selection],
            force=None,
        )

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

    def add_forces(self, forces):
        return Data(
            pos=self.pos,
            box=self.box,
            energy=self.energy,
            force=jnp.array(forces).reshape(self.pos.shape),
        )


@pytree_dataclass(frozen=True)
class PreprocessedData:

    pos: jnp.ndarray
    box: jnp.ndarray
    energy: jnp.ndarray
    force: jnp.ndarray | None

    @staticmethod
    def from_data(data: Data) -> "PreprocessedData":

        ## remove com position (but it should already be almost ok)
        pos = unwrap(data.pos, data.box)
        pos = (
            pos
            - pos.mean(axis=(1, 2), keepdims=True)
            + pos.mean(axis=(0, 1, 2), keepdims=True)
        )
        # make sure oxygen positions are in the box
        pos = boxify_oxy(pos, data.box)

        return PreprocessedData(pos, data.box, data.energy, data.force)


@pytree_dataclass(frozen=True)
class DataWithAuxiliary:
    """Data augmented with auxilaries and quaternion signs."""

    pos: Array
    aux: Array | None
    sign: Array
    box: SimulationBox
    force: jnp.ndarray | None
