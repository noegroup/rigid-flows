import logging

import jax
import lenses
import numpy as np
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass

from flox import geom

from .system import ErrorHandling, OpenMMEnergyModel, SimulationBox, SystemSpecification
from .utils import smooth_maximum

logger = logging.getLogger("rigid-flows")


def unwrap(pos: jnp.ndarray, box: SimulationBox):
    """ "Using as reference the first configuration"""
    return jnp.where(
        jnp.abs(pos - pos[0]) / box.size > 0.5,
        pos - jnp.sign(pos - pos[0]) * box.size,
        pos,
    )


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
        box: SimulationBox,
    ):
        path = f"{specs.path}/MDtraj-{specs}.npz"
        logging.info(f"Loading data from {path}")
        raw = np.load(path)
        data = Data(*map(jnp.array, raw.values()))
        data = lenses.bind(data).pos.set(
            data.pos.reshape(data.pos.shape[0], -1, 4, 3)
        )
        data = lenses.bind(data).pos.set(unwrap(data.pos, box))
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
    modes: jnp.ndarray
    stds: jnp.ndarray

    @staticmethod
    def from_data(data: Data, box: SimulationBox) -> "PreprocessedData":
        oxy = data.pos[:, :, 0]

        modes = jax.vmap(
            jax.vmap(smooth_maximum, in_axes=1, out_axes=0),
            in_axes=2,
            out_axes=1,
        )(oxy)

        modes = jnp.mean(oxy, axis=0)
        stds = jnp.std(oxy, axis=0)

        # # unwrap positions
        # pos = modes[:, None] + geom.Torus(box.size).tangent(
        #     data.pos, data.pos - modes[:, None]
        # )

        return PreprocessedData(
            data.pos, data.box, data.energy, data.force, modes, stds
        )

    def estimate_com_stats(self):
        pos = self.pos.reshape(self.pos.shape[0], -1, 4, 3)
        coms = pos.mean(axis=(1, 2))
        return jnp.mean(coms, axis=0), jnp.std(coms, axis=0)


@pytree_dataclass(frozen=True)
class DataWithAuxiliary:
    """Data augmented with auxilaries and quaternion signs."""

    pos: Array
    aux: Array
    sign: Array
    box: SimulationBox
    force: jnp.ndarray | None
