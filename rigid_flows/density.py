from functools import partial
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax as tfp  # type: ignore
from flox.flow import Transformed
from flox.util import key_chain
from jax import Array

from .data import Data, DataWithAuxiliary, PreprocessedData
from .specs import SystemSpecification
from .system import OpenMMEnergyModel, SimulationBox, wrap_openmm_model

T = TypeVar("T")

KeyArray = Array | jax.random.PRNGKeyArray


class DensityModel(Protocol[T]):
    def potential(self, inp: T) -> Array:
        ...

    def sample(self, key: KeyArray) -> Transformed[T]:
        ...


class OpenMMDensity(DensityModel[DataWithAuxiliary]):
    def __init__(
        self,
        sys_specs: SystemSpecification,
        omm_model: OpenMMEnergyModel,
        aux_model: tfp.distributions.Distribution | None,
        data: PreprocessedData,
    ):
        self.aux_model = aux_model
        self.sys_specs = sys_specs
        self.omm_model = omm_model
        self.data = data

    @property
    def box(self) -> SimulationBox:
        return SimulationBox(jnp.diag(self.omm_model.model.box))

    def compute_energies(
        self,
        inp: DataWithAuxiliary,
        omm: bool,
        aux: bool,
        has_batch_dim: bool,
    ):

        results = {}

        if omm:
            if not self.sys_specs.fixed_box:
                raise NotImplementedError()

            energy = partial(
                wrap_openmm_model(self.omm_model)[0],
                box=None,
                has_batch_dim=has_batch_dim,
            )
            results["omm"] = energy(inp.pos)
        if aux and self.aux_model is not None:
            results["aux"] = 0
            # -self.aux_model.log_prob(inp.aux).sum(
            # axis=(-2, -1)
            # )

        return results

    def potential(self, inp: DataWithAuxiliary) -> Array:
        """Evaluate the target density for a state

        Args:
            state (State): the state to be evaluated

        Returns:
            Array: the energy of the state

        """
        return sum(
            self.compute_energies(
                inp, omm=True, aux=True, has_batch_dim=False
            ).values(),
            jnp.zeros(()),
        )

    def sample(self, key: KeyArray) -> Transformed[DataWithAuxiliary]:
        """Samples from the target (data) distribution."""
        idx = jax.random.randint(key, minval=0, maxval=len(self.data.pos), shape=())
        return self.sample_idx(key, idx)

    def sample_idx(
        self, key: KeyArray, idx: jnp.ndarray
    ) -> Transformed[DataWithAuxiliary]:
        """Samples from the target (data) distribution.

        Positions are taken from MD trajectory.

        Auxiliaries are drawn from a standard normal distribution.

        Quaternion signs are drawn from {-1, 1} uniformily random.

        Args:
            key (KeyArray): PRNG Key

        Returns:
            Transformed[AugmentedData]: Sample from the target distribution.
        """
        pos = self.data.pos[idx].reshape(-1, 4, 3)

        chain = key_chain(key)
        if self.aux_model is None:
            aux = None
        else:
            aux = self.aux_model.sample(seed=next(chain))

        force = None
        sign = jnp.sign(
            jax.random.normal(next(chain), shape=(self.sys_specs.num_molecules, 1))
        )

        sample = DataWithAuxiliary(pos, aux, sign, self.box, force)
        # energy = self.data.energy[idx] / self.omm_model.kbT
        # if self.aux_model is not None:
        #     energy = energy + self.compute_energies(sample, omm=False, aux=True, has_batch_dim=False)["aux"]
        energy = self.potential(sample)
        return Transformed(sample, energy)

    @staticmethod
    def from_specs(
        auxiliary_shape: tuple[int, ...] | None,
        sys_specs: SystemSpecification,
        selection: slice = np.s_[:],
    ):

        omm_model = OpenMMEnergyModel.from_specs(sys_specs)
        omm_model.set_softcore_cutoff(
            sys_specs.softcore_cutoff,
            sys_specs.softcore_potential,
            sys_specs.softcore_slope,
        )

        data = Data.from_specs(sys_specs, omm_model, selection)
        if sys_specs.recompute_forces:
            data = data.recompute_forces(omm_model)
        elif sys_specs.forces_path:
            forces = np.load(sys_specs.forces_path)["forces"]
            data = data.add_forces(forces)
        if sys_specs.store_forces and sys_specs.forces_path:
            assert data.force is not None
            np.savez(sys_specs.forces_path, forces=np.array(data.force))

        data = PreprocessedData.from_data(data)

        if auxiliary_shape is None:
            aux_model = None
        else:
            aux_model = tfp.distributions.Normal(
                jnp.zeros(auxiliary_shape), jnp.ones(auxiliary_shape)
            )

        return OpenMMDensity(
            sys_specs=sys_specs,
            omm_model=omm_model,
            aux_model=aux_model,
            data=data,
        )
