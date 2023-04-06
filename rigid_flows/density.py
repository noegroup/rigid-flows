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
from .system import SPATIAL_DIM, OpenMMEnergyModel, SimulationBox, wrap_openmm_model

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
        stored_energies: bool = True,
    ):
        self.aux_model = aux_model
        self.sys_specs = sys_specs
        self.omm_model = omm_model
        self.data = data

        if stored_energies:
            self.omm_energies = self.compute_energies(
                DataWithAuxiliary(self.data.pos, None, jnp.array([]), self.box),
                omm=True,
                aux=False,
                has_batch_dim=True,
            )["omm"]
            if not jnp.allclose(
                self.omm_energies, self.data.energy / self.omm_model.kbT, rtol=1e-4
            ):
                raise ValueError(
                    "omm_model energies are inconsisten with the stored ones"
                )
        else:
            self.omm_energies = None

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
            energy = partial(
                wrap_openmm_model(self.omm_model)[0],
                box=None,
                has_batch_dim=has_batch_dim,
            )
            results["omm"] = energy(inp.pos)
        if aux and self.aux_model is not None:
            results["aux"] = -self.aux_model.log_prob(inp.aux).sum(axis=(-2, -1))

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
        """Samples randomly from the target (data) distribution."""
        idx = jax.random.randint(key, minval=0, maxval=len(self.data.pos), shape=())
        return self.sample_idx(key, idx)

    def sample_idx(self, key: KeyArray, idx: int) -> Transformed[DataWithAuxiliary]:
        """Samples from the target (data) distribution at specific index.

        Positions are taken from MD trajectory.

        Auxiliaries are drawn from a standard normal distribution.

        Quaternion signs are drawn from {-1, 1} uniformily random.

        Args:
            key (KeyArray): PRNG Key

        Returns:
            Transformed[AugmentedData]: Sample from the target distribution.
        """
        pos = self.data.pos[idx]

        chain = key_chain(key)
        if self.aux_model is None:
            aux = None
        else:
            aux = self.aux_model.sample(seed=next(chain))

        sign = jnp.sign(
            jax.random.normal(next(chain), shape=(self.sys_specs.num_molecules, 1))
        )

        sample = DataWithAuxiliary(pos, aux, sign, self.box)
        if self.omm_energies is not None:
            energy = self.omm_energies[idx]
            if self.aux_model is not None:
                energy = (
                    energy
                    + self.compute_energies(
                        sample, omm=False, aux=True, has_batch_dim=False
                    )["aux"]
                )
        else:
            energy = self.potential(sample)

        return Transformed(sample, energy)

    @staticmethod
    def from_specs(
        use_auxiliary: bool,
        sys_specs: SystemSpecification,
        selection: slice = np.s_[:],
    ):

        omm_model = OpenMMEnergyModel.from_specs(sys_specs)

        data = Data.from_specs(sys_specs, omm_model, selection)
        data = PreprocessedData.from_data(data)

        if use_auxiliary:
            auxiliary_shape = [omm_model.model.n_molecules, SPATIAL_DIM]
            aux_model = tfp.distributions.Normal(
                jnp.zeros(auxiliary_shape), jnp.ones(auxiliary_shape)
            )
        else:
            aux_model = None

        return OpenMMDensity(
            sys_specs=sys_specs,
            omm_model=omm_model,
            aux_model=aux_model,
            data=data,
        )
