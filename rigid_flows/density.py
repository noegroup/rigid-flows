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
        com_model: tfp.distributions.Distribution | None,
        data: PreprocessedData,
    ):
        self.aux_model = aux_model
        self.com_model = com_model
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
        com: bool,
        has_batch_dim: bool,
    ):

        results = {}

        if omm:
            pos = inp.pos
            if self.sys_specs.fixed_box:
                box = None
            else:
                box = inp.box.size

            energy = partial(
                wrap_openmm_model(self.omm_model)[0],
                box=box,
                has_batch_dim=has_batch_dim,
            )
            results["omm"] = energy(pos)
        if com and self.com_model is not None:
            data_com = inp.pos.mean(axis=(-2, -3))
            com_prob = self.com_model.log_prob(data_com)
            for _ in range(len(self.com_model.batch_shape)):
                com_prob = com_prob.sum(-1)
            results["com"] = -com_prob
        if aux and self.aux_model is not None:
            aux_prob = self.aux_model.log_prob(inp.aux)
            for _ in range(len(self.aux_model.batch_shape)):
                aux_prob = aux_prob.sum(-1)
            results["aux"] = -aux_prob

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
                inp, omm=True, aux=True, com=True, has_batch_dim=False
            ).values(),
            jnp.zeros(()),
        )

    def sample(self, key: KeyArray) -> Transformed[DataWithAuxiliary]:
        """Samples from the target (data) distribution.

        Auxiliaries are drawn from a standard normal distribution.

        Quaternion signs are drawn from {-1, 1} uniformily random.

        Args:
            key (KeyArray): PRNG Key

        Returns:
            Transformed[AugmentedData]: Sample from the target distribution.
        """
        if self.data is None:
            raise NotImplementedError(
                "Sampling without provided data is not implemented."
            )
        else:
            chain = key_chain(key)
            idx = jax.random.randint(
                next(chain), minval=0, maxval=len(self.data.pos), shape=()
            )
            box = SimulationBox(self.data.box[idx])
            pos = self.data.pos[idx].reshape(
                self.omm_model.model.n_waters, self.omm_model.model.n_sites, 3
            )

            if self.aux_model is None:
                aux = None
            else:
                aux = self.aux_model.sample(seed=next(chain))
            if self.com_model is not None:
                com = self.com_model.sample(seed=next(chain))
                pos = (
                    pos - pos.mean(axis=(0, 1), keepdims=True) + com[None, None]
                )

            force = None
            sign = jnp.sign(
                jax.random.normal(
                    next(chain), shape=(self.sys_specs.num_molecules, 1)
                )
            )

            sample = DataWithAuxiliary(pos, aux, sign, box, force)
            energy = self.potential(sample)
            return Transformed(sample, energy)

    @staticmethod
    def from_specs(
        auxiliary_shape: tuple[int, ...] | None,
        sys_specs: SystemSpecification,
    ):

        omm_model = OpenMMEnergyModel.from_specs(sys_specs)
        if sys_specs.softcore_cutoff is not None:
            omm_model.set_softcore_cutoff(
                sys_specs.softcore_cutoff,
                sys_specs.softcore_potential,
                sys_specs.softcore_slope,
            )

        box = SimulationBox(jnp.diag(omm_model.model.box))

        data = Data.from_specs(sys_specs, box)
        if sys_specs.recompute_forces:
            data = data.recompute_forces(omm_model)
        elif sys_specs.forces_path:
            forces = np.load(sys_specs.forces_path)["forces"]
            data = data.add_forces(forces)
        if sys_specs.store_forces and sys_specs.forces_path:
            assert data.force is not None
            np.savez(sys_specs.forces_path, forces=np.array(data.force))

        data = PreprocessedData.from_data(
            data,
            SimulationBox(jnp.diag(omm_model.model.box)),
            sys_specs.fixed_com,
        )

        if auxiliary_shape is None:
            aux_model = None
        else:
            aux_model = tfp.distributions.Normal(
                jnp.zeros(auxiliary_shape), jnp.ones(auxiliary_shape)
            )

        if sys_specs.fixed_com:
            com_model = None
        else:
            com_model = tfp.distributions.Normal(*data.estimate_com_stats())

        return OpenMMDensity(
            sys_specs=sys_specs,
            omm_model=omm_model,
            aux_model=aux_model,
            com_model=com_model,
            data=data,
        )
