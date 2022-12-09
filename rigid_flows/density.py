from collections.abc import Callable
from functools import partial
from typing import Protocol, TypeVar

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfp  # type: ignore
from jax import Array

from flox.flow import Transformed
from flox.util import key_chain

from .data import AugmentedData, Data
from .flow import InternalCoordinates, State
from .specs import BaseSpecification, SystemSpecification, TargetSpecification
from .system import OpenMMEnergyModel, SimulationBox, wrap_openmm_model

T = TypeVar("T")

KeyArray = Array | jax.random.PRNGKeyArray


class DensityModel(Protocol[T]):
    def potential(self, inp: T) -> Array:
        ...

    def sample(self, key: KeyArray) -> Transformed[T]:
        ...


class BaseDensity(DensityModel[State]):
    def __init__(
        self,
        box: SimulationBox,
        rot_modes: Array,
        rot_concentration: Array,
        # pos_means: Array,
        # pos_stds: Array,
        pos_modes: Array,
        pos_concentration,
        aux_means: Array,
        aux_stds: Array,
    ):
        self.box = box
        self.rot_model = tfp.distributions.VonMisesFisher(
            rot_modes, rot_concentration
        )
        # self.pos_model = tfp.distributions.Normal(pos_means, pos_stds)
        self.pos_model = tfp.distributions.VonMisesFisher(
            pos_modes, pos_concentration
        )
        self.aux_model = tfp.distributions.Normal(aux_means, aux_stds)

    def potential(self, inp: State) -> Array:
        """Evaluate the base density for a state

        Args:
            state (State): the state to be evaluated

        Returns:
            Array: the energy of the state
        """
        # symmetrize latent distribution over rotations
        rot_prob = jax.nn.logsumexp(
            jnp.stack(
                [
                    self.rot_model.log_prob(inp.rot),
                    self.rot_model.log_prob(-inp.rot),
                ]
            ),
            axis=0,
        ).sum()
        pos_prob = self.pos_model.log_prob(inp.pos).sum()
        aux_prob = self.aux_model.log_prob(inp.aux).sum()
        return -(rot_prob + aux_prob + pos_prob)

    def sample(self, key: KeyArray) -> Transformed[State]:
        """Samples from the base density

        Args:
            key (KeyArray): PRNG Key
            box (Box): simulation box

        Returns:
            Transformed[State]: a state sampled from the prior density
        """
        chain = key_chain(key)

        rot = self.rot_model.sample(seed=next(chain))
        rot = rot * jnp.sign(
            jax.random.normal(next(chain), shape=(rot.shape[0], 1))
        )
        pos = self.pos_model.sample(seed=next(chain))
        ics = InternalCoordinates()
        aux = self.aux_model.sample(seed=next(chain))
        state = State(rot, pos, ics, aux, self.box)
        log_prob = self.potential(state)
        return Transformed(state, log_prob)

    @staticmethod
    def from_specs(
        system_specs: SystemSpecification,
        base_specs: BaseSpecification,
        box: SimulationBox,
        auxiliary_shape: tuple[int, ...],
    ):
        return BaseDensity(
            box=box,
            rot_modes=jnp.tile(
                jnp.array([1.0, 0.0, 0.0, 0.0])[None],
                (system_specs.num_molecules, 1),
            ),
            rot_concentration=base_specs.rot_concentration
            * jnp.ones((system_specs.num_molecules,)),
            # pos_means=jnp.zeros((system_specs.num_molecules, 3,)),
            # pos_stds=jnp.ones((system_specs.num_molecules, 3)),
            pos_modes=jnp.tile(
                jnp.array([1.0, 0.0])[None, None],
                (system_specs.num_molecules, 3, 1),
            ),
            pos_concentration=base_specs.pos_concentration
            * jnp.ones((system_specs.num_molecules, 3)),
            aux_means=jnp.zeros(auxiliary_shape),
            aux_stds=jnp.ones(auxiliary_shape),
        )


def cutoff_potential(
    potential: Callable[[Array], Array], reference: Array, threshold: float
):
    def approximate_potential(inp):
        return 0.5 * jnp.square(inp - reference).sum()

    def eval_fwd(inp):
        original, grad = jax.value_and_grad(potential)(inp)
        gnorm = jnp.sqrt(1e-12 + jnp.sum(jnp.square(grad)))

        approx, grad_approx = jax.value_and_grad(approximate_potential)(inp)
        gnorm_approx = jnp.sqrt(1e-12 + jnp.sum(jnp.square(grad_approx)))
        grad_approx = grad_approx / gnorm_approx * threshold

        out = jnp.where(gnorm > threshold, approx, original)
        grad = jnp.where(gnorm > threshold, grad_approx, grad)
        return out, grad

    def eval_bwd(g_inp, g_out):
        return (g_out * g_inp,)

    @jax.custom_vjp
    def eval(inp):
        return eval_fwd(inp)[0]

    eval.defvjp(eval_fwd, eval_bwd)

    return eval


class TargetDensity(DensityModel[AugmentedData]):
    def __init__(
        self,
        auxiliary_shape: tuple[int, ...],
        sys_specs: SystemSpecification,
        model: OpenMMEnergyModel,
        data: Data | None = None,
        cutoff_threshold: float | None = None,
    ):
        aux_means = jnp.zeros(auxiliary_shape)
        aux_stds = jnp.ones(auxiliary_shape)
        self.aux_model = tfp.distributions.Normal(aux_means, aux_stds)
        self.sys_specs = sys_specs
        self.model = model
        self.data = data
        self.cutoff = cutoff_threshold

        # set simulation box
        if self.sys_specs.fixed_box:
            self.model.set_box(self.box)

    @property
    def box(self) -> SimulationBox:
        if self.data is None:
            raise ValueError("Data not loaded.")
        return SimulationBox(self.data.box[0])

    def potential(self, inp: AugmentedData) -> Array:
        """Evaluate the target density for a state

        Args:
            state (State): the state to be evaluated

        Returns:
            Array: the energy of the state

        """
        aux_prob = self.aux_model.log_prob(inp.aux).sum()

        if self.sys_specs.fixed_box:
            box = None
        else:
            box = inp.box.size

        energy = partial(
            wrap_openmm_model(self.model),
            box=box,
            has_batch_dim=False,
        )

        if self.cutoff is not None:
            if self.data is None:
                raise ValueError("cutoff != None requires data.")
            else:
                energy = cutoff_potential(energy, self.data.pos[0], self.cutoff)

        pot = energy(inp.pos)
        return -aux_prob + pot

    def sample(self, key: KeyArray) -> Transformed[AugmentedData]:
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
            pos = self.data.pos[idx].reshape(-1, 4, 3)
            energy = self.data.energy[idx]
            if self.data.force is not None:
                force = self.data.force[idx]
            else:
                force = None
            aux = self.aux_model.sample(seed=next(chain))
            sign = jnp.sign(
                jax.random.normal(
                    next(chain), shape=(self.sys_specs.num_molecules, 1)
                )
            )
            return Transformed(
                AugmentedData(pos, aux, sign, box, force), energy
            )

    @staticmethod
    def from_specs(
        auxiliary_shape: tuple[int, ...],
        target_specs: TargetSpecification,
        sys_specs: SystemSpecification,
    ):
        data = Data.from_specs(sys_specs)
        model = OpenMMEnergyModel.from_specs(sys_specs)
        if sys_specs.recompute_forces:
            data = data.recompute_forces(model)

        return TargetDensity(
            auxiliary_shape=auxiliary_shape,
            sys_specs=sys_specs,
            model=model,
            data=data,
            cutoff_threshold=target_specs.cutoff_threshold,
        )
