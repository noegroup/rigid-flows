import enum
import inspect
import logging
from functools import partial
from typing import Callable

import jax
import numpy as np
import openmm  # type: ignore
from jax import Array
from jax import numpy as jnp
from jax_dataclasses import pytree_dataclass
from openmm import unit  # type: ignore

from .specs import SystemSpecification
from .systems.watermodel import WaterModel


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
        self.kbT = (
            unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
                unit.kilojoule_per_mole / unit.kelvin
            )
            * temperature
        )
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
        self.context = openmm.Context(model.system, integrator)
        self.error_handling = error_handling

    def set_softcore_cutoff(
        self,
        cutoff: float | None,
        type: str,
        keep_positions: bool = False,  # not really needed here
        **kwargs,
    ):
        """
        cutoff is in units of LJ sigma
        """
        my_lennard_jones = partial(
            lennard_jones, sigma=self.model.sigma_O, epsilon=self.model.epsilon_O
        )
        match type:
            case "linear":
                expr = get_approx_expr(
                    my_lennard_jones,
                    approx_with_linear,
                    cutoff=cutoff * self.model.sigma_O,
                )
            case "square":
                if "slope" not in kwargs:
                    raise ValueError(
                        "'slope' needed in kwargs when using square approximation."
                    )
                expr = get_approx_expr(
                    my_lennard_jones,
                    approx_with_square,
                    cutoff=cutoff * self.model.sigma_O,
                    slope=kwargs["slope"],
                )
            case _:
                expr = parse_jaxpr(my_lennard_jones)

        self.model.set_customLJ(expr, self.context, keep_positions=keep_positions)

    def set_box(self, box: SimulationBox):
        box_vectors = np.diag(np.array(box.size))
        self.context.setPeriodicBoxVectors(*box_vectors)

    def compute_energies_and_forces(
        self,
        pos: np.ndarray,
        box: np.ndarray | None,
        error_handling: ErrorHandling | None = None,
    ):
        energies = np.empty(pos.shape[0], dtype=np.float32)
        forces = np.empty_like(pos, dtype=np.float32)

        if error_handling is None:
            error_handling = self.error_handling

        if box is not None:
            assert box.shape == (3, 3), f"box.shape = {box.shape}"
            self.context.setPeriodicBoxVectors(*box)

        # iterate over batch dimension
        for i in range(len(pos)):

            energy = jnp.empty_like(energies[i])
            force = jnp.empty_like(forces[i])

            try:
                self.context.setPositions(pos[i])
                # self.context.computeVirtualSites()

                state = self.context.getState(getEnergy=True, getForces=True)
                energy = (
                    state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                    / self.kbT
                )
                force = (
                    state.getForces(asNumpy=True).value_in_unit(
                        unit.kilojoule_per_mole / unit.nanometer
                    )
                    / self.kbT
                )
            except Exception as e:
                match self.error_handling:
                    case ErrorHandling.RaiseException:
                        raise e
                    case ErrorHandling.LogWarning:
                        logging.warning(str(e))
                    case _:
                        pass
            finally:
                energies[i] = energy
                forces[i] = force

        return energies, forces

    @staticmethod
    def from_specs(specs: SystemSpecification):
        path = f"{specs.path}/model-{specs}.json"
        logging.info(f"Loading OpenMM model specs from {path}")
        model = WaterModel.load_from_json(path)
        return OpenMMEnergyModel(model, specs.temperature)


def wrap_openmm_model(model: OpenMMEnergyModel):
    def compute_energy_and_forces(pos: Array, box: Array | None, has_batch_dim: bool):

        if box is not None:
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

    def eval_fwd(pos: Array, box: Array | None, has_batch_dim: bool):
        energy, force = compute_energy_and_forces(pos, box, has_batch_dim)
        return energy, force

    def eval_bwd(_1, _2, res, g):
        force = res
        return (-g * force,)

    @partial(jax.custom_vjp, nondiff_argnums=(1, 2))
    def eval(pos: Array, box: Array | None, has_batch_dim: bool):
        return eval_fwd(pos, box, has_batch_dim)[0]

    eval.defvjp(eval_fwd, eval_bwd)

    return eval


def lennard_jones(r, sigma, epsilon):
    return 4 * epsilon * ((sigma / r) ** 12 - (sigma / r) ** 6)


def parse_jaxpr(fn: Callable, args: tuple[str] | None = None, symbols: dict = {}):

    placeholders = []

    if args is None:
        sig = inspect.signature(fn)
        for key, val in sig.parameters.items():
            if val.default == inspect._empty:
                placeholders.append(key)

    jaxpr_kwargs = {key: jax.ShapedArray((), dtype=jnp.float32) for key in placeholders}
    jaxpr = jax.make_jaxpr(fn)(**jaxpr_kwargs)
    symbols = {
        str(sym): val if val not in symbols else symbols[val]
        for (sym, val) in zip(jaxpr.jaxpr.invars, placeholders)
    }

    def fetch_symbol(arg):
        if isinstance(arg, jax.core.Var):
            return symbols[str(arg)]
        elif isinstance(arg, jax.core.Literal):
            return str(arg)
        else:
            raise ValueError()

    for eqn in jaxpr.eqns:
        match eqn.primitive.name:
            case "mul":
                fst, snd = map(fetch_symbol, eqn.invars)
                out = str(eqn.outvars[0])
                symbols[out] = f"({fst} * {snd})"
            case "div":
                fst, snd = map(fetch_symbol, eqn.invars)
                out = str(eqn.outvars[0])
                symbols[out] = f"({fst} / {snd})"
            case "integer_pow":
                (fst,) = map(fetch_symbol, eqn.invars)
                exp = str(eqn.params["y"])
                out = str(eqn.outvars[0])
                symbols[out] = f"{fst}^{exp}"
            case "sub":
                fst, snd = map(fetch_symbol, eqn.invars)
                out = str(eqn.outvars[0])
                symbols[out] = f"({fst} - {snd})"
            case "add":
                fst, snd = map(fetch_symbol, eqn.invars)
                out = str(eqn.outvars[0])
                symbols[out] = f"({fst} + {snd})"
            case "convert_element_type":
                (fst,) = map(fetch_symbol, eqn.invars)
                out = str(eqn.outvars[0])
                symbols[out] = f"{fst}"
            case _:
                raise ValueError(eqn.primitive.name)

    return tuple(map(fetch_symbol, jaxpr.jaxpr.outvars))


def approx_with_square(original, cutoff: Array, slope: float):
    y0, dyx0 = jax.value_and_grad(original)(cutoff)
    a = slope
    b = dyx0 - 2 * a * cutoff
    c = y0 - a * cutoff**2 - b * cutoff

    def approx(r):
        return a * r**2 + b * r + c

    return approx


def approx_with_linear(original, cutoff: Array):
    y0, dyx0 = jax.value_and_grad(original)(cutoff)
    a = dyx0
    b = y0 - a * cutoff

    def approx(r):
        return a * r + b

    return approx


def get_approx_expr(fun, approximation, cutoff, **kwargs):
    original = parse_jaxpr(fun)[0]
    fun_ = partial(fun, **kwargs)
    approx = parse_jaxpr(approximation(fun_, cutoff))[0]
    sig = inspect.signature(fun)
    inp = tuple(sig.parameters.keys())[0]
    filter = f"step({inp} - {cutoff})"
    return f"select({filter}, {original}, {approx})"
