#!/usr/bin/env python
#SBATCH -p gpu --gres=gpu:1 --exclusive
#SBATCH --time=120:00:00
### #SBATCH --output=output.out

# # Setup ice box
# - get ice config with https://github.com/vitroid/GenIce
# - equilibrate with anisotropic barostat (optional)
# - equilibrate with fixed volume
# - run MD at fixed volume

import numpy as np

import mdtraj as md
import openmm, openmm.app
from openmm import unit
kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)

# from watermodel import *
########################################################################################################

# some useful stuff

import json
from sys import stderr

class WaterModel:
    def __init__(
        self,
        positions,
        box,
        water_type="tip4pew",
        rigidWater=True,
        nonbondedCutoff=1,
        barostat=None,
        external_field=None,
    ):
        if water_type not in [
            "tip3p",
            "tip4pew",
            "tip5p",
            "spce",
            "tip4pew-customLJ",
        ]:
            print(
                f"+++ WARNING: Unknown water_type `{water_type}` +++",
                file=stderr,
            )  # might still work
        if "spc" in water_type:
            n_sites = 3
        else:
            n_sites = int(
                "".join(x for x in water_type if x.isdigit())
            )  # get n_sites from water_type name
        assert (
            len(positions) % n_sites == 0
        ), "mismatch between number of atoms per molecule and total number of atoms"
        n_waters = len(positions) // n_sites

        mdtraj_topology = self.generate_mdtraj_topology(n_waters, n_sites)

        topology = mdtraj_topology.to_openmm()
        topology.setPeriodicBoxVectors(box)

        if nonbondedCutoff > np.diagonal(box).min() / 2:
            epsilon = 1e-5
            nonbondedCutoff = np.diagonal(box).min() / 2 - epsilon
            print(
                f"+++ WARNING: `nonbondedCutoff` too large, changed to {nonbondedCutoff} +++",
                file=stderr,
            )

        ff = openmm.app.ForceField(
            water_type.removesuffix("-customLJ") + ".xml"
        )
        system = ff.createSystem(
            topology,
            nonbondedMethod=openmm.app.PME,
            nonbondedCutoff=nonbondedCutoff,
            rigidWater=rigidWater,
            removeCMMotion=True,
        )
        forces = {f.__class__.__name__: f for f in system.getForces()}
        forces["NonbondedForce"].setUseSwitchingFunction(False)
        forces["NonbondedForce"].setUseDispersionCorrection(True)
        forces["NonbondedForce"].setEwaldErrorTolerance(1e-4)  # default is 5e-4
        oxygen_parameters = forces["NonbondedForce"].getParticleParameters(0)
        self.charge_O = oxygen_parameters[0].value_in_unit(
            unit.elementary_charge
        )
        self.sigma_O = oxygen_parameters[1].value_in_unit(unit.nanometer)
        self.epsilon_O = oxygen_parameters[2].value_in_unit(
            unit.kilojoule_per_mole
        )

        if "customLJ" in water_type:
            # remove LJ interaction
            oxygen_parameters[2] *= 0  # set epsilon_O to zero
            for i in range(0, system.getNumParticles(), n_sites):
                forces["NonbondedForce"].setParticleParameters(
                    i, *oxygen_parameters
                )

            # add equivalent CustomNonbondedForce
            Ulj_str = f"4*{self.epsilon_O}*(({self.sigma_O}/r)^12-({self.sigma_O}/r)^6)"
            energy_expression = f"select(isOxy1*isOxy2, {Ulj_str}, 0)"
            lj_OO = openmm.CustomNonbondedForce(energy_expression)
            lj_OO.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
            lj_OO.setCutoffDistance(nonbondedCutoff)
            lj_OO.setUseLongRangeCorrection(True)

            # Oindices = np.arange(0, system.getNumParticles(), n_sites)
            # lj_OO.addInteractionGroup(Oindices, Oindices) #for obscure reasons this does not work as it should
            # for _ in range(system.getNumParticles()):
            #     lj_OO.addParticle()
            lj_OO.addPerParticleParameter("isOxy")
            for i in range(system.getNumParticles()):
                if i % n_sites == 0:
                    lj_OO.addParticle([1])
                else:
                    lj_OO.addParticle([0])
            bonds = []
            for i in range(0, system.getNumParticles(), n_sites):
                for j in range(1, n_sites):
                    bonds.append((i, i + j))
            lj_OO.createExclusionsFromBonds(bonds, 2)
            system.addForce(lj_OO)
            forces = {f.__class__.__name__: f for f in system.getForces()}

        self.system = system
        self.topology = topology
        self.mdtraj_topology = mdtraj_topology

        self.set_barostat(barostat)
        self.set_external_field(external_field)

        self._positions = np.array(positions)
        self._box = np.array(box)
        self.is_box_orthorombic = not np.count_nonzero(
            box - np.diag(np.diag(box))
        )

        self.n_waters = n_waters
        self.n_sites = n_sites
        self.n_atoms = n_waters * n_sites
        self.water_type = water_type
        self.nonbondedCutoff = nonbondedCutoff
        self.rigidWater = rigidWater

    @property
    def positions(self):
        return self._positions

    @positions.setter
    def positions(self, positions):
        self._positions = np.array(positions)

    @property
    def box(self):
        return self._box

    @box.setter
    def box(self, box):
        self._box = np.array(box)
        self.topology.setPeriodicBoxVectors(box)
        self.system.setDefaultPeriodicBoxVectors(*box)
        self.is_box_orthorombic = not np.count_nonzero(
            box - np.diag(np.diag(box))
        )

    @staticmethod
    def generate_mdtraj_topology(n_waters, n_sites=4):
        assert n_sites >= 3, "only 3 or more sites are supported"
        H = md.element.Element.getBySymbol("H")
        O = md.element.Element.getBySymbol("O")
        VS = md.element.Element.getBySymbol("VS")
        water_top = md.Topology()
        water_top.add_chain()
        for i in range(n_waters):
            water_top.add_residue("HOH", water_top.chain(0))
            water_top.add_atom("O", O, water_top.residue(i))
            water_top.add_atom("H1", H, water_top.residue(i))
            water_top.add_atom("H2", H, water_top.residue(i))
            for _ in range(n_sites - 3):
                water_top.add_atom("M", VS, water_top.residue(i))
            water_top.add_bond(
                water_top.atom(n_sites * i), water_top.atom(n_sites * i + 1)
            )
            water_top.add_bond(
                water_top.atom(n_sites * i), water_top.atom(n_sites * i + 2)
            )
        return water_top

    def set_barostat(
        self, barostat, pressure=1 * unit.bar, temperature=300 * unit.kelvin
    ):
        """
        three possible barostats: 'iso', 'aniso', 'tri'
        see http://docs.openmm.org/latest/api-python/generated/openmm.openmm.MonteCarloAnisotropicBarostat.html
        """
        # make sure no other barostat is defined
        for i in reversed(range(self.system.getNumForces())):
            if isinstance(
                self.system.getForce(i),
                (
                    openmm.MonteCarloBarostat,
                    openmm.MonteCarloAnisotropicBarostat,
                    openmm.MonteCarloFlexibleBarostat,
                ),
            ):
                self.system.removeForce(i)
        # add new one
        if barostat is None:
            pass
        elif barostat == "iso":
            self.system.addForce(
                openmm.MonteCarloBarostat(pressure, temperature)
            )
        elif barostat == "aniso":
            self.system.addForce(
                openmm.MonteCarloAnisotropicBarostat(
                    3 * [pressure], temperature
                )
            )
        elif barostat == "tri":
            self.system.addForce(
                openmm.MonteCarloFlexibleBarostat(pressure, temperature)
            )
        else:
            raise ValueError(f"Unknown barostat: {barostat}")

        self.barostat = barostat

    def set_external_field(self, external_field):
        """external_field should be a 3D list"""
        # make sure no other external_field is defined
        for i in reversed(range(self.system.getNumForces())):
            if isinstance(self.system.getForce(i), openmm.CustomExternalForce):
                self.system.removeForce(i)
        # add new one
        if external_field is None:
            pass
        elif isinstance(external_field, list) and len(external_field) == 3:
            potential = "-q*(x*Ex+y*Ey+z*Ez)"
            force = openmm.CustomExternalForce(potential)
            force.addPerParticleParameter("q")
            force.addGlobalParameter("Ex", external_field[0])
            force.addGlobalParameter("Ey", external_field[1])
            force.addGlobalParameter("Ez", external_field[2])
            nonbonded = [
                f
                for f in self.system.getForces()
                if isinstance(f, openmm.NonbondedForce)
            ][0]
            for i in range(self.system.getNumParticles()):
                charge, sigma, epsilon = nonbonded.getParticleParameters(i)
                force.addParticle(i, [charge])
            self.system.addForce(force)
        else:
            raise ValueError("`external_field` must be a 3-element list")

        self.external_field = external_field

    @property
    def init_info(self):
        init_info = {
            "positions": self._positions.tolist(),
            "box": self._box.tolist(),
            "water_type": self.water_type,
            "rigidWater": self.rigidWater,
            "nonbondedCutoff": self.nonbondedCutoff,
            "barostat": self.barostat,
            "external_field": self.external_field,
        }
        return init_info

    def save_to_json(self, filename):
        with open(filename, "w") as f:
            json.dump(self.init_info, f)

    @staticmethod
    def load_from_json(filename):
        with open(filename, "r") as f:
            init_info = json.load(f)
        return WaterModel(**init_info)

    def setup_simulation(
        self,
        temperature,
        frictionCoeff=1 / unit.picosecond,
        stepSize=1 * unit.femtosecond,
        minimizeEnergy=False,
    ):
        integrator = openmm.LangevinMiddleIntegrator(
            temperature, frictionCoeff, stepSize
        )
        if self.barostat is not None:
            for force in self.system.getForces():
                if isinstance(
                    force,
                    (
                        openmm.MonteCarloBarostat,
                        openmm.MonteCarloAnisotropicBarostat,
                        openmm.MonteCarloFlexibleBarostat,
                    ),
                ):
                    force.setDefaultTemperature(temperature)
        simulation = openmm.app.Simulation(
            self.topology, self.system, integrator
        )
        simulation.context.setPositions(self.positions)

        if minimizeEnergy:
            print(
                "old energy:",
                simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole),
            )
            simulation.minimizeEnergy()  # volume is not changed
            print(
                "new energy:",
                simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole),
            )

        return simulation

    def set_customLJ(
        self, energy_expression, context=None, keep_positions=False
    ):
        """energy_expression: the new Lennard Jones expression, using 'r' for atoms distance"""

        if "customLJ" not in self.water_type:
            raise ValueError(
                f"use water_type={self.water_type}-customLJ to set custom LJ interactions"
            )

        forces = {f.__class__.__name__: f for f in self.system.getForces()}
        forces["CustomNonbondedForce"].setEnergyFunction(
            f"select(isOxy1*isOxy2, {energy_expression}, 0)"
        )

        if context is not None:
            if keep_positions:
                pos = context.getState(getPositions=True).getPositions()
                context.reinitialize()
                context.setPositions(pos)
            else:
                context.reinitialize()

    def get_mdtraj(self, pos=None, box=None):
        if pos is None:
            pos = self._positions
        if box is None:
            box = self._box

        traj = md.Trajectory(pos, self.mdtraj_topology)
        traj.unitcell_vectors = np.resize(box, (len(traj), 3, 3))

        return traj

########################################################################################################


# ## generate ice configuration and setup water model
# see https://github.com/vitroid/GenIce

#some options
import sys

if len(sys.argv) != 2:
  raise TypeError('needs the temperature as arg')

temp = float(sys.argv[1])
# water_type = 'tip4pew-customLJ'
water_type = 'tip4pew'
ice_type = 'XI'
rep = 3*[2]

nonbondedCutoff = 1
rigidWater = True
external_field = None

anisotropic_equilibration = False

# !genice2 -h  #see available ice_types

#setup the model
from genice2.genice import GenIce
from genice2.plugin import Lattice, Format, Molecule
from tempfile import NamedTemporaryFile

gro = GenIce(Lattice(ice_type), rep=rep).generate_ice(Format('gromacs'), water=Molecule(water_type[:5]))
tmp = NamedTemporaryFile()
with open(tmp.name, 'w') as f:
    f.write(gro)
config = openmm.app.GromacsGroFile(tmp.name)

pos = np.array(config.getPositions().value_in_unit(unit.nanometer))
box = np.array(config.getPeriodicBoxVectors().value_in_unit(unit.nanometer))
model = WaterModel(
    positions=pos,
    box=box,
    water_type=water_type,
    nonbondedCutoff=nonbondedCutoff,
    rigidWater=rigidWater,
    external_field=external_field,
)

#save trajectory
#NB: positions can be out of PBC, to avoid breaking molecules
info = f'ice{ice_type}_T{temp:g}_N{model.n_waters}'
if model.water_type != 'tip4pew':
    info = f'{model.water_type}_{info}'
if not rigidWater:
    info = f'flex_{info}'
filename_model = f'model-{info}.json'
filename_MDtraj = f'MDtraj-{info}.npz'

logfile = f'output-{info}.log'
with open(logfile, 'w') as log:
  log.write(f'initializing {info}\n')
  log.write(f'full temperature = {temp}\n')
  log.write(f"n_waters: {model.n_waters}\n")
  log.write(f"orthorombic: {model.is_box_orthorombic}\n")
  log.write(f"nonbondedCutoff: {model.nonbondedCutoff} ({model.nonbondedCutoff/0.316435:g} [sigmaLJ])\n")  # 3.16 is common, 2.5 is ok, below 1.14 is very bad

# ## fixed volume equilibration
#Equilibrate

model.set_barostat = None
with open(logfile, 'a') as log:
  log.write(f"barostat: {model.barostat}\n")

pace = 500
n_iter = 10_000
simulation = model.setup_simulation(temp, minimizeEnergy=(not anisotropic_equilibration))

MDene = np.full(n_iter, np.nan)
MDpos = np.full((n_iter, *model.positions.shape), np.nan)
if model.barostat is None:
    MDbox = np.resize(model.box, (1,3,3))
else:
    MDbox = np.full((n_iter, 3, 3), np.nan)

for n in range(n_iter):
    simulation.step(pace)
    MDene[n] = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    MDpos[n] = simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    if model.barostat is not None:
        MDbox[n] = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometers)

#update model
model.positions = MDpos[-1]
model.box = MDbox[-1]
model.save_to_json(filename_model)

with open(logfile, 'a') as log:
  log.write('equilibration done\n')

# ## run MD storing model and trajectory
#production run

pace = 500
n_iter = 100_000
# simulation = model.setup_simulation(temp)

MDene = np.full(n_iter, np.nan)
MDpos = np.full((n_iter, *model.positions.shape), np.nan)
if model.barostat is None:
    MDbox = np.resize(model.box, (1,3,3))
else:
    MDbox = np.full((n_iter, 3, 3), np.nan)

for n in range(n_iter):
    simulation.step(pace)
    MDene[n] = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    MDpos[n] = simulation.context.getState(getPositions=True, enforcePeriodicBox=False).getPositions(asNumpy=True).value_in_unit(unit.nanometers)
    if model.barostat is not None:
        MDbox[n] = simulation.context.getState().getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometers)
    if (n+1) % n_iter//10 == 0:
        with open(logfile, 'a') as log:
          np.savez(filename_MDtraj, pos=MDpos[:n+1], box=MDbox[:n+1], ene=MDene[:n+1])
          log.write(f'step {n+1}\n')

np.savez(filename_MDtraj, pos=MDpos, box=MDbox, ene=MDene)

with open(logfile, 'a') as log:
  log.write('the end\n')
