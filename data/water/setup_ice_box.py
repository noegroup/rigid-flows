#!/usr/bin/env python
#SBATCH -p gpu --gres=gpu:1 --exclusive
#SBATCH --time=120:00:00
### #SBATCH --output=output.out

# # Setup ice box
# - get ice config with https://github.com/vitroid/GenIce
# - equilibrate with anisotropic barostat (optional)
# - equilibrate with fixed volume
# - run MD at fixed volume

import mdtraj as md
import numpy as np
import openmm
import openmm.app
from openmm import unit

kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole / unit.kelvin)

import sys

from rigid_flows.systems.watermodel import WaterModel

# ## generate ice configuration and setup water model
# see https://github.com/vitroid/GenIce

# some options
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--temp", type=float, required=True)
# parser.add_argument("--rep", type=int, required=True, help="number of repetitions, sets the box size")
# args = parser.parse_args()
# temp = args.temp
# size = args.rep

if len(sys.argv) != 3:
    raise TypeError("needs the size and temperature as arg")

size = int(sys.argv[1])
temp = float(sys.argv[2])

water_type = "tip4pew"
ice_type = "XI"
rep = 3 * [size]

nonbondedCutoff = 1
rigidWater = True
external_field = None

anisotropic_equilibration = False

# !genice2 -h  #see available ice_types

from tempfile import NamedTemporaryFile

# setup the model
from genice2.genice import GenIce
from genice2.plugin import Format, Lattice, Molecule

gro = GenIce(Lattice(ice_type), rep=rep).generate_ice(
    Format("gromacs"), water=Molecule(water_type[:5])
)
tmp = NamedTemporaryFile()
with open(tmp.name, "w") as f:
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

# save trajectory
# NB: positions can be out of PBC, to avoid breaking molecules
info = f"ice{ice_type}_T{temp:g}_N{model.n_molecules}"
if model.water_type != "tip4pew":
    info = f"{model.water_type}_{info}"
if not rigidWater:
    info = f"flex_{info}"
filename_model = f"model-{info}.json"
filename_MDtraj = f"MDtraj-{info}.npz"

logfile = f"output-{info}.log"
with open(logfile, "w") as log:
    log.write(f"initializing {info}\n")
    log.write(f"full temperature = {temp}\n")
    log.write(f"n_molecules: {model.n_molecules}\n")
    log.write(f"orthorombic: {model.is_box_orthorombic}\n")
    log.write(
        f"nonbondedCutoff: {model.nonbondedCutoff} ({model.nonbondedCutoff/0.316435:g} [sigmaLJ])\n"
    )  # 3.16 is common, 2.5 is ok, below 1.14 is very bad

# ## fixed volume equilibration
# Equilibrate

model.set_barostat = None
with open(logfile, "a") as log:
    log.write(f"barostat: {model.barostat}\n")

pace = 500
n_iter = 10_000
simulation = model.setup_simulation(
    temp, minimizeEnergy=(not anisotropic_equilibration)
)

MDene = np.full(n_iter, np.nan)
MDpos = np.full((n_iter, *model.positions.shape), np.nan)
if model.barostat is None:
    MDbox = np.resize(model.box, (1, 3, 3))
else:
    MDbox = np.full((n_iter, 3, 3), np.nan)

for n in range(n_iter):
    simulation.step(pace)
    MDene[n] = (
        simulation.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    MDpos[n] = (
        simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometers)
    )
    if model.barostat is not None:
        MDbox[n] = (
            simulation.context.getState()
            .getPeriodicBoxVectors(asNumpy=True)
            .value_in_unit(unit.nanometers)
        )

# update model
model.positions = MDpos[-1]
model.box = MDbox[-1]
model.save_to_json(filename_model)

with open(logfile, "a") as log:
    log.write("equilibration done\n")

# ## run MD storing model and trajectory
# production run

pace = 500
n_iter = 100_000
# simulation = model.setup_simulation(temp)

MDene = np.full(n_iter, np.nan)
MDpos = np.full((n_iter, *model.positions.shape), np.nan)
if model.barostat is None:
    MDbox = np.resize(model.box, (1, 3, 3))
else:
    MDbox = np.full((n_iter, 3, 3), np.nan)

for n in range(n_iter):
    simulation.step(pace)
    MDene[n] = (
        simulation.context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilojoule_per_mole)
    )
    MDpos[n] = (
        simulation.context.getState(getPositions=True, enforcePeriodicBox=False)
        .getPositions(asNumpy=True)
        .value_in_unit(unit.nanometers)
    )
    if model.barostat is not None:
        MDbox[n] = (
            simulation.context.getState()
            .getPeriodicBoxVectors(asNumpy=True)
            .value_in_unit(unit.nanometers)
        )
    if (n + 1) % n_iter // 10 == 0:
        with open(logfile, "a") as log:
            np.savez(
                filename_MDtraj,
                pos=MDpos[: n + 1],
                box=MDbox[: n + 1],
                ene=MDene[: n + 1],
            )
            log.write(f"step {n+1}\n")

np.savez(filename_MDtraj, pos=MDpos, box=MDbox, ene=MDene)

with open(logfile, "a") as log:
    log.write("the end\n")
