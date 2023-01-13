#!/usr/bin/env python

# some useful stuff

import json
from sys import stderr

import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
import openmm
import openmm.app
from openmm import unit

try:
    import nglview as nv  # type: ignore
except:
    print("+++ WARNING: nglview not available +++", file=stderr)
    nv = None


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
            box - np.diag(np.diagonal(box))
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
            box - np.diag(np.diagonal(box))
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

    def plot_2Dview(self, pos=None, box=None, toPBC=False):
        if pos is None:
            pos = self._positions
        if box is None:
            box = self._box

        if len(pos.squeeze().shape) == 2:
            marker = "o"
            alpha = 1
        elif len(pos.squeeze().shape) == 3:
            marker = "."
            alpha = 0.05
        else:
            raise ValueError("pos should be of shape (nframes, natoms, 3)")
        if pos.shape[-1] != 3:
            raise ValueError("pos should be in 3D")

        av_box = box.mean(axis=0) if len(box.shape) == 3 else box
        if av_box.shape != (3, 3):
            raise ValueError("box should be a 3x3 matrix")

        if toPBC:
            if self.is_box_orthorombic:
                mypos = (pos / np.diagonal(av_box) % 1) * np.diagonal(av_box)
            else:
                raise NotImplementedError(
                    "only available for fixed orthorombic box"
                )
        else:
            mypos = pos

        plt.figure(figsize=(15, 4))
        for i in range(3):
            ii = (i + 1) % 3
            iii = (i + 2) % 3
            plt.subplot(1, 3, 1 + i)

            # draw particles
            plt.scatter(
                mypos[..., 1 :: self.n_sites, i],
                mypos[..., 1 :: self.n_sites, ii],
                marker=marker,
                alpha=alpha,
                c="gray",
            )
            plt.scatter(
                mypos[..., 2 :: self.n_sites, i],
                mypos[..., 2 :: self.n_sites, ii],
                marker=marker,
                alpha=alpha,
                c="gray",
            )
            plt.scatter(
                mypos[..., :: self.n_sites, i],
                mypos[..., :: self.n_sites, ii],
                marker=marker,
                alpha=alpha,
                c="r",
            )

            # draw box
            coord = [
                [0, 0],
                [av_box[i, i], av_box[i, ii]],
                [av_box[i, i] + av_box[ii, i], av_box[i, ii] + av_box[ii, ii]],
                [av_box[ii, i], av_box[ii, ii]],
                [0, 0],
            ]
            xs, ys = zip(*coord)
            plt.plot(xs, ys, "k:")
            if not self.is_box_orthorombic:
                coord2 = [
                    coord[1],
                    [
                        coord[1][0] + av_box[iii, i],
                        coord[1][1] + av_box[iii, ii],
                    ],
                    [
                        coord[2][0] + av_box[iii, i],
                        coord[2][1] + av_box[iii, ii],
                    ],
                    [
                        coord[3][0] + av_box[iii, i],
                        coord[3][1] + av_box[iii, ii],
                    ],
                    coord[3],
                ]
                xs, ys = zip(*coord2)
                plt.plot(xs, ys, "k:")
                coord = [coord[2], coord2[2]]
                xs, ys = zip(*coord)
                plt.plot(xs, ys, "k:")

            plt.xlabel(f"x{i} [nm]")
            plt.ylabel(f"x{ii} [nm]")
            plt.gca().set_aspect(1)
        plt.show()

    def get_mdtraj(self, pos=None, box=None):
        if pos is None:
            pos = self._positions
        if box is None:
            box = self._box

        traj = md.Trajectory(pos, self.mdtraj_topology)
        traj.unitcell_vectors = np.resize(box, (len(traj), 3, 3))

        return traj

    def plot_rdf(
        self,
        pos=None,
        box=None,
        r_range=[0, 1],
        selection="name == O",
        **kwargs,
    ):
        traj = self.get_mdtraj(pos, box)
        ij = self.mdtraj_topology.select_pairs(selection, selection)
        rdf = md.compute_rdf(traj, ij, r_range=r_range)

        plt.plot(*rdf, **kwargs)
        plt.ylabel("g(r)")
        plt.xlabel("r [nm]")
        plt.xlim(r_range)

    def get_view(self, pos=None, box=None):
        """visualize in notebook with nglview"""
        if nv is None:
            print("+++ WARNING: nglview not available +++", file=stderr)
            return None

        view = nv.show_mdtraj(self.get_mdtraj(pos, box))
        view.add_representation("ball+stick", selection="water")
        view.add_unitcell()
        return view


# some plotting functions
def plot_energy(ene):
    plt.figure(figsize=(15, 4))

    plt.subplot(1, 2, 1)
    plt.plot(ene, ".")
    plt.xlim(0, len(ene))
    plt.xlabel("time")
    plt.ylabel("energy [kJ/mol]")

    plt.subplot(1, 2, 2)
    plt.hist(ene, bins="auto")
    plt.xlabel("energy [kJ/mol]")

    plt.show()
