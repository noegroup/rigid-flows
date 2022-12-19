#!/usr/bin/env python

# some useful stuff

import json
import numpy as np
from sys import stderr

import openmm
import openmm.app
from openmm import unit
from openmmtools.testsystems import LennardJonesFluid

import matplotlib.pyplot as plt
import mdtraj as md
try:
    import nglview as nv  # type: ignore
except:
    print("+++ WARNING: nglview not available +++", file=stderr)
    nv = None


class LennardJonesModel:
    def __init__(
        self,
        nparticles,
        reduced_density,
        reduced_cutoff=2.5,
        lattice=True,
        barostat=None,
        external_field=None,
    ):
        #typical values for Argon, also default in https://openmmtools.readthedocs.io/en/stable/api/generated/openmmtools.testsystems.LennardJonesFluid.html
        self.sigma = 0.34
        self.mass = 39.9
        self.epsilon = (0.238 * unit.kilocalorie_per_mole).value_in_unit(unit.kilojoule_per_mole)

        reduced_box_edge = np.cbrt(nparticles / reduced_density)
        if reduced_box_edge < 2 * reduced_cutoff:
            raise ValueError('cutoff should not be smaller than half box size')

        model = LennardJonesFluid(
            nparticles=nparticles,
            reduced_density=reduced_density,
            mass=self.mass * unit.amu,
            sigma=self.sigma * unit.nanometer,
            epsilon=self.epsilon * unit.kilojoule_per_mole,
            cutoff=reduced_cutoff * self.sigma * unit.nanometer,
            lattice=lattice,
            shift=True,
        )
        model.system.addForce(openmm.CMMotionRemover())
        self.system = model.system
        self.topology = model.topology
        self.mdtraj_topology = model.mdtraj_topology

        self.set_barostat(barostat)
        self.set_external_field(external_field)

        model.positions += (model.system.getDefaultPeriodicBoxVectors()[0][0] - max(max(model.positions)))/2
        self._positions = np.array(model.positions.value_in_unit(unit.nanometer))

        self._box = np.array([b.value_in_unit(unit.nanometer) for b in model.system.getDefaultPeriodicBoxVectors()])
        self.is_box_orthorombic = not np.count_nonzero(
            self.box - np.diag(np.diagonal(self.box))
        )

        self.nparticles = nparticles
        self.reduced_density = reduced_density
        self.reduced_cutoff = reduced_cutoff
        self.lattice = lattice

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

    def save_to_json(self, filename):
        init_info = {
            "nparticles": self.nparticles,
            "reduced_density": self.reduced_density,
            "reduced_cutoff": self.reduced_cutoff,
            "lattice": self.lattice,
            "barostat": self.barostat,
            "external_field": self.external_field,
        }
        with open(filename, "w") as f:
            json.dump(init_info, f)

    @staticmethod
    def load_from_json(filename):
        with open(filename, "r") as f:
            init_info = json.load(f)
        return LennardJonesModel(**init_info)

    def setup_simulation(
        self,
        reduced_temperature,
        frictionCoeff=1 / unit.picosecond,
        stepSize=1 * unit.femtosecond,
        minimizeEnergy=False,
    ):
        kB = unit.MOLAR_GAS_CONSTANT_R.value_in_unit(unit.kilojoule_per_mole/unit.kelvin)
        temperature = (reduced_temperature * self.epsilon / kB)
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
        simulation = openmm.app.Simulation(self.topology, self.system, integrator)
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

    def plot_2Dview(self, pos=None, box=None, toPBC=False):
        if pos is None:
            pos = self._positions
        if box is None:
            box = self._box

        if len(pos.squeeze().shape) == 2:
            marker = 'o'
            alpha = 1
        elif len(pos.squeeze().shape) == 3:
            marker = '.'
            alpha = 0.05
        else:
            raise ValueError('pos should be of shape (nframes, natoms, 3)')
        if pos.shape[-1] != 3:
            raise ValueError('pos should be in 3D')

        av_box = box.mean(axis=0) if len(box.shape) == 3 else box
        if av_box.shape != (3, 3):
            raise ValueError('box should be a 3x3 matrix')

        if toPBC:
            if self.is_box_orthorombic:
                mypos = (pos / np.diagonal(av_box) % 1) * np.diagonal(av_box)
            else:
                raise NotImplementedError('only available for fixed orthorombic box')
        else:
            mypos = pos

        plt.figure(figsize=(15, 4))
        for i in range(3):
            ii = (i + 1) % 3
            iii = (i + 2) % 3
            plt.subplot(1, 3, 1+i)

            #draw particles
            plt.scatter(mypos[..., :, i], mypos[..., :, ii], marker=marker, alpha=alpha, c='gray')

            #draw box
            coord = [
                [0, 0],
                [av_box[i,i], av_box[i,ii]],
                [av_box[i,i] + av_box[ii,i], av_box[i,ii] + av_box[ii,ii]],
                [av_box[ii,i], av_box[ii,ii]],
                [0, 0]
            ]
            xs, ys = zip(*coord)
            plt.plot(xs, ys, 'k:')
            if not self.is_box_orthorombic:
                coord2 = [
                    coord[1],
                    [coord[1][0] + av_box[iii,i], coord[1][1] + av_box[iii,ii]],
                    [coord[2][0] + av_box[iii,i], coord[2][1] + av_box[iii,ii]],
                    [coord[3][0] + av_box[iii,i], coord[3][1] + av_box[iii,ii]],
                    coord[3],
                ]
                xs, ys = zip(*coord2)
                plt.plot(xs, ys, 'k:')
                coord = [coord[2], coord2[2]]
                xs, ys = zip(*coord)
                plt.plot(xs, ys, 'k:')

            plt.xlabel(f'x{i} [nm]')
            plt.ylabel(f'x{ii} [nm]')
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

    def plot_rdf(self, pos=None, box=None, r_range=[0,1], **kwargs):
        traj = self.get_mdtraj(pos, box)

        rdf = md.compute_rdf(traj, self.mdtraj_topology.select_pairs('True', 'True'), r_range)
        plt.plot(*rdf, **kwargs)
        plt.xlim(r_range)
        plt.xlabel('r [nm]')
        plt.ylabel('g(r)')

    def get_view(self, pos=None, box=None):
        """visualize in notebook with nglview"""
        if nv is None:
            print("+++ WARNING: nglview not available +++", file=stderr)
            return None

        traj = traj = self.get_mdtraj(pos, box)
        view = nv.show_mdtraj(traj)
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

