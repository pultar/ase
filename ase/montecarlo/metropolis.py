"""Definition of Metropolis Class for Metropolis Monte Carlo Simulations."""
import sys
from copy import deepcopy
import numpy as np
from ase.utils import basestring
from ase.units import kB
from ase.montecarlo.swap_atoms import SwapAtoms
from ase.atoms import Atoms


class Metropolis(object):
    """Class for performing Metropolis Monte Carlo sampling.

    A calculator object needs to be attached to the Metropolis object in order
    to calculate energies.

    Arguments
    =========
    atoms: Atoms object to specify the initial structure. A calculator need to
           attached to *atoms* in order to calculate energy.

    temp: temperature in Kelvin for Monte Carlo simulation

    constraint: types of constraints imposed on swapping two atoms.
        - None: any two atoms can be swapped
        - 'nn': any atom selected, swapped only with its nearest neighbor
        - 'basis': any atom selected, swapped only with another atom in the
                   same basis
        - 'nn-basis': any atom selected, swapped only with its nearest neighbor
                      in the same basis

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.
    """

    def __init__(self, atoms, temp, constraint=None, logfile=None):
        if not isinstance(atoms, Atoms):
            raise TypeError('Passed argument should be Atoms object')
        self.atoms = atoms
        self.energy = None
        if not isinstance(temp, (int, float)):
            raise TypeError('temp must be in int or float type')
        self.kT = kB * temp

        # constraints
        allowed_constraints = [None, 'nn', 'basis', 'nn-basis']
        if constraint not in allowed_constraints:
            raise TypeError('constraint needs to be one of: '
                            '{}'.format(allowed_constraints))
        self.constraint = constraint

        # logfile
        if isinstance(logfile, basestring):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

        # observers
        # observers will be called every nth step specified by the user
        self.observers = []
        self.nsteps = 0

    def run(self, num_steps=10, average=False):
        """Run Monte Carlo simulation.

        Perform Metropolis Monte Carlo simulation using the number of steps
        specified by a user. Returns energy and energy**2.

        If average is set to True, returns average energy and energy**2.
        If average is set to False, returns the sum of energy and energy**2
        over the total number of steps.

        Arguments
        =========
        num_steps: Number of steps in Monte Carlo simulation.
        average: whether or not to return the average values.
            - True: returns average energy and energy**2 over entire simulation
            - False: returns the sum of energy and energy**2 over the entire
                     simulation
        """
        # starting energy
        self.energy = self.atoms.get_potential_energy()
        energy_sum = deepcopy(self.energy)
        energy_sq_sum = self.energy**2
        self.log()

        for _ in range(num_steps):
            accept, energy = self._step()
            energy_sum += self.energy
            energy_sq_sum += self.energy**2
            self.log(accept, energy)

        if average:
            energy_sum /= num_steps
            energy_sq_sum /= num_steps

        return energy_sum, energy_sq_sum

    def _step(self):
        """Perform one Monte Carlo step."""
        if self.constraint is None:
            swapped_indices = SwapAtoms.swap_any_two_atoms(self.atoms)
        elif self.constraint == 'nn':
            swapped_indices = SwapAtoms.swap_nn_atoms(self.atoms)
        else:
            raise NotImplementedError('This feature is not implemented')

        energy = self.atoms.get_potential_energy()
        accept = np.exp((self.energy - energy) / self.kT) > np.random.uniform()

        if accept:
            self.energy = deepcopy(energy)
        else:
            # Swap atoms back to the original
            SwapAtoms.swap_by_indices(self.atoms, swapped_indices[0],
                                      swapped_indices[1])
            # CE calculator needs to call a *restore* method
            if self.atoms.calc.__class__.__name__ == 'ClusterExpansion':
                self.atoms.calc.restore()

        self.nsteps += 1

        return accept, energy

    def log(self, accept=None, new_energy=None):
        """Write energy and energy^2 to log file"""
        if self.logfile is None:
            return True

        if self.nsteps == 0:
            self.logfile.write('\t\t\t\tselected structure \t\t\t'
                               'candidate structure\n')
            self.logfile.write('steps \taccept \tEnergy \t\t\tEnergy^2'
                               '\t\tEnergy \t\t\tEnergy^2\n')
            self.logfile.write('{}\t{}\t{}\t{}'.format(self.nsteps, "-----",
                                                       self.energy,
                                                       self.energy**2))
            self.logfile.write('\n')
        else:
            self.logfile.write('{}\t{}\t{}\t{}\t{}\t{}'.format(self.nsteps,
                                                               accept,
                                                               self.energy,
                                                               self.energy**2,
                                                               new_energy,
                                                               new_energy**2))
            self.logfile.write('\n')

        self.logfile.flush()

    def attach(self, observer, interval=1):
        """Needs to be implemented
        """
        return True
