import numpy as np

from ase.neighborlist import NeighborList
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import PropertyNotImplementedError


class LennardJones(Calculator):
    """Lennard Jones potential calculator

    see https://en.wikipedia.org/wiki/Lennard-Jones_potential

    Formulas for virial stresses from Fan et al., Physical Review B 92, 094301 (2015).

    """
    implemented_properties = ['energy', 'forces', 'stress', 'energies', 'stresses']
    default_parameters = {'epsilon': 1.0,
                          'sigma': 1.0,
                          'rc': None}
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        sigma: float
          The potential minimum is at  2**(1/6) * sigma, default 1.0
        epsilon: float
          The potential depth, default 1.0
        rc: float, None
          Cut-off for the NeighborList is set to 3 * sigma if None.
          The energy is upshifted to be continuous at rc.
          Default None
        """
        Calculator.__init__(self, **kwargs)

        if self.parameters.rc is None:
            self.parameters.rc = 3 * self.parameters.sigma

        self.nl = None

    def calculate(self, atoms=None,
                  properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        natoms = len(self.atoms)

        sigma = self.parameters.sigma
        epsilon = self.parameters.epsilon
        rc = self.parameters.rc

        if self.nl is None or 'numbers' in system_changes:
            self.nl = NeighborList([rc / 2] * natoms, self_interaction=False)

        self.nl.update(self.atoms)

        positions = self.atoms.positions
        cell = self.atoms.cell

        # potential value at rc. should be small since we later
        # subtract this from the energy inside the cutoff,
        # to ensure that it goes smoothly to zero as we approach rc.
        # so if this value is big, we introduce a large shift into
        # the potential, which might lead to *unexpected* results
        e0 = 4 * epsilon * ((sigma / rc)**12 - (sigma / rc)**6)

        energies = np.zeros(natoms)
        forces = np.zeros((natoms, 3))
        stresses = np.zeros((natoms, 3, 3))

        for i in range(natoms):
            neighbors, offsets = self.nl.get_neighbors(i)
            cells = np.dot(offsets, cell)

            # pointing *towards* neighbours
            d_ij = positions[neighbors] + cells - positions[i]
            r2_ij = (d_ij**2).sum(1)

            c6 = (sigma**2 / r2_ij)**3
            c6[r2_ij > rc**2] = 0.0
            c12 = c6**2
            energies[i] += 4 * epsilon * (c12 - c6).sum()

            # shift potential to ensure smooth cutoff
            energies[i] -= e0 * (c6 != 0.0).sum()

            # forces acting on this atom (i) due to other atoms
            f_ij = (-24 * epsilon * (2 * c12 - c6) / r2_ij)[:, np.newaxis] * d_ij

            forces[i] += f_ij.sum(axis=0)
            stresses[i] += -0.5 * np.dot(f_ij.T, d_ij)  # equivalent to outer product

            # since bothways=False in our NeighbourList,
            # the contributions of this atom (i) to quantities
            # on other atoms (j) need to be added here.
            # (the signs flipped to account for switching j <-> i)
            for index_j, j in enumerate(neighbors):
                forces[j] += -f_ij[index_j]
                stresses[j] += -0.5 * np.outer(-f_ij[index_j], -d_ij[index_j])

        if 'stress' in properties:
            if self.atoms.number_of_lattice_vectors == 3:
                # stress += stress.T.copy() ?
                stress = stresses.sum(axis=0) / self.atoms.get_volume()
                self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
            else:
                raise PropertyNotImplementedError

        if 'stresses' in properties:
            if self.atoms.number_of_lattice_vectors == 3:
                self.results['stresses'] = stresses / self.atoms.get_volume()
            else:
                raise PropertyNotImplementedError

        if 'energies' in properties:
            self.results['energies'] = energies

        energy = energies.sum()
        self.results['energy'] = energy
        self.results['free_energy'] = energy

        self.results['forces'] = forces
