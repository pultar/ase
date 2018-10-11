"""Class containing a manager for creating template atoms."""
import numpy as np
from itertools import product
from numpy.linalg import inv, det
from random import choice

class TemplateAtoms(object):
    def __init__(self, supercell_factor=None, size=None, unit_cells=None,
                 skew_threshold=4):
        if size is None and supercell_factor is None:
            raise TypeError("Either size or supercell_factor needs to be "
                            "specified.\n size: list or numpy array.\n "
                            "supercell_factor: int")

        self.size = size
        if size is None:
            self.supercell_factor = int(supercell_factor)
        self.unit_cells = unit_cells
        self.skew_threshold = skew_threshold
        templates = self._generate_template_atoms()
        self.templates = self._filter_equivalent_templates(templates)

    def _generate_template_atoms(self):
        """Generate all template atoms up to a certain multiplicity factor."""
        templates = {"atoms": [], "dim": []}
        # case 1: size of the cell is given
        if self.size is not None:
            if len(self.unit_cells) != 1:
                raise ValueError("Either one of primitive or conventional "
                                 "cell must be used when the size of the cell "
                                 "is specified.")
            for atom in self.unit_cells:
                templates["atoms"].append(atom * self.size)
                templates["dim"].append(self.size)
            return templates

        for size in product(range(1, self.supercell_factor+1), repeat=3):
            # If the product of all factors is larger than the
            # supercell factor, we skip this combination
            if np.prod(size) > self.supercell_factor:
                continue

            for atoms in self.unit_cell:
                templates["atoms"].append(atoms * size)
                templates["dim"].append(size)
        return templates

    def _are_equivalent(self, cell1, cell2):
        """Compare two cells to check if they are equivalent.

        It is assumed that the cell vectors are columns of each matrix.
        """
        R = inv(cell1).dot(cell2)
        determinant = det(R)
        return abs(abs(determinant) - 1.0) < 1E-4

    def _filter_equivalent_templates(self, templates):
        """Remove symmetrically equivalent clusters."""
        templates = self._filter_very_skewed_templates(templates)
        filtered = {"atoms": [], "dim": []}
        for i, atom in enumerate(templates["atoms"]):
            current = atom.get_cell().T
            duplicate = False
            for j in range(0, len(filtered["atoms"])):
                ref = filtered[j].get_cell().T
                if self._are_equivalent(current, ref):
                    duplicate = True
                    break

            if not duplicate:
                filtered["atoms"].append(atom)
                filtered["dim"].append(templates["dim"][i])
        return filtered

    def _filter_very_skewed_templates(self, templates):
        """Remove templates that have a very skewed unit cell
        """
        filtered = {"atoms": [], "dim": []}
        for i, atoms in enumerate(templates["atoms"]):
            ratio = self._get_max_min_diag_ratio(atoms)
            if ratio < self.skew_threshold:
                filtered["atoms"].append(atoms)
                filtered["dim"].append(templates["dim"][i])
        return filtered

    def random_template(self, max_supercell_factor=1000, return_dim=False):
        """Select a random template atoms.

        Arguments:
        =========
            max_supercell_factor: int
                Maximum supercell factor the returned object can have
        """
        found = False
        num = 0
        while not found:
            num = choice(range(len(self.templates["atoms"])))
            factor = np.prod(self.templates["dim"][num])
            if factor <= max_supercell_factor:
                found = True
        if return_dim:
            return self.templates["atoms"][num], self.templates["dim"][num]
        return self.templates["atoms"][num]

    def _get_max_min_diag_ratio(self, atoms):
        """Return the ratio between the maximum and the minimum diagonal."""
        diag_lengths = []
        cell = atoms.get_cell().T
        for w in product([-1, 0, 1], repeat=3):
            if np.allclose(w, 0):
                continue
            diag = cell.dot(w)
            length = np.sqrt(diag.dot(diag))
            diag_lengths.append(length)
        max_length = np.max(diag_lengths)
        min_length = np.min(diag_lengths)
        return max_length/min_length

    def weighted_random_template(self, return_dim=False):
        """Select a random template atoms with a bias towards cells that are
        small and has similar values for x-, y- and z-dimension sizes.
        """
        p_select = []
        for dim in self.templates["dim"]:
            mean = np.mean(dim)
            prod = np.prod(dim)
            p = np.exp(-np.sum(((np.array(dim)-mean)/mean)**2))
            p_select.append(p/prod)
        p_select = np.array(p_select)
        p_select /= np.sum(p_select)

        cum_prob = np.cumsum(p_select)
        rand_num = np.random.rand()
        indx = np.argmax(cum_prob > rand_num)
        if return_dim:
            return self.templates["atoms"][num], self.templates["dim"][num]
        return self.templates["atoms"][num]
