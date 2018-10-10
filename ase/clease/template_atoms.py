"""Class containing a manager for creating template atoms."""
import numpy as np
from itertools import product
from numpy.linalg import inv, det

class TemplateAtoms(object):
    def __init__(self, supercell_factor=None, size=None, unit_cells=None):
        if size is None and supercell_factor is None:
            raise TypeError("Either size or supercell_factor needs to be "
                            "specified.\n size: list or numpy array.\n "
                            "supercell_factor: int")

        self.size = size
        if size is None:
            self.supercell_factor = int(supercell_factor)
        self.unit_cells = unit_cells
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
        """Remove the symmetrically equivalent clusters."""
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
