"""Class containing a manager for creating template atoms."""
import numpy as np
from itertools import product, permutations
from numpy.linalg import inv
from random import choice


class TemplateAtoms(object):
    def __init__(self, supercell_factor=None, size=None, skew_threshold=4,
                 unit_cells=None, unit_cell_ids=None):
        if size is None and supercell_factor is None:
            raise TypeError("Either size or supercell_factor needs to be "
                            "specified.\n size: list or numpy array.\n "
                            "supercell_factor: int")

        self.supercell_factor = supercell_factor
        self.size = size
        self.skew_threshold = skew_threshold
        self.unit_cells = unit_cells
        self.unit_cell_ids = unit_cell_ids

        if size is None:
            self.supercell_factor = int(supercell_factor)
            templates = self._generate_template_atoms()
            self.templates = self._filter_equivalent_templates(templates)
            if not self.templates['atoms']:
                raise RuntimeError("No template atoms matching criteria!")
        else:
            # if size and supercell_factor are both specified,
            # size will be used
            self.templates = {'atoms': [self.unit_cells[0] * size],
                              'size': [size],
                              'unit_cell_id': [self.unit_cell_ids[0]]}
        # self._check_templates_datastructure()

    def __str__(self):
        """Print a summary of the class."""
        msg = "=== TemplateAtoms ===\n"
        msg += "Supercell factor: {}\n".format(self.supercell_factor)
        msg += "Skewed threshold: {}\n".format(self.skew_threshold)
        msg += "Template sizes:\n"
        for size in self.templates['size']:
            msg += "{}\n".format(size)
        return msg

    def _generate_template_atoms(self):
        """Generate all template atoms up to a certain multiplicity factor."""
        templates = {'atoms': [], 'size': [], 'unit_cell_id': []}
        # case 1: size of the cell is given
        if self.size is not None:
            # if len(self.unit_cells) != 1:
            #     raise ValueError("Exactly one unit cell must be used when "
            #                      "the size of the cell is specified.")
            for i, _ in enumerate(self.unit_cells):
                templates['atoms'].append(self.unit_cells[i] * self.size)
                templates['size'].append(self.size)
                templates['unit_cell_id'].append(self.unit_cell_ids[i])
            return templates

        # case 2: supercell_factor is given
        for size in product(range(1, self.supercell_factor+1), repeat=3):
            # Skip cases where the product of factors is larger than the
            # supercell factor.
            if np.prod(size) > self.supercell_factor:
                continue

            for i, _ in enumerate(self.unit_cells):
                templates['atoms'].append(self.unit_cells[i] * size)
                templates['size'].append(size)
                templates['unit_cell_id'].append(self.unit_cell_ids[i])
        return templates

    @property
    def num_templates(self):
        return len(self.templates['atoms'])

    def get_unit_cell_id(self, uid):
        """Return the unit cell id."""
        return self.templates['unit_cell_id'][uid]

    def get_atoms(self, uid, return_size=False):
        """Return atoms at position."""
        if return_size:
            return self.templates['atoms'][uid], self.templates['size'][uid]

        return self.templates['atoms'][uid]

    def get_uid_with_given_size(self, size, generate_template=False):
        """Get the UID of the template with given size.

        Arguments:
        =========
        size: list of length 3

        generate_template: bool (optional)
            If *True*, generate a new template if a template with matching
            size is not found.
        """
        num_occurrences = self.templates['size'].count(size)
        if num_occurrences > 1:
            raise RuntimeError("There are {} templates of size {}.\n"
                               "There might be more than one unit cells used."
                               "".format(num_occurrences, size))

        elif num_occurrences == 1:
            return self.templates['size'].index(size)

        if not generate_template:
            raise ValueError("There is no template with size {}."
                             "".format(size))

        # get dims based on the passed atoms and append.
        print("Template that matches the specified size not found. "
              "Generating...")
        if len(self.unit_cells) > 1:
            raise RuntimeError("Cannot generate a template because there "
                               "are {} unit cells in TemplateAtoms.\n"
                               "You can still generate a template by passing "
                               "an Atoms object."
                               "".format(len(self.unit_cells)))

        self.templates['atoms'].append(self.unit_cells[0]*size)
        self.templates['size'].append(size)
        self.templates['unit_cell_id'].append(self.unit_cell_ids[0])
        # self._check_templates_datastructure()

        return len(self.templates['atoms']) - 1

    def get_uid_matching_atoms(self, atoms, generate_template=False):
        """Get the UID for the template matching atoms.

        Arguments:
        =========
        atoms: Atoms object
            structure to compare its size against template atoms

        generate_template: bool (optional)
            If *True*, generate a new template if a template with matching
            size is not found.
        """
        shape = atoms.get_cell_lengths_and_angles()
        for uid, template in enumerate(self.templates['atoms']):
            shape_template = template.get_cell_lengths_and_angles()
            if np.allclose(shape, shape_template):
                return uid

        if not generate_template:
            raise ValueError("There is no template that matches the shape "
                             "of given atoms object")

        # get dims based on the passed atoms and append.
        print("Template that matches the size of passed atoms not found. "
              "Generating...")
        index, size = self._get_scale_factor(atoms)
        self.templates['atoms'].append(self.unit_cells[index]*size)
        self.templates['size'].append(size)
        self.templates['unit_cell_id'].append(self.unit_cell_ids[index])
        # self._check_templates_datastructure()

        return len(self.templates['atoms']) - 1

    def _get_scale_factor(self, atoms):
        """Return the index of unit cell and scale factor."""
        lengths = atoms.get_cell_lengths_and_angles()[:3]
        for i, unit_cell in enumerate(self.unit_cells):
            lengths_unit = unit_cell.get_cell_lengths_and_angles()[:3]
            scale = lengths / lengths_unit
            scale_int = scale.round(decimals=0).astype(int)
            if np.allclose(scale, scale_int):
                return i, scale_int

        raise ValueError("The passed atoms object cannot be described by "
                         "repeating any of the unit cells")

    def _is_unitary(self, matrix):
        return np.allclose(matrix.T.dot(matrix), np.identity(matrix.shape[0]))

    def _are_equivalent(self, cell1, cell2):
        """Compare two cells to check if they are equivalent.

        It is assumed that the cell vectors are columns of each matrix.
        """
        inv_cell1 = inv(cell1)
        for perm in permutations(range(3)):
            permute_cell = cell2[:, perm]
            R = permute_cell.dot(inv_cell1)
            if self._is_unitary(R):
                return True
        return False

    def get_size(self):
        """Get size of the templates."""
        return self.templates['size']

    def _filter_equivalent_templates(self, templates):
        """Remove symmetrically equivalent clusters."""
        templates = self._filter_very_skewed_templates(templates)
        filtered = {'atoms': [], 'size': [], 'unit_cell_id': []}
        for i, atoms in enumerate(templates['atoms']):
            current = atoms.get_cell().T
            duplicate = False
            for j in range(0, len(filtered['atoms'])):
                ref = filtered['atoms'][j].get_cell().T
                if self._are_equivalent(current, ref):
                    duplicate = True
                    break

            if not duplicate:
                filtered['atoms'].append(atoms)
                filtered['size'].append(templates['size'][i])
                filtered['unit_cell_id'].append(templates['unit_cell_id'][i])
        return filtered

    def _filter_very_skewed_templates(self, templates):
        """Remove templates that have a very skewed unit cell."""
        filtered = {'atoms': [], 'size': [], 'unit_cell_id': []}
        for i, atoms in enumerate(templates['atoms']):
            ratio = self._get_max_min_diag_ratio(atoms)
            if ratio < self.skew_threshold:
                filtered['atoms'].append(atoms)
                filtered['size'].append(templates['size'][i])
                filtered['unit_cell_id'].append(templates['unit_cell_id'][i])
        return filtered

    def random_template(self, max_supercell_factor=1000, return_size=False):
        """Select a random template atoms.

        Arguments:
        =========
        max_supercell_factor: int
            Maximum supercell factor the returned object can have
        """
        found = False
        num = 0
        while not found:
            num = choice(range(len(self.templates['atoms'])))
            factor = np.prod(self.templates['size'][num])
            if factor <= max_supercell_factor:
                found = True
        return num

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

    def weighted_random_template(self, return_size=False):
        """Select a random template atoms with a bias towards a cubic cell.

        The bias is towards cells that have similar values for x-, y- and
        z-dimension sizes.
        """
        p_select = []
        for atoms in self.templates['atoms']:
            ratio = self._get_max_min_diag_ratio(atoms)
            p = np.exp(-4.0*(ratio-1.0)/self.skew_threshold)
            p_select.append(p)
        p_select = np.array(p_select)
        p_select /= np.sum(p_select)

        cum_prob = np.cumsum(p_select)
        rand_num = np.random.rand()
        indx = np.argmax(cum_prob > rand_num)
        return indx

    def load_from_database(self, db_name):
        from ase.db import connect
        db = connect(db_name)
        templates = {'atoms': [], 'size': [], 'unit_cell_id': []}
        for row in db.select([('name', '=', 'template')]):
            templates['atoms'].append(row.toatoms())
            templates['size'].append(row.dims)
            templates['unit_cell_id'].append(row.unit_cell_type)
        return templates

    def _check_templates_datastructure(self):
        """Fails if the datastructure is inconsistent."""
        num_entries = len(self.templates['atoms'])
        for _, v in self.templates.items():
            assert len(v) == num_entries
