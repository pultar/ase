import numpy as np
from ase.build import bulk
from ase.spacegroup import crystal, Spacegroup
from ase.ce.tools import wrap_and_sort_by_position
from ase.ce.settings import ClusterExpansionSetting


class BulkCrystal(ClusterExpansionSetting):
    """Store settings for Cluster Expansion on bulk materials defined based on
    crystal structures.

    Arguments:
    =========
    basis_elements: list
        List of chemical symbols of elements to occupy each basis.
        Even for the cases where there is only one basis (e.g., fcc, bcc, sc),
        a list of symbols should be grouped by basis as in [['Cu', 'Au']]
        (note the nested list form).
    crystalstructure: str
        Must be one of sc, fcc, bcc, hcp, diamond, zincblende, rocksalt,
        cesiumchloride, fluorite or wurtzite.
    a: float
        Lattice constant.
    c: float
        Lattice constant.
    covera: float
        c/a ratio used for hcp.  Default is ideal ratio: sqrt(8/3).
    u: float
        Internal coordinate for Wurtzite structure.
    orthorhombic: bool
        Construct orthorhombic unit cell instead of primitive cell
        which is the default.
    cubic: bool
        Construct cubic unit cell if possible.
    size: list
        size of the supercell (e.g., [2, 2, 2] for 2x2x2 cell)
    conc_args: dict
        ratios of the elements for different concentrations.
    db_name: str
        name of the database file
    max_cluster_size: int
        maximum size (number of atoms in a cluster)
    max_cluster_dia: float or int
        maximum diameter of cluster (in angstrom)
    grouped_basis: list
        indices of basis_elements that are considered to be equivalent when
        specifying concentration (e.g., useful when two basis are shared by
        the same set of elements and no distinctions are made between them)
    """
    def __init__(self, basis_elements=None, crystalstructure=None,
                 a=None, c=None, covera=None, u=None, orthorhombic=False,
                 cubic=False, size=None, conc_args=None, db_name=None,
                 max_cluster_size=4, max_cluster_dia=None, grouped_basis=None):

        self.basis_elements = basis_elements
        self.structures = {'sc': 1, 'fcc': 1, 'bcc': 1, 'hcp': 1, 'diamond': 1,
                           'zincblende': 2, 'rocksalt': 2, 'cesiumchloride': 2,
                           'fluorite': 3, 'wurtzite': 2}
        self.crystalstructure = crystalstructure
        self.a = a
        self.c = c
        self.covera = covera
        self.u = u
        self.orthorhombic = orthorhombic
        self.cubic = cubic
        self.size = size

        self.num_basis = len(basis_elements)
        if self.num_basis != self.structures[self.crystalstructure]:
            raise ValueError("{} has ".format(self.crystalstructure) +
                             "{} basis."
                             .format(self.structures[self.crystalstructure]) +
                             "The number of basis specified by basis_elements "
                             "is {}".format(self.num_basis))
        self.unit_cell = self._get_unit_cell()
        self.atoms_with_given_dim = self._get_atoms_with_given_dim()
        self.min_lat = self._get_min_lat()

        ClusterExpansionSetting.__init__(self, conc_args, db_name,
                                         max_cluster_size, max_cluster_dia,
                                         basis_elements)

        self.index_by_basis = self._group_index_by_basis()
        if grouped_basis is not None:
            self.num_grouped_basis = len(grouped_basis)
            self.index_by_grouped_basis = self._group_index_by_basis_group()
            self.grouped_basis_elements = self._get_grouped_basis_elements()

    def _get_unit_cell(self):
        if self.num_basis == 1:
            atoms = bulk(name='{}'.format(self.basis_elements[0][0]),
                         crystalstructure=self.crystalstructure, a=self.a,
                         c=self.c, covera=self.covera, u=self.u,
                         orthorhombic=self.orthorhombic, cubic=self.cubic)

        elif self.num_basis == 2:
            atoms = bulk(name='{}{}'.format(self.basis_elements[0][0],
                                            self.basis_elements[1][0]),
                         crystalstructure=self.crystalstructure, a=self.a,
                         c=self.c, covera=self.covera, u=self.u,
                         orthorhombic=self.orthorhombic, cubic=self.cubic)

        else:
            atoms = bulk(name='{}{}{}'.format(self.basis_elements[0][0],
                                              self.basis_elements[1][0],
                                              self.basis_elements[2][0]),
                         crystalstructure=self.crystalstructure, a=self.a,
                         c=self.c, covera=self.covera, u=self.u,
                         orthorhombic=self.orthorhombic, cubic=self.cubic)
        atoms = wrap_and_sort_by_position(atoms)
        return atoms

    def _get_min_lat(self):
        """Get the minimum length of the lattice vectors of the unit cell."""
        atoms = self.unit_cell
        return min(atoms.get_cell_lengths_and_angles()[:3])

    def _group_index_by_basis(self):
        num_basis = self.structures[self.crystalstructure]
        if num_basis == 1:
            indx_by_basis = [[a.index for a in self.atoms]]
            return indx_by_basis

        # This condition can be relaxed in the future
        first_elements = []
        for elements in self.basis_elements:
            first_elements.append(elements[0])
        if len(set(first_elements)) != num_basis:
            raise ValueError("First element of different basis should not be "
                             "the same.")

        indx_by_basis = []
        for basis in self.basis_elements:
            indx_by_basis.append([a.index for a in self.atoms_with_given_dim if
                                  a.symbol == basis[0]])

        for basis in indx_by_basis:
            basis.sort()
        return indx_by_basis


class BulkSpacegroup(ClusterExpansionSetting):
    """"Store settings for Cluster Expansion on bulk materials defined based on
    space group.

    Arguments:
    =========
    basis_elements: list
        List of chemical symbols of elements to occupy each basis.
    basis : list of scaled coordinates
        Positions of the unique sites corresponding to symbols given
        either as scaled positions or through an atoms instance.
    spacegroup : int | string | Spacegroup instance
        Space group given either as its number in International Tables
        or as its Hermann-Mauguin symbol.
    cell : 3x3 matrix
        Unit cell vectors.
    cellpar : [a, b, c, alpha, beta, gamma]
        Cell parameters with angles in degree. Is not used when `cell`
        is given.
    ab_normal : vector
        Is used to define the orientation of the unit cell relative
        to the Cartesian system when `cell` is not given. It is the
        normal vector of the plane spanned by a and b.
    primitive_cell : bool
        Wheter to return the primitive instead of the conventional
        unit cell.
    size: list of 3 positive integers
        How many times the conventional unit cell should be repeated
        in each direction.
    conc_args: dict
        ratios of the elements for different concentrations.
    db_name: str
        name of the database file
    max_cluster_size: int
        maximum size (number of atoms in a cluster)
    max_cluster_dia: float or int
        maximum diameter of cluster (in angstrom)
    grouped_basis: list
        indices of basis_elements that are considered to be equivalent when
        specifying concentration (e.g., useful when two basis are shared by
        the same set of elements and no distinctions are made between them)
    """
    def __init__(self, basis_elements=None, basis=None, spacegroup=1,
                 cell=None, cellpar=None, ab_normal=(0, 0, 1), size=None,
                 primitive_cell=False, conc_args=None, db_name=None,
                 max_cluster_size=4, max_cluster_dia=None, grouped_basis=None):
        # Set parameters for spacegroup crystal
        self.basis = basis
        self.num_basis = len(basis)
        self.spacegroup = spacegroup
        self.cell = cell
        self.cellpar = cellpar
        self.ab_normal = ab_normal
        self.size = np.array(size)
        self.primitive_cell = primitive_cell
        self.symbols = []
        for x in range(self.num_basis):
            self.symbols.append(basis_elements[x][0])
        self.unit_cell = self._get_unit_cell()
        self.min_lat = self._get_min_lat()
        self.atoms_with_given_dim = self._get_atoms_with_given_dim()

        ClusterExpansionSetting.__init__(self, conc_args, db_name,
                                         max_cluster_size, max_cluster_dia,
                                         basis_elements, grouped_basis)

        self.index_by_basis = self._group_index_by_basis()
        if grouped_basis is not None:
            self.num_grouped_basis = len(grouped_basis)
            self.index_by_grouped_basis = self._group_index_by_basis_group()
            self.grouped_basis_elements = self._get_grouped_basis_elements()

    def _get_min_lat(self):
        # use cellpar only when cell is not defined
        if self.cell is None:
            lat = float(min(self.cellpar[:3]))
        else:
            lat = float(min(np.sum(self.cell, axis=1)))
        return lat

    def _get_unit_cell(self):
        atoms = crystal(symbols=self.symbols, basis=self.basis,
                        spacegroup=self.spacegroup, cell=self.cell,
                        cellpar=self.cellpar, ab_normal=self.ab_normal,
                        size=[1, 1, 1], primitive_cell=self.primitive_cell)
        atoms = wrap_and_sort_by_position(atoms)
        return atoms

    def _group_index_by_basis(self):
        indx_by_basis = [[] for _ in range(self.num_basis)]
        sg = Spacegroup(self.spacegroup)
        sites, kinds = sg.equivalent_sites(self.basis)

        scale_factor = np.multiply(self.supercell_scale_factor, self.size)

        # account for the case where a supercell is needed
        if not np.array_equal(scale_factor, np.array([1, 1, 1])):
            sites = np.divide(sites, scale_factor)
            # x dimension
            kinds = np.tile(kinds, scale_factor[0])
            sites_temp = np.copy(sites)
            for x in range(1, scale_factor[0]):
                shift = np.add(sites_temp, [float(x) / scale_factor[0], 0, 0])
                sites = np.append(sites, shift, axis=0)
            # y dimension
            kinds = np.tile(kinds, scale_factor[1])
            sites_temp = np.copy(sites)
            for y in range(1, scale_factor[1]):
                shift = np.add(sites_temp, [0, float(y) / scale_factor[1], 0])
                sites = np.append(sites, shift, axis=0)
            # z dimension
            kinds = np.tile(kinds, scale_factor[2])
            sites_temp = np.copy(sites)
            for z in range(1, scale_factor[2]):
                shift = np.add(sites_temp, [0, 0, float(z) / scale_factor[2]])
                sites = np.append(sites, shift, axis=0)

        positions = self.atoms.get_scaled_positions()
        for i, site in enumerate(sites):
            for j, pos in enumerate(positions):
                indx = None
                if np.allclose(site, pos, atol=1.e-5):
                    indx = j
                    break
            indx_by_basis[kinds[i]].append(indx)

        for basis in indx_by_basis:
            basis.sort()

        return indx_by_basis
