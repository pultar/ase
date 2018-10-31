"""Definition of ClusterExpansionSetting Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
import os
from itertools import combinations, product
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase.db import connect

from ase.clease.floating_point_classification import FloatingPointClassifier
from ase.clease.tools import (wrap_and_sort_by_position, index_by_position,
                              flatten, sort_by_internal_distances,
                              create_cluster, dec_string, get_unique_name,
                              nested_array2list)
from ase.clease.basis_function import BasisFunction
from ase.clease.template_atoms import TemplateAtoms
from ase.clease.concentration import Concentration


class ClusterExpansionSetting(object):
    """Base class for all Cluster Expansion settings."""

    def __init__(self, size=None, supercell_factor=None, dist_num_dec=3,
                 concentration=None, db_name=None, max_cluster_size=4,
                 max_cluster_dia=None, basis_function='sanchez',
                 ignore_background_atoms=False):
        self.kwargs = {'size': size,
                       'supercell_factor': supercell_factor,
                       'db_name': db_name,
                       'max_cluster_size': max_cluster_size,
                       'max_cluster_dia': deepcopy(max_cluster_dia),
                       'dist_num_dec': dist_num_dec,
                       'ignore_background_atoms': ignore_background_atoms}

        if isinstance(concentration, Concentration):
            self.concentration = concentration
        elif isinstance(concentration, dict):
            self.concentration = Concentration.from_dict(concentration)
        else:
            raise TypeError("concentration has to be either dict or "
                            "instance of Concentration")
            
        self.kwargs["concentration"] = self.concentration.to_dict()
        self.basis_elements = deepcopy(concentration.basis_elements)
        self.num_basis = len(self.basis_elements)
        self.size = size
        self.unit_cell_type = 0
        self.unit_cell = self._get_unit_cell()
        self._tag_unit_cell()
        self.template_atoms = TemplateAtoms(supercell_factor=supercell_factor,
                                            size=size, skew_threshold=4,
                                            unit_cells=[self.unit_cell])
        

        self.dist_num_dec = dist_num_dec
        self.db_name = db_name
        self.max_cluster_size = max_cluster_size
        self.max_cluster_dia = max_cluster_dia
        self.all_elements = sorted([item for row in self.basis_elements for
                                    item in row])
        self.ignore_background_atoms = ignore_background_atoms
        self.background_indices = self._get_background_indices()
        self.num_elements = len(self.all_elements)
        self.unique_elements = sorted(list(set(deepcopy(self.all_elements))))
        self.num_unique_elements = len(self.unique_elements)
        self.index_by_basis = None
        # self.index_by_grouped_basis = None
        # if len(self.basis_elements) != self.num_basis:
        #     raise ValueError("list of elements is needed for each basis")
        # if grouped_basis is None:
        #     self._check_basis_elements()
        # else:
        #     if not isinstance(grouped_basis, list):
        #         raise TypeError('grouped_basis should be a list')
        #     self.num_groups = len(self.grouped_basis)
        #     self._check_grouped_basis_elements()

        if isinstance(basis_function, BasisFunction):
            if basis_function.unique_elements != self.unique_elements:
                raise ValueError("Unique elements in BasiFunction instance "
                                 "is different from the one in settings")
            self.bf_scheme = basis_function
        elif isinstance(basis_function, str):
            if basis_function.lower() == 'sanchez':
                from ase.clease.basis_function import Sanchez
                self.bf_scheme = Sanchez(self.unique_elements)
            elif basis_function.lower() == 'vandewalle':
                from ase.clease.basis_function import VandeWalle
                self.bf_scheme = VandeWalle(self.unique_elements)
            elif basis_function.lower() == "sluiter":
                from ase.clease.basis_function import Sluiter
                self.bf_scheme = Sluiter(self.unique_elements)
            else:
                msg = "basis function scheme {} ".format(basis_function)
                msg += "is not supported."
                raise ValueError(msg)
        else:
            raise ValueError("basis_function has to be instance of "
                             "BasisFunction or a string")

        self.spin_dict = self.bf_scheme.spin_dict
        self.basis_functions = self.bf_scheme.basis_functions
        self.cluster_info = []
        self.index_by_trans_symm = []
        self.ref_index_trans_symm = []
        self.kd_trees = None
        self.set_template_atoms(0)

        if len(self.basis_elements) != self.num_basis:
            raise ValueError("list of elements is needed for each basis")

        # self.atoms_with_given_dim = self._get_atoms_with_given_dim()
        # self._check_conc_ratios(conc_args)
        # self.cluster_info = []
        #
        # self.max_cluster_dia, self.supercell_scale_factor = \
        #     self._get_max_cluster_dia_and_scale_factor(self.max_cluster_dia)
        #
        # self.atoms = self._create_template_atoms()
        #
        # self.index_by_trans_symm = self._group_indices_by_trans_symmetry()
        # self.num_trans_symm = len(self.index_by_trans_symm)
        # self.ref_index_trans_symm = [i[0] for i in self.index_by_trans_symm]
        # self.kd_trees = self._create_kdtrees()

        if not os.path.exists(db_name):
            self._store_data()
        else:
            self._read_data()

    def _group_index_by_basis(self):
        raise NotImplementedError("Has to be implemented in derived classes!")

    def _size2string(self):
        """Converts the current size into a string."""
        return "x".join((str(item) for item in self.size))

    def set_template_atoms(self, uid):
        """Sets a fixed template atoms object as the active."""
        self.atoms_with_given_dim, self.size = \
            self.template_atoms.get_atoms(uid, return_dims=True)
        
        self.index_by_basis = self._group_index_by_basis()
        self.cluster_info = []

        self.max_cluster_dia, self.supercell_scale_factor = \
            self._get_max_cluster_dia_and_scale_factor(self.max_cluster_dia)

        self.atoms = self._create_template_atoms()
        self.background_indices = self._get_background_indices()

        self.index_by_trans_symm = self._group_indices_by_trans_symmetry()
        self.num_trans_symm = len(self.index_by_trans_symm)
        self.ref_index_trans_symm = [i[0] for i in self.index_by_trans_symm]
        self.kd_trees = self._create_kdtrees()

        # Read information from database
        # Note that if the data is not found, it will generate
        # the nessecary data structures and store them in the database
        self._read_data()

    def set_new_template(self):
        """Set a new template atoms object."""
        uid = self.template_atoms.weighted_random_template()
        self.set_template_atoms(uid)

    def _check_conc_ratios(self, conc_args):
        # check for concentration ratios
        conc_names = ['conc_ratio_min_1', 'conc_ratio_min_2',
                      'conc_ratio_max_1', 'conc_ratio_max_2']
        conc_ratio_min_1 = None
        conc_ratio_min_2 = None
        conc_ratio_max_1 = None
        conc_ratio_max_2 = None
        for ratio_type, ratio in conc_args.items():
            if ratio_type not in conc_names:
                raise NameError('The name {} is not supported. \n'
                                'The allowed names are {}'
                                .format(ratio_type, conc_names))
            if ratio_type == conc_names[0]:
                conc_ratio_min_1 = ratio
            elif ratio_type == conc_names[1]:
                conc_ratio_min_2 = ratio
            elif ratio_type == conc_names[2]:
                conc_ratio_max_1 = ratio
            else:
                conc_ratio_max_2 = ratio

        if (conc_ratio_min_1 is None or conc_ratio_max_1 is None):
            raise ValueError('Both min and max concentration ratios need be'
                             ' specified')
        if (conc_ratio_min_2 is not None and conc_ratio_max_2 is not None):
            num_conc_var = 2
        elif (conc_ratio_min_1 is None or conc_ratio_max_1 is None):
            raise ValueError('Both min and max concentration ratios need be'
                             ' specified')
        else:
            num_conc_var = 1
        # Assign parameters
        self.num_conc_var = num_conc_var
        self.conc_ratio_min_1 = conc_ratio_min_1
        self.conc_ratio_max_1 = conc_ratio_max_1
        if num_conc_var == 2:
            self.conc_ratio_min_2 = conc_ratio_min_2
            self.conc_ratio_max_2 = conc_ratio_max_2

    def _tag_unit_cell(self):
        """Add a tag to all the atoms in the unit cell to track the index."""
        for atom in self.unit_cell:
            atom.tag = atom.index

    def _get_unit_cell(self):
        raise NotImplementedError("This function has to be implemented in "
                                  "in derived classes.")

    def _get_max_cluster_dia_and_scale_factor(self, max_cluster_dia):
        cell = self.atoms_with_given_dim.get_cell().T
        min_length = self._get_max_cluster_dia(cell)

        # ------------------------------------- #
        # Get max_cluster_dia in an array form #
        # ------------------------------------- #
        # max_cluster_dia is list or array
        if isinstance(max_cluster_dia, (list, np.ndarray)):
            # Length should be either max_cluster_size+1 or max_cluster_size-1
            if len(max_cluster_dia) == self.max_cluster_size + 1:
                for i in range(2):
                    max_cluster_dia[i] = 0.
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
            elif len(max_cluster_dia) == self.max_cluster_size - 1:
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
                max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])
            else:
                raise ValueError("Invalid length for max_cluster_dia.")
        # max_cluster_dia is int or float
        elif isinstance(max_cluster_dia, (int, float)):
            max_cluster_dia *= np.ones(self.max_cluster_size - 1, dtype=float)
            max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])
        # Case for *None* or something else
        else:
            max_cluster_dia = np.ones(self.max_cluster_size - 1, dtype=float)
            max_cluster_dia *= min_length
            max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])

        # --------------------------------- #
        # Get scale_factor in an array form #
        # --------------------------------- #
        atoms = self.atoms_with_given_dim

        # TODO: Do we need to do something here?
        # It is not the cell vectors that needs to be twice as long as the
        # maximum cluster distance, but the smallest length of the vector
        # formed by any linear combination of the cell vectors
        lengths = atoms.get_cell_lengths_and_angles()[:3] / 2.0
        scale_factor = max(max_cluster_dia) / lengths
        scale_factor = np.ceil(scale_factor).astype(int)
        scale_factor = self._get_scale_factor(cell, max(max_cluster_dia))
        return np.around(max_cluster_dia, self.dist_num_dec), scale_factor

    def _get_max_cluster_dia(self, cell, ret_weights=False):
        lengths = []
        weights = []
        for w in product([-1, 0, 1], repeat=3):
            vec = cell.dot(w)
            if w == (0, 0, 0):
                continue
            lengths.append(np.sqrt(vec.dot(vec)))
            weights.append(w)

        # Introduce tolerance to make max distance strictly
        # smaller than half of the shortest cell dimension
        tol = 2 * 10**(-self.dist_num_dec)
        min_length = min(lengths) / 2
        min_length = np.round(min_length, self.dist_num_dec) - tol

        if ret_weights:
            min_indx = np.argmin(lengths)
            return min_length, weights[min_indx]
        return min_length

    def _get_scale_factor(self, cell, max_cluster_dia):
        """Compute the scale factor nessecary to resolve max_cluster_dia."""
        cell_to_small = True
        scale_factor = [1, 1, 1]
        orig_cell = cell.copy()
        while cell_to_small:

            # Check what the maximum cluster distance is for the current
            # cell
            max_size, w = self._get_max_cluster_dia(cell, ret_weights=True)
            if max_size < max_cluster_dia:
                # Find which vectors formed the shortest diagonal
                indices_in_w = [i for i, weight in enumerate(w) if weight != 0]
                shortest_vec = -1
                shortest_length = 1E10

                # Find the shortest vector of the ones that formed the
                # shortest diagonal
                for indx in indices_in_w:
                    vec = cell[:, indx]
                    length = np.sqrt(vec.dot(vec))
                    if length < shortest_length:
                        shortest_vec = indx
                        shortest_length = length

                # Increase the scale factor in the direction of the shortest
                # vector in the diagonal by 1
                scale_factor[shortest_vec] += 1

                # Update the cell to the new scale factor
                for i in range(3):
                    cell[:, i] = orig_cell[:, i] * scale_factor[i]
            else:
                cell_to_small = False
        return scale_factor

    def _get_background_indices(self):
        """Get symbol of the background atoms.

        This method also modifies grouped_basis, conc_args, num_basis, basis,
        and all_elements attributes to reflect the changes from ignoring the
        background atoms.
        """
        # check if any basis consists of only one element type
        basis = [i for i, b in enumerate(self.basis_elements) if len(b) == 1]

        bkg_indices = []
        for b_indx in basis:
            bkg_indices += self.index_by_basis[b_indx]
        return bkg_indices
        
        # if not basis:
        #     return []

        # symbol = [b[0] for b in self.basis_elements if len(b) == 1]
        # self.num_basis -= len(basis)

        # # change grouped_basis and conc_ratio_min/max if basis are grouped
        # if self.concentration.grouped_basis is not None:
        #     self._modify_group_basis_and_conc(basis)

        # # change basis_elements
        # for i in sorted(basis, reverse=True):
        #     if hasattr(self, 'basis'):
        #         del self.basis[i]
        #     del self.basis_elements[i]
        #     if self.grouped_basis is None:
        #         del self.conc_ratio_min_1[i]
        #         del self.conc_ratio_max_1[i]
        #         if self.num_conc_var == 2:
        #             del self.conc_ratio_min_2[i]
        #             del self.conc_ratio_max_2[i]

        # # change all_elements
        # for s in symbol:
        #     self.all_elements.remove(s)

        # if self.grouped_basis is None:
        #     return list(set(symbol))

        # # reassign grouped_basis if they are grouped
        # for ref in sorted(basis, reverse=True):
        #     for i, group in enumerate(self.grouped_basis):
        #         for j, element in enumerate(group):
        #             if element > ref:
        #                 self.grouped_basis[i][j] -= 1
        # return list(set(symbol))

    # def _modify_group_basis_and_conc(self, basis):
    #     """Remove indices of background atoms in group_basis and conc_args."""
    #     remapped = []
    #     for i, group in enumerate(self.grouped_basis):
    #         if group[0] in basis:
    #             remapped.append(i)
    #     for i in sorted(remapped, reverse=True):
    #         del self.grouped_basis[i]
    #         del self.conc_ratio_min_1[i]
    #         del self.conc_ratio_max_1[i]
    #         if self.num_conc_var == 2:
    #             del self.conc_ratio_min_2[i]
    #             del self.conc_ratio_max_2[i]

    # def _check_basis_elements(self):
    #     error = False
    #     # check dimensions of the element list and concentration ratio lists
    #     if self.num_basis == 1:
    #         if not(self.num_basis == len(self.conc_ratio_min_1)
    #                == len(self.conc_ratio_max_1)):
    #             error = True
    #
    #         if self.num_conc_var == 2:
    #             if not(self.num_basis == len(self.conc_ratio_min_2)
    #                    == len(self.conc_ratio_max_2)):
    #                 error = True
    #     else:
    #         element_size = [len(row) for row in self.basis_elements]
    #         min_1_size = [len(row) for row in self.conc_ratio_min_1]
    #         max_1_size = [len(row) for row in self.conc_ratio_max_1]
    #         if not element_size == min_1_size == max_1_size:
    #             error = True
    #         if self.num_conc_var == 2:
    #             min_2_size = [len(row) for row in self.conc_ratio_min_2]
    #             max_2_size = [len(row) for row in self.conc_ratio_max_2]
    #             if not element_size == min_2_size == max_2_size:
    #                 error = True
    #     if error:
    #         raise ValueError('lengths of the basis_elements and conc_ratio'
    #                          ' lists are not the same')

    def _get_atoms_with_given_dim(self):
        """Create atoms with a user-specified size."""
        atoms = self.unit_cell.copy() * self.size
        return wrap_and_sort_by_position(atoms)

    def _create_template_atoms(self):
        """Return atoms that can handle the specified maximum diameter.

        If maximum diameter is not specified, the user-specified cell
        size will be used.
        """
        atoms = self.atoms_with_given_dim * self.supercell_scale_factor
        return wrap_and_sort_by_position(atoms)

    def _create_kdtrees(self):
        kd_trees = []
        trans = []

        cell = self.atoms.get_cell().T
        weights = [-1, 0, 1]
        for comb in product(weights, repeat=3):
            vec = cell.dot(comb) / 2.0
            trans.append(vec)

        # NOTE: If the error message
        # 'The correlation function changed after simulated annealing'
        # appears when probestructures are generated, uncommenting
        # the next line might be a quick fix. However, this introduce
        # a lot of overhead. For big systems one might easily run out of
        # memory.
        # trans += [atom.position for atom in self.atoms]

        for t in trans:
            shifted = self.atoms.copy()
            shifted.translate(t)
            shifted.wrap()
            kd_trees.append(KDTree(shifted.get_positions()))
        return kd_trees

    def _group_indices_by_trans_symmetry(self):
        """Group indices by translational symmetry."""
        indices = [a.index for a in self.unit_cell]
        ref_indx = indices[0]
        # Group all the indices together if its atomic number and position
        # sequences are same
        indx_by_equiv = []
        bkg_indx_unit_cell = []

        if self.ignore_background_atoms:
            bkg_indx_unit_cell = [a.index for a in self.unit_cell
                                  if a.symbol in self.background_symbol]

        for i, indx in enumerate(indices):
            if indx not in bkg_indx_unit_cell:
                break

        vec = self.unit_cell.get_distance(indices[i], ref_indx, vector=True)
        shifted = self.unit_cell.copy()
        shifted.translate(vec)
        shifted = wrap_and_sort_by_position(shifted)
        an = shifted.get_atomic_numbers()
        pos = shifted.get_positions()

        temp = [[indices[i]]]
        equiv_group_an = [an]
        equiv_group_pos = [pos]
        for indx in indices[i + 1:]:
            if indx in bkg_indx_unit_cell:
                continue
            vec = self.unit_cell.get_distance(indx, ref_indx, vector=True)
            shifted = self.unit_cell.copy()
            shifted.translate(vec)
            shifted = wrap_and_sort_by_position(shifted)
            an = shifted.get_atomic_numbers()
            pos = shifted.get_positions()

            for equiv_group in range(len(temp)):
                if (an == equiv_group_an[equiv_group]).all() and \
                        np.allclose(pos, equiv_group_pos[equiv_group]):
                    temp[equiv_group].append(indx)
                    break
                else:
                    if equiv_group == len(temp) - 1:
                        temp.append([indx])
                        equiv_group_an.append(an)
                        equiv_group_pos.append(pos)

        for equiv_group in temp:
            indx_by_equiv.append(equiv_group)

        # Now we have found the translational symmetry group of all the atoms
        # in the unit cell, now put all the indices of self.atoms into the
        # matrix based on the tag
        indx_by_equiv_all_atoms = [[] for _ in range(len(indx_by_equiv))]
        symm_group_of_tag = [-1 for _ in range(len(self.unit_cell))]
        for gr_id, group in enumerate(indx_by_equiv):
            for item in group:
                symm_group_of_tag[item] = gr_id

        for atom in self.atoms:
            if atom.index in self.background_indices:
                continue
            symm_gr = symm_group_of_tag[atom.tag]
            indx_by_equiv_all_atoms[symm_gr].append(atom.index)
        return indx_by_equiv_all_atoms

    def _assign_correct_family_identifier(self):
        """Make the familily IDs increase size."""
        new_names = {}
        for i, name in enumerate(self.cluster_family_names_by_size):
            if name == "c0" or name == "c1":
                new_names[name] = name
            else:
                new_name = name.rpartition("_")[0] + "_{}".format(i)
                new_names[name] = new_name

        new_cluster_info = []
        for item in self.cluster_info:
            new_dict = {}
            for name, info in item.items():
                new_dict[new_names[name]] = info
            new_cluster_info.append(new_dict)
        self.cluster_info = new_cluster_info

    def _create_cluster_information(self):
        """Create a set of parameters describing the structure.

        Return cluster_info.

        cluster_info: list
            list of dictionaries with information of all clusters
            The dictionaries have the following form:
            {
             "name": Unique name describing the cluster.
                     Example:
                        "c3_3p725_1"
                     means it is a 3-body cluster (c3) with a cluster diameter
                     3.725 angstroms (3p275). The last number is a unique
                     family identification number assigned to all cluster
                     families.

             "descriptor": A string that contains a description of the cluster
                           including all of the internal distances and angles.

             "size": Number of atoms in the clusters.

             "symm_group": Translational symmetry group of the cluster

             "ref_indx": Index of a reference atom for the prototype cluster.

             "indices": List containing the indices of atoms in a cluster.
                        There can be more than on set of indices that form the
                        same cluster family, so it is in a list of list format.
                        An example of a three body cluster:
                            ref_indx: 0
                            indices: [[1, 2, 6], [7, 8, 27], [10, 19, 30]]
                        A full list of indices in the cluster is obtained by
                            [ref_indx] + [10, 19, 30] --> [0, 10, 19, 30]

             "order": The order in which the atoms in the clusters are
                      represented. The indices of atoms in a cluster need to be
                      sorted in a prefined order to ensure a consistent
                      assignment of the basis function.
                      With the same 4-body cluster above, the 3 sets of indices
                      can have the order defined as:
                        [[0, 1, 2, 3], [1, 3, 2, 0], [2, 0, 3, 1]].
                      Then, the third cluster in the example is sorted as
                        Unordered: [ref_indx] + [10, 19, 30] -> [0, 10, 19, 30]
                        Ordered: [19, 0, 30, 10]

             "equiv_sites": List of indices of symmetrically equivalent sites.
                            After ordering, the symmetrically equivalent sites
                            are the same for all clusters in the family.
                            The same 4-body cluster example:
                            1) If the clusters have no equivalent sites,
                               this equiv_sites = []
                            2) If atoms in position 1 and 2 of the ordered
                               cluster list are equivalent, then
                               equiv_sites = [[1, 2]]. Which means that
                               0 and 30 in the cluster [19, 0, 30, 10] are
                               equivalent
                            3) If atom 1, 2 are equivalent and atom 0, 3
                               are equivalent equiv_sites = [[1, 2], [0, 3]]
                               For the cluster [19, 0, 30, 10] that means that
                               0 and 30 are equivalent, and 19 and 10 are
                               equivalent
                            4) If all atoms are symmetrically equivalent
                               equiv_sites = [[0, 1, 2, 3]]

            }
        """
        cluster_info = []
        fam_identifier = []
        float_dist = FloatingPointClassifier(self.dist_num_dec)
        float_ang = FloatingPointClassifier(0)
        float_max_dia = FloatingPointClassifier(self.dist_num_dec)

        # Need newer version
        if np.version.version <= '1.13':
            raise ValueError('Numpy version > 1.13 required')

        # determine cluster information for each inequivalent site
        # (based on translation symmetry)
        for site, ref_indx in enumerate(self.ref_index_trans_symm):
            cluster_info_symm = {}
            cluster_info_symm['c0'] = {
                "indices": [],
                "equiv_sites": [],
                "order": [],
                "ref_indx": ref_indx,
                "symm_group": site,
                "descriptor": "empty",
                "name": "c0",
                "max_cluster_dia": 0.0,
                "size": 0
            }

            cluster_info_symm['c1'] = {
                "indices": [],
                "equiv_sites": [],
                "order": [0],
                "ref_indx": ref_indx,
                "symm_group": site,
                "descriptor": "point_cluster",
                "name": 'c1',
                "max_cluster_dia": 0.0,
                "size": 1
            }

            for size in range(2, self.max_cluster_size + 1):
                indices = self.indices_of_nearby_atom(ref_indx, size)
                if self.ignore_background_atoms:
                    indices = [i for i in indices if
                               i not in self.background_indices]
                indx_set = []
                descriptor_str = []
                order_set = []
                equiv_sites_set = []
                max_cluster_diameter = []
                for k in combinations(indices, size - 1):
                    d = self.get_min_distance((ref_indx,) + k)
                    if max(d) > self.max_cluster_dia[size]:
                        continue
                    order, eq_sites, string_description = \
                        sort_by_internal_distances(self.atoms, (ref_indx,) + k,
                                                   float_dist,
                                                   float_ang)
                    descriptor_str.append(string_description)
                    indx_set.append(k)
                    order_set.append(order)
                    equiv_sites_set.append(eq_sites)
                    max_cluster_diameter.append(float_max_dia.get(max(d)))

                if not descriptor_str:
                    msg = "There is no cluster with size {}.\n".format(size)
                    msg += "Reduce max_cluster_size or "
                    msg += "increase max_cluster_dia."
                    raise ValueError(msg)

                # categorize the distances
                unique_descriptors = list(set(descriptor_str))
                unique_descriptors = sorted(unique_descriptors, reverse=True)

                for descr in unique_descriptors:
                    if descr not in fam_identifier:
                        fam_identifier.append(descr)

                for desc in unique_descriptors:
                    # Find the maximum cluster diameter of this category
                    indx = descriptor_str.index(desc)
                    max_dia = max_cluster_diameter[indx]
                    fam_id = fam_identifier.index(desc)
                    name = get_unique_name(size, max_dia, fam_id)

                    cluster_info_symm[name] = {
                        "indices": [],
                        "equiv_sites": equiv_sites_set[indx],
                        "order": [],
                        "ref_indx": ref_indx,
                        "symm_group": site,
                        "descriptor": desc,
                        "name": name,
                        "max_cluster_dia": max_dia,
                        "size": size,
                    }

                for x in range(len(indx_set)):
                    category = unique_descriptors.index(descriptor_str[x])
                    max_dia = max_cluster_diameter[x]
                    fam_id = fam_identifier.index(unique_descriptors[category])
                    name = get_unique_name(size, max_dia, fam_id)
                    cluster_info_symm[name]["indices"].append(indx_set[x])
                    cluster_info_symm[name]["order"].append(order_set[x])

                    assert cluster_info_symm[name]["equiv_sites"] \
                        == equiv_sites_set[x]
                    assert cluster_info_symm[name]["max_cluster_dia"] == \
                        max_cluster_diameter[x]
                    assert cluster_info_symm[name]["descriptor"] == \
                        descriptor_str[x]

            cluster_info.append(cluster_info_symm)
        self.cluster_info = cluster_info
        self._assign_correct_family_identifier()

    @property
    def unique_indices(self):
        """Creates a list with the unique indices."""
        all_indices = deepcopy(self.ref_index_trans_symm)
        for item in self.cluster_info:
            for name, info in item.items():
                    all_indices += flatten(info["indices"])
        return list(set(all_indices))

    @property
    def multiplicity_factor(self):
        """Return the multiplicity factor of each cluster."""
        names = self.cluster_family_names
        mult_factor = {name: 0. for name in names}
        name_found = {name: False for name in names}
        normalization = {name: 0 for name in names}
        for name in names:
            for item in self.cluster_info:
                if name not in item.keys():
                    continue
                name_found[name] = True
                cluster = item[name]
                num_in_group = \
                    len(self.index_by_trans_symm[cluster["symm_group"]])
                mult_factor[name] += len(cluster["indices"]) * num_in_group
                normalization[name] += num_in_group

        for name in mult_factor.keys():
            mult_factor[name] = mult_factor[name] / normalization[name]
        for key, found in name_found.items():
            assert found
        return mult_factor

    def cluster_info_by_name(self, name):
        """Get info entries of all clusters with name."""
        name = str(name)
        info = []
        for item in self.cluster_info:
            if name in item.keys():
                info.append(item[name])
        return info

    def _create_translation_matrix(self):
        """Create and return translation matrix.

        Translation matrix maps the indices of the atoms when an atom with the
        translational symmetry is moved to the location of the reference atom.
        Used in conjunction with cluster_indx to calculate the correlation
        function of the structure.
        """
        natoms = len(self.atoms)
        unique_indices = self.unique_indices

        tm = [{} for _ in range(natoms)]

        # Add the index in the main atoms object to the tag
        for indx, atom in enumerate(self.atoms):
            atom.tag = indx

        for i, ref_indx in enumerate(self.ref_index_trans_symm):
            indices = index_by_position(self.atoms)
            tm[ref_indx] = {col: indices[col] for col in unique_indices}

            for indx in self.index_by_trans_symm[i]:
                if indx == ref_indx:
                    continue
                shifted = self.atoms.copy()
                vec = self.atoms.get_distance(indx, ref_indx, vector=True)
                shifted.translate(vec)
                shifted.wrap()

                indices = index_by_position(shifted)
                tm[indx] = {col: indices[col] for col in unique_indices}
        return tm

    def get_min_distance(self, cluster):
        """Get minimum distances.

        Get the minimum distances between the atoms in a cluster according to
        dist_matrix and return the sorted distances (reverse order)
        """
        d = []
        for t, tree in enumerate(self.kd_trees):
            row = []
            for x in combinations(cluster, 2):
                x0 = tree.data[x[0], :]
                x1 = tree.data[x[1], :]
                row.append(self._get_distance(x0, x1))
            d.append(sorted(row, reverse=True))
        return np.array(min(d))

    def _get_distance(self, x0, x1):
        """Compute the Euclidean distance between two points."""
        diff = x1 - x0
        length = np.sqrt(diff.dot(diff))
        return np.round(length, self.dist_num_dec)

    def indices_of_nearby_atom(self, ref_indx, size):
        """Return the indices of the atoms nearby.

        Indices of the atoms are only included if distances smaller than
        specified by max_cluster_dia from the reference atom index.
        """
        nearby_indices = []
        for tree in self.kd_trees:
            x0 = tree.data[ref_indx, :]
            result = tree.query_ball_point(x0, self.max_cluster_dia[size])
            nearby_indices += list(result)

        nearby_indices = list(set(nearby_indices))
        nearby_indices.remove(ref_indx)
        return nearby_indices

    # def _create_concentration_matrix(self):
    #     min_1 = np.array([i for row in self.conc_ratio_min_1 for i in row])
    #     max_1 = np.array([i for row in self.conc_ratio_max_1 for i in row])
    #     if sum(min_1) != sum(max_1):
    #         raise ValueError('conc_ratio values must be on the same scale')
    #
    #     natoms_cell = len(self.atoms_with_given_dim)
    #     if self.ignore_background_atoms:
    #         num_background = len([a.index for a in self.atoms_with_given_dim
    #                               if a.symbol in self.background_symbol])
    #         natoms_cell -= num_background
    #     natoms_ratio = sum(min_1)
    #     scale = int(natoms_cell / natoms_ratio)
    #     min_1 *= scale
    #     max_1 *= scale
    #     # special case where there is only one concentration
    #     if np.array_equal(min_1, max_1):
    #         return min_1
    #
    #     diff_1 = [i - j for i, j in zip(max_1, min_1)]
    #     nsteps_1 = max(diff_1)
    #     increment_1 = diff_1 / nsteps_1
    #
    #     if self.num_conc_var == 1:
    #         conc = np.zeros((nsteps_1 + 1, len(min_1)), dtype=int)
    #         for n in range(nsteps_1 + 1):
    #             if n == 0:
    #                 conc[0] = min_1
    #                 continue
    #             conc[n] = conc[n - 1] + increment_1
    #
    #     if self.num_conc_var == 2:
    #         min_2 = np.array([i for row in self.conc_ratio_min_2 for i in row])
    #         max_2 = np.array([i for row in self.conc_ratio_max_2 for i in row])
    #         if sum(min_2) != sum(max_2):
    #             raise ValueError('conc_ratio values must be on the same scale')
    #         scale = int(natoms_cell / natoms_ratio)
    #         min_2 *= scale
    #         max_2 *= scale
    #         diff_2 = [i - j for i, j in zip(max_2, min_2)]
    #         nsteps_2 = max(diff_2)
    #         increment_2 = diff_2 / nsteps_2
    #         conc = np.zeros((nsteps_1 + 1, nsteps_2 + 1, len(min_1)),
    #                         dtype=int)
    #         for i in range(nsteps_1 + 1):
    #             if i == 0:
    #                 conc[i][0] = min_1
    #             else:
    #                 conc[i][0] = conc[i - 1][0] + increment_1
    #             for j in range(1, nsteps_2 + 1):
    #                 conc[i][j] = conc[i][j - 1] + increment_2
    #     return conc

    @property
    def cluster_family_names(self):
        """Return a list of all cluster names."""
        families = []
        for item in self.cluster_info:
            families += list(item.keys())
        return list(set(families))

    @property
    def cluster_family_names_by_size(self):
        """Return a list of cluster familes sorted by size."""
        sort_list = []
        for item in self.cluster_info:
            for cname, c_info in item.items():
                sort_list.append((c_info["size"],
                                  c_info["max_cluster_dia"], cname))
        sort_list.sort()
        sorted_names = []
        for item in sort_list:
            if item[2] not in sorted_names:
                sorted_names.append(item[2])
        return sorted_names

    @property
    def cluster_names(self):
        """Return the cluster names including decoration numbers."""
        names = ["c0"]
        bf_list = list(range(len(self.basis_functions)))
        for item in self.cluster_info:
            for name, info in item.items():
                if info["size"] == 0:
                    continue
                eq_sites = list(info["equiv_sites"])
                for dec in product(bf_list, repeat=info["size"]):
                    dec_str = dec_string(dec, eq_sites)
                    names.append(name + '_' + dec_str)
        return list(set(names))

    def cluster_info_given_size(self, size):
        """Get the cluster info of all clusters with a given size."""
        clusters = []
        for item in self.cluster_info:
            info_dict = {}
            for key, info in item.items():
                if info["size"] == size:
                    info_dict[key] = info
            clusters.append(info_dict)
        return clusters

    def _store_data(self):
        print('Generating cluster data. It may take several minutes depending'
              ' on the values of max_cluster_size and max_cluster_dia...')
        self._create_cluster_information()
        self.trans_matrix = self._create_translation_matrix()
        # self._check_equiv_sites()
        db = connect(self.db_name)
        data = {'cluster_info': self.cluster_info,
                'trans_matrix': self.trans_matrix}

        db.write(self.atoms, name='template', data=data,
                 dims=self._size2string(), unit_cell_type=self.unit_cell_type)

    def _read_data(self):
        db = connect(self.db_name)
        try:
            select_cond = [('name', '=', 'template'),
                           ('dims', '=', self._size2string()),
                           ('unit_cell_type', '=', self.unit_cell_type)]
            row = db.get(select_cond)
            self.cluster_info = row.data.cluster_info
            self._info_entries_to_list()
            self.trans_matrix = row.data.trans_matrix
        except KeyError:
            self._store_data()
        except (AssertionError, AttributeError, RuntimeError):
            self.reconfigure_settings()

    def _info_entries_to_list(self):
        """Convert entries in cluster info to list."""
        for info in self.cluster_info:
            for name, cluster in info.items():
                cluster['indices'] = nested_array2list(cluster['indices'])
                cluster['equiv_sites'] = \
                    nested_array2list(cluster['equiv_sites'])
                cluster['order'] = nested_array2list(cluster['order'])

    def _get_name_indx(self, unique_name):
        size = int(unique_name[1])
        for symm in range(self.num_trans_symm):
            name_list = self.cluster_names[symm][size]
            try:
                n_indx = name_list.index(unique_name)
                return symm, n_indx
            except ValueError:
                continue

    # def in_conc_matrix(self, atoms):
    #     """Check to see if the passed atoms object has allowed concentration.
    #
    #     Return True if it has allowed concentration, return False otherwise.
    #     """
    #     # determine the concentration of the given atoms
    #     if self.grouped_basis is None:
    #         num_elements = self.num_elements
    #         all_elements = self.all_elements
    #     else:
    #         num_elements = self.num_grouped_elements
    #         all_elements = self.all_grouped_elements
    #
    #     conc = np.zeros(num_elements, dtype=int)
    #
    #     for x in range(num_elements):
    #         element = all_elements[x]
    #         conc[x] = len([a for a in atoms if a.symbol == element])
    #     # determine the dimensions of the concentration matrix
    #     # then, search to see if there is a match
    #     conc_shape = self.conc_matrix.shape
    #     if self.conc_matrix.ndim == 1:
    #         if np.array_equal(conc, self.conc_matrix):
    #             return True
    #     elif self.conc_matrix.ndim == 2:
    #         for x in range(conc_shape[0]):
    #             if np.array_equal(conc, self.conc_matrix[x]):
    #                 return True
    #     else:
    #         for x in range(conc_shape[0]):
    #             for y in range(conc_shape[1]):
    #                 if np.array_equal(conc, self.conc_matrix[x][y]):
    #                     return True
    #     return False

    def _group_index_by_basis_group(self):
        if self.concentration.grouped_basis is None:
            return self.index_by_basis

        index_by_grouped_basis = []
        for group in self.concentration.grouped_basis:
            indices = []
            for basis in group:
                indices.extend(self.index_by_basis[basis])
            index_by_grouped_basis.append(indices)

        for basis in index_by_grouped_basis:
            basis.sort()
        return index_by_grouped_basis

    def view_clusters(self):
        """Display all clusters along with their names."""
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        already_included_names = []
        cluster_atoms = []
        for unique_name in self.cluster_family_names_by_size:
            if unique_name in already_included_names:
                continue
            already_included_names.append(unique_name)
            for symm, entry in enumerate(self.cluster_info):
                if unique_name in entry:
                    cluster = entry[unique_name]
                    break
            if cluster["size"] <= 1:
                continue
            ref_indx = self.ref_index_trans_symm[symm]
            name = cluster["name"]

            atoms = self.atoms.copy()

            keep_indx = [ref_indx] + list(cluster["indices"][0])
            equiv = list(cluster["equiv_sites"])
            order = list(cluster["order"][0])

            if order is not None:
                keep_indx = [keep_indx[n] for n in order]

            for tag, indx in enumerate(keep_indx):
                atoms[indx].tag = tag
            if equiv:
                for group in equiv:
                    for i in range(1, len(group)):
                        atoms[keep_indx[group[i]]].tag = \
                            atoms[keep_indx[group[0]]].tag
            atoms = create_cluster(atoms, keep_indx)
            atoms.info = {'name': name}
            cluster_atoms.append(atoms)

        images = Images()
        images.initialize(cluster_atoms)
        gui = GUI(images, expr='')
        gui.show_name = True
        gui.run()

    def reconfigure_settings(self):
        """Reconfigure settings stored in DB file."""
        db = connect(self.db_name)
        ids = [row.id for row in db.select(name='information')]
        db.delete(ids)
        self._store_data()

    def _check_first_elements(self):
        basis_elements = self.basis_elements
        num_basis = self.num_basis
        # This condition can be relaxed in the future
        first_elements = []
        for elements in basis_elements:
            first_elements.append(elements[0])
        if len(set(first_elements)) != num_basis:
            raise ValueError("First element of different basis should not be "
                             "the same.")

    def save(self, filename):
        """Write Setting object to a file in JSON format.

        Arguments:
        =========
        filename: str
            Name of the file to store the necessary settings to initialize
            the Cluster Expansion calculations.
        """
        class_types = ['CEBulk', 'CECrystal']
        if type(self).__name__ not in class_types:
            raise NotImplementedError('Class {}'.format(type(self).__name__)
                                      + 'is not supported.')

        import json
        if type(self).__name__ == 'CEBulk':
            self.kwargs['classtype'] = 'CEBulk'
        else:
            self.kwargs['classtype'] = 'CECrystal'
        # Write keyword arguments necessary for initializing the class
        with open(filename, 'w') as outfile:
            json.dump(self.kwargs, outfile, indent=2)

    def _get_shortest_distance_in_unitcell(self):
        """Find the shortest distance between the atoms in the unit cell."""
        if len(self.unit_cell) == 1:
            lengths = self.unit_cell.get_cell_lengths_and_angles()[:3]
            return min(lengths)

        dists = []
        for ref_atom in range(len(self.unit_cell)):
            indices = list(range(len(self.unit_cell)))
            indices.remove(ref_atom)
            dists += list(self.unit_cell.get_distances(ref_atom, indices,
                                                       mic=True))
        return min(dists)
