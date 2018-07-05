"""Definition of ClusterExpansionSetting Class.

This module defines the base-class for storing the settings for performing
Cluster Expansion in different conditions.
"""
import os
from itertools import combinations, combinations_with_replacement, product
from copy import deepcopy
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase.db import connect

from ase.ce.tools import wrap_and_sort_by_position, index_by_position, flatten
from ase.ce.tools import sort_by_internal_distances, create_cluster
from ase.ce.tools import sorted_internal_angles, ndarray2list



class ClusterExpansionSetting:
    """Base-class for all Cluster Expansion settings."""

    def __init__(self, conc_args=None, db_name=None, max_cluster_size=4,
                 max_cluster_dist=None, basis_elements=None,
                 grouped_basis=None, ignore_background_atoms=False):
        self._check_conc_ratios(conc_args)
        self.db_name = db_name
        self.max_cluster_size = max_cluster_size
        self.basis_elements = basis_elements
        self.grouped_basis = grouped_basis
        self.all_elements = sorted([item for row in basis_elements for
                                    item in row])
        self.ignore_background_atoms = ignore_background_atoms
        self.background_symbol = []
        if ignore_background_atoms:
            self.background_symbol = self._get_background_symbol()

        self.num_elements = len(self.all_elements)
        self.unique_elements = sorted(list(set(deepcopy(self.all_elements))))
        self.num_unique_elements = len(self.unique_elements)
        self.max_cluster_dist, self.supercell_scale_factor = \
            self._get_max_cluster_dist_and_scale_factor(max_cluster_dist)
        if len(self.basis_elements) != self.num_basis:
            raise ValueError("list of elements is needed for each basis")
        if grouped_basis is None:
            self._check_basis_elements()
        else:
            if not isinstance(grouped_basis, list):
                raise TypeError('grouped_basis should be a list')
            self.num_groups = len(self.grouped_basis)
            self._check_grouped_basis_elements()

        self.spin_dict = self._get_spin_dict()
        self.basis_functions = self._get_basis_functions()
        self.unique_cluster_names = None
        self.atoms = self._create_template_atoms()
        self.background_atom_indices = []
        if ignore_background_atoms:
            self.background_atom_indices = [a.index for a in self.atoms if
                                            a.symbol in self.background_symbol]

        self.index_by_trans_symm = self._group_indices_by_trans_symmetry()
        self.num_trans_symm = len(self.index_by_trans_symm)
        self.ref_index_trans_symm = [i[0] for i in self.index_by_trans_symm]
        self.kd_trees = self._create_kdtrees()

        # print('num_unique_elements: {}'.format(self.num_unique_elements))
        # print('unique_elements: {}'.format(self.unique_elements))
        # print('all_elements: {}'.format(self.all_elements))
        # print('basis_elements: {}'.format(self.basis_elements))
        # print('spin_dict: {}'.format(self.spin_dict))
        # print('basis_functions: {}'.format(self.basis_functions))
        # print('background_symbol: {}'.format(self.background_symbol))

        if not os.path.exists(db_name):
            self._store_data()
        else:
            self._read_data()

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
        return True

    def _get_diag_lengths(self):
        """Return the length of all diagonals"""
        cell = self.atoms_with_given_dim.get_cell()
        a1 = cell[0,:]
        a2 = cell[1,:]
        a3 = cell[2,:]
        diags = [a1+a2, a1+a3, a2+a3, a1+a2+a3]
        lengths = []
        for diag in diags:
            length = np.sqrt(np.sum(diag**2))
            lengths.append(length)
        return np.array(lengths)

    def _get_max_cluster_dist_and_scale_factor(self, max_cluster_dist):
        min_length = self._get_max_cluster_dist()

        # ------------------------------------- #
        # Get max_cluster_dist in an array form #
        # ------------------------------------- #
        # max_cluster_dist is list or array
        if isinstance(max_cluster_dist, (list, np.ndarray)):
            # Length should be either max_cluster_size+1 or max_cluster_size-1
            if len(max_cluster_dist) == self.max_cluster_size + 1:
                for i in range(2):
                    max_cluster_dist[i] = 0.
                max_cluster_dist = np.array(max_cluster_dist, dtype=float)
            elif len(max_cluster_dist) == self.max_cluster_size - 1:
                max_cluster_dist = np.array(max_cluster_dist, dtype=float)
                max_cluster_dist = np.insert(max_cluster_dist, 0, [0., 0.])
            else:
                raise ValueError("Invalid length for max_cluster_dist.")
        # max_cluster_dist is int or float
        elif isinstance(max_cluster_dist, (int, float)):
            max_cluster_dist *= np.ones(self.max_cluster_size - 1, dtype=float)
            max_cluster_dist = np.insert(max_cluster_dist, 0, [0., 0.])
        # Case for *None* or something else
        else:
            max_cluster_dist = np.ones(self.max_cluster_size - 1, dtype=float)
            max_cluster_dist *= min_length
            max_cluster_dist = np.insert(max_cluster_dist, 0, [0., 0.])

        # --------------------------------- #
        # Get scale_factor in an array form #
        # --------------------------------- #
        atoms = self.atoms_with_given_dim
        lengths = atoms.get_cell_lengths_and_angles()[:3] / 2
        scale_factor = max(max_cluster_dist) / lengths
        scale_factor = np.ceil(scale_factor).astype(int)

        return np.around(max_cluster_dist, self.dist_num_dec), scale_factor

    def _get_max_cluster_dist(self):
        cell = self.atoms_with_given_dim.get_cell()
        lengths = []
        for w in product([-1, 0, 1], repeat=3):
            vec = cell.T.dot(w)
            if w == (0, 0, 0):
                continue
            lengths.append(np.sqrt(vec.dot(vec)))
        # Introduce tolerance to make max distance strictly
        # smaller than half of the shortest cell dimension
        tol = 1E-4
        return min(lengths) / 2 - tol

    def _get_background_symbol(self):
        """Get symbol of the background atoms.

        This method also modifies grouped_basis, conc_args, num_basis, basis,
        and all_elements attributes to reflect the changes from ignoring the
        background atoms.
        """
        # check if any basis consists of only one element type
        basis = [i for i, b in enumerate(self.basis_elements) if len(b) == 1]
        if not basis:
            return []

        symbol = [b[0] for b in self.basis_elements if len(b) == 1]
        self.num_basis -= len(basis)

        # remap indices if the basis are grouped, then change grouped_basis
        # and conc_ratio_min/max accordingly
        if self.grouped_basis is not None:
            remapped = []
            for i, group in enumerate(self.grouped_basis):
                if group[0] in basis:
                    remapped.append(i)
            for i in sorted(remapped, reverse=True):
                del self.grouped_basis[i]
                del self.conc_ratio_min_1[i]
                del self.conc_ratio_max_1[i]
                if self.num_conc_var == 2:
                    del self.conc_ratio_min_2[i]
                    del self.conc_ratio_max_2[i]

        # change basis_elements
        for i in sorted(basis, reverse=True):
            if hasattr(self, 'basis'):
                del self.basis[i]
            del self.basis_elements[i]
            if self.grouped_basis is None:
                del self.conc_ratio_min_1[i]
                del self.conc_ratio_max_1[i]
                if self.num_conc_var == 2:
                    del self.conc_ratio_min_2[i]
                    del self.conc_ratio_max_2[i]

        # change all_elements
        for s in symbol:
            self.all_elements.remove(s)

        if self.grouped_basis is None:
            return list(set(symbol))

        # reassign grouped_basis if they are grouped
        for ref in sorted(basis, reverse=True):
            for i, group in enumerate(self.grouped_basis):
                for j, element in enumerate(group):
                    if element > ref:
                        self.grouped_basis[i][j] -= 1
        return list(set(symbol))

    def _check_basis_elements(self):
        error = False
        # check dimensions of the element list and concentration ratio lists
        if self.num_basis == 1:
            if not(self.num_basis == len(self.conc_ratio_min_1)
                   == len(self.conc_ratio_max_1)):
                error = True

            if self.num_conc_var == 2:
                if not(self.num_basis == len(self.conc_ratio_min_2)
                       == len(self.conc_ratio_max_2)):
                    error = True
        else:
            element_size = [len(row) for row in self.basis_elements]
            min_1_size = [len(row) for row in self.conc_ratio_min_1]
            max_1_size = [len(row) for row in self.conc_ratio_max_1]
            if not element_size == min_1_size == max_1_size:
                error = True
            if self.num_conc_var == 2:
                min_2_size = [len(row) for row in self.conc_ratio_min_2]
                max_2_size = [len(row) for row in self.conc_ratio_max_2]
                if not element_size == min_2_size == max_2_size:
                    error = True
        if error:
            raise ValueError('lengths of the basis_elements and conc_ratio'
                             ' lists are not the same')

    def _check_grouped_basis_elements(self):
        # check number of basis
        num_basis = len([i for sub in self.grouped_basis for i in sub])
        if num_basis != self.num_basis:
            raise ValueError('grouped_basis do not contain all the basis')

        # check if grouped basis have same elements
        for group in self.grouped_basis:
            ref_elements = self.basis_elements[group[0]]
            for indx in group[1:]:
                if self.basis_elements[indx] != ref_elements:
                    raise ValueError('elements in the same group must be same')

    def _get_spin_dict(self):
        # Find odd/even
        spin_values = []
        if self.num_unique_elements % 2 == 1:
            highest = (self.num_unique_elements - 1) / 2
        else:
            highest = self.num_unique_elements / 2
        # Assign spin value for each element
        while highest > 0:
            spin_values.append(highest)
            spin_values.append(-highest)
            highest -= 1
        if self.num_unique_elements % 2 == 1:
            spin_values.append(0)

        spin_dict = {}
        for x in range(self.num_unique_elements):
            spin_dict[self.unique_elements[x]] = spin_values[x]
        return spin_dict

    def _get_basis_functions(self):
        """Create basis functions to guarantee the orthonormality."""
        if self.num_unique_elements == 2:
            d0_0 = 1.
        elif self.num_unique_elements == 3:
            d0_0 = np.sqrt(3. / 2)
            c0_1 = np.sqrt(2)
            c1_1 = -3 / np.sqrt(2)
        elif self.num_unique_elements == 4:
            d0_0 = np.sqrt(2. / 5)
            c0_1 = -5. / 3
            c1_1 = 2. / 3
            d0_1 = -17. / (3 * np.sqrt(10))
            d1_1 = np.sqrt(5. / 2) / 3
        elif self.num_unique_elements == 5:
            d0_0 = 1. / np.sqrt(2)
            c0_1 = -1 * np.sqrt(10. / 7)
            c1_1 = np.sqrt(5. / 14)
            d0_1 = -17. / (6 * np.sqrt(2))
            d1_1 = 5. / (6 * np.sqrt(2))
            c0_2 = 3 * np.sqrt(2. / 7)
            c1_2 = -155. / (12 * np.sqrt(14))
            c2_2 = 15 * np.sqrt(7. / 2) / 12
        elif self.num_unique_elements == 6:
            d0_0 = np.sqrt(3. / 14)
            c0_1 = -np.sqrt(2)
            c1_1 = 3. / (7 * np.sqrt(2))
            d0_1 = -7. / 6
            d1_1 = 1. / 6
            c0_2 = 9 * np.sqrt(3. / 2) / 5
            c1_2 = -101. / (28 * np.sqrt(6))
            c2_2 = 7. / (20 * np.sqrt(6))
            d0_2 = 131. / (15 * np.sqrt(4))
            d1_2 = -7 * np.sqrt(7. / 2) / 12
            d2_2 = np.sqrt(7. / 2) / 20
        else:
            raise ValueError("only compounds consisting of 2 to 6 types of"
                             " elements are supported")

        bf_list = []

        bf = {}
        for key, value in self.spin_dict.items():
            bf[key] = d0_0 * value
        bf_list.append(bf)

        if self.num_unique_elements > 2:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = c0_1 + (c1_1 * value**2)
            bf_list.append(bf)

        if self.num_unique_elements > 3:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = d0_1 * value + (d1_1 * (value**3))
            bf_list.append(bf)

        if self.num_unique_elements > 4:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = c0_2 + (c1_2 * (value**2)) + (c2_2 * (value**4))
            bf_list.append(bf)

        if self.num_unique_elements > 5:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = d0_2 + (d1_2 * (value**3)) + (d2_2 * (value**5))
            bf_list.append(bf)

        return bf_list

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
        vec = self.atoms.get_cell()
        trans = [[0., 0., 0.],
                 vec[0] / 2,
                 vec[1] / 2,
                 vec[2] / 2,
                 (vec[0] + vec[1]) / 2,
                 (vec[0] + vec[2]) / 2,
                 (vec[1] + vec[2]) / 2,
                 (vec[0] + vec[1] + vec[2]) / 2]
        kd_trees = []
        for t in range(8):
            shifted = self.atoms.copy()
            shifted.translate(trans[t])
            shifted.wrap()
            kd_trees.append(KDTree(shifted.get_positions()))
        return kd_trees

    def _group_indices_by_trans_symmetry(self):
        """Group indices by translational symmetry."""
        indices = [a.index for a in self.atoms]
        ref_indx = indices[0]
        # Group all the indices together if its atomic number and position
        # sequences are same
        indx_by_equiv = []
        for i, indx in enumerate(indices):
            if indx not in self.background_atom_indices:
                break

        vec = self.atoms.get_distance(indices[i], ref_indx, vector=True)
        shifted = self.atoms.copy()
        shifted.translate(vec)
        shifted = wrap_and_sort_by_position(shifted)
        an = shifted.get_atomic_numbers()
        pos = shifted.get_positions()

        temp = [[indices[i]]]
        equiv_group_an = [an]
        equiv_group_pos = [pos]
        for indx in indices[i + 1:]:
            if indx in self.background_atom_indices:
                continue
            vec = self.atoms.get_distance(indx, ref_indx, vector=True)
            shifted = self.atoms.copy()
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
        return indx_by_equiv

    def _get_grouped_basis_elements(self):
        """Group elements in the 'equivalent group' together in a list."""
        grouped_basis_elements = []
        for group in self.grouped_basis:
            grouped_basis_elements.append(self.basis_elements[group[0]])
        return grouped_basis_elements

    def _get_cluster_information(self):
        """Create a set of parameters used to describe the structure.

        Return cluster_names, cluster_dist, cluster_indx.

        cluster_names: list
            names of clusters (i.e., c{x}_{y}_{#1}_{#2}, where
                x  = number of atoms in cluster
                y  = maximum distance of cluster
                #1 = names are sequenced in a decreasing order when they have
                     the same number of atoms and max. diameter
                #2 = account for different combination of basis function for
                     the cases where # of constituting elements is 3 or more).

        cluster_dist: list
            distances of the constituting atoms in cluster.

        cluster_indx: list
            list of indices that constitute the cluster based on the reference
            atom (atom with lowest index for each translational symmetry
            inequivalent site).
        """
        cluster_names = []
        cluster_dist = []
        cluster_indx = []
        cluster_order = []
        cluster_eq_sites = []
        self.unique_cluster_names = ['c0', 'c1']
        # Need newer version
        if np.version.version <= '1.13':
            raise ValueError('Numpy version > 1.13 required')

        # determine cluster information for each inequivalent site
        # (based on translation symmetry)
        for site, ref_indx in enumerate(self.ref_index_trans_symm):
            # Add None for dist and indx for c0 and c1
            cluster_names.append([['c0'], ['c1']])
            cluster_dist.append([[None], [None]])
            cluster_indx.append([[None], [None]])
            cluster_order.append([[None], [None]])
            cluster_eq_sites.append([[None], [None]])

            for size in range(2, self.max_cluster_size + 1):
                print("Generating clusters with size {}".format(size))
                indices = self.indices_of_nearby_atom(ref_indx, size)
                if self.ignore_background_atoms:
                    indices = [i for i in indices if
                               i not in self.background_atom_indices]
                indx_set = []
                dist_set = []
                order_set = []
                equiv_sites_set = []
                for k in combinations(indices, size - 1):
                    d = self.get_min_distance((ref_indx,) + k)
                    if max(d) > self.max_cluster_dist[size]:
                        continue
                    d_list = sorted(d.tolist(), reverse=True)
                    internal_angles = \
                        sorted_internal_angles(create_cluster(self.atoms,
                                                              (ref_indx,) + k))
                    d_list += internal_angles
                    dist_set.append(d_list)

                    order, eq_sites = \
                        sort_by_internal_distances(self.atoms, (ref_indx,) + k)
                    indx_set.append(k)
                    order_set.append(order)
                    equiv_sites_set.append(eq_sites)

                if not dist_set:
                    msg = "There is no cluster with size {}.\n".format(size)
                    msg += "Reduce max_cluster_size or "
                    msg += "increase max_cluster_dist."
                    raise ValueError(msg)

                # categorize the distances
                dist_types = np.unique(dist_set, axis=0).tolist()
                dist_types = sorted(dist_types, reverse=True)
                cluster_dist[site].append(dist_types)

                # categorieze the indices to the distance types it belongs
                indx_types = [[] for _ in range(len(dist_types))]
                ordered_indx = [[] for _ in range(len(dist_types))]
                equiv_sites = [None for _ in range(len(dist_types))]
                for x in range(len(indx_set)):
                    category = dist_types.index(dist_set[x])
                    indx_types[category].append(list(indx_set[x]))
                    ordered_indx[category].append(order_set[x])

                    if equiv_sites[category] is None:
                        equiv_sites[category] = equiv_sites_set[x]
                    else:
                        assert equiv_sites[category] == equiv_sites_set[x]
                    # indx_types[category].append(indx_set[x])
                cluster_indx[site].append(indx_types)
                cluster_order[site].append(ordered_indx)
                cluster_eq_sites[site].append(equiv_sites)

        # Cluster names can be incorrectly assigned for cluster size of 2 and
        # above. Assign global names for those clusters to avoid wrong name
        # assignments.
        for size in range(2, self.max_cluster_size + 1):
            unique_dist = []
            for site in range(self.num_trans_symm):
                unique_dist.extend(cluster_dist[site][size])
            unique_dist = sorted(np.unique(unique_dist, axis=0).tolist(),
                                 reverse=True)

            # name the clusters based on its max. distance
            unique_names = []
            counter = 1
            for k in range(len(unique_dist)):
                name = 'c{0}_{1:.{prec}f}'.format(size, unique_dist[k][0],
                                                  prec=self.dist_num_dec)
                name = name.replace('.', 'p')
                if k == 0:
                    name += '_{}'.format(counter)
                    unique_names.append(name)
                    counter += 1
                    continue
                if unique_dist[k][0] != unique_dist[k - 1][0]:
                    counter = 1
                name += '_{}'.format(counter)
                unique_names.append(name)
                counter += 1

            self.unique_cluster_names.extend(unique_names)
            # Assign name in the correct position of the cluster_name list
            for basis in range(self.num_trans_symm):
                names = []
                for dist in cluster_dist[basis][size]:
                    indx = unique_dist.index(dist)
                    names.append(unique_names[indx])
                cluster_names[basis].append(names)

        return cluster_names, cluster_dist, cluster_indx, cluster_order, \
            cluster_eq_sites

    def _create_translation_matrix(self):
        """Create and return translation matrix.

        Translation matrix maps the indices of the atoms when an atom with the
        translational symmetry is moved to the location of the reference atom.
        Used in conjunction with cluster_indx to calculate the correlation
        function of the structure.
        """
        natoms = len(self.atoms)
        unique_indices = list(set(flatten(deepcopy(self.cluster_indx))))
        unique_indices.remove(None)
        unique_indices = [0] + unique_indices

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
        specified by max_cluster_dist from the reference atom index.
        """
        nearby_indices = []
        for tree in self.kd_trees:
            x0 = tree.data[ref_indx, :]
            result = tree.query_ball_point(x0, self.max_cluster_dist[size])
            nearby_indices += list(result)

        nearby_indices = list(set(nearby_indices))
        nearby_indices.remove(ref_indx)
        return nearby_indices

    def _create_concentration_matrix(self):
        min_1 = np.array([i for row in self.conc_ratio_min_1 for i in row])
        max_1 = np.array([i for row in self.conc_ratio_max_1 for i in row])
        if sum(min_1) != sum(max_1):
            raise ValueError('conc_ratio values must be on the same scale')

        natoms_cell = len(self.atoms_with_given_dim)
        if self.ignore_background_atoms:
            num_background = len([a.index for a in self.atoms_with_given_dim
                                  if a.symbol in self.background_symbol])
            natoms_cell -= num_background
        natoms_ratio = sum(min_1)
        scale = int(natoms_cell / natoms_ratio)
        min_1 *= scale
        max_1 *= scale
        # special case where there is only one concentration
        if np.array_equal(min_1, max_1):
            return min_1

        diff_1 = [i - j for i, j in zip(max_1, min_1)]
        nsteps_1 = max(diff_1)
        increment_1 = diff_1 / nsteps_1

        if self.num_conc_var == 1:
            conc = np.zeros((nsteps_1 + 1, len(min_1)), dtype=int)
            for n in range(nsteps_1 + 1):
                if n == 0:
                    conc[0] = min_1
                    continue
                conc[n] = conc[n - 1] + increment_1

        if self.num_conc_var == 2:
            min_2 = np.array([i for row in self.conc_ratio_min_2 for i in row])
            max_2 = np.array([i for row in self.conc_ratio_max_2 for i in row])
            if sum(min_2) != sum(max_2):
                raise ValueError('conc_ratio values must be on the same scale')
            scale = int(natoms_cell / natoms_ratio)
            min_2 *= scale
            max_2 *= scale
            diff_2 = [i - j for i, j in zip(max_2, min_2)]
            nsteps_2 = max(diff_2)
            increment_2 = diff_2 / nsteps_2
            conc = np.zeros((nsteps_1 + 1, nsteps_2 + 1, len(min_1)),
                            dtype=int)
            for i in range(nsteps_1 + 1):
                if i == 0:
                    conc[i][0] = min_1
                else:
                    conc[i][0] = conc[i - 1][0] + increment_1
                for j in range(1, nsteps_2 + 1):
                    conc[i][j] = conc[i][j - 1] + increment_2
        return conc

    def _store_data(self):
        print('Generating cluster data. It may take several minutes depending'
              ' on the values of max_cluster_size and max_cluster_dist...')
        self.cluster_names, self.cluster_dist, self.cluster_indx, \
        self.cluster_order, self.cluster_eq_sites = \
            self._get_cluster_information()
        self.trans_matrix = self._create_translation_matrix()
        self.conc_matrix = self._create_concentration_matrix()
        self.full_cluster_names = self._get_full_cluster_names()
        self._check_equiv_sites()
        db = connect(self.db_name)
        data = {'cluster_names': self.cluster_names,
                'cluster_dist': self.cluster_dist,
                'cluster_indx': self.cluster_indx,
                'cluster_order': self.cluster_order,
                'cluster_eq_sites': self.cluster_eq_sites,
                'trans_matrix': self.trans_matrix,
                'conc_matrix': self.conc_matrix,
                'full_cluster_names': self.full_cluster_names,
                'unique_cluster_names': self.unique_cluster_names}

        db.write(self.atoms, name='information', data=data)

    def _read_data(self):
        db = connect(self.db_name)
        try:
            row = db.get('name=information')
            self.cluster_names = row.data.cluster_names
            self.cluster_dist = row.data.cluster_dist
            self.cluster_indx = ndarray2list(row.data.cluster_indx)
            self.cluster_order = ndarray2list(row.data.cluster_order)
            self.cluster_eq_sites = ndarray2list(row.data.cluster_eq_sites)
            self.trans_matrix = row.data.trans_matrix
            self.conc_matrix = row.data.conc_matrix
            self.full_cluster_names = row.data.full_cluster_names
            self.unique_cluster_names = row.data.unique_cluster_names
        except KeyError:
            self._store_data()
        except (AssertionError, AttributeError):
            self.reconfigure_settings()

    def _get_full_cluster_names(self):
        full_names = []
        bf_list = list(range(len(self.basis_functions)))
        for prefix in self.unique_cluster_names:
            n = int(prefix[1])
            if n == 0:
                full_names.append(prefix)
            else:
                comb = list(combinations_with_replacement(bf_list, r=n))
                for dec in comb:
                    dec_string = ''.join(str(i) for i in dec)
                    full_names.append(prefix + '_' + dec_string)
        return full_names

    def in_conc_matrix(self, atoms):
        """Check to see if the passed atoms object has allowed concentration.

        Return True if it has allowed concentration, return False otherwise.
        """
        # determine the concentration of the given atoms
        if self.grouped_basis is None:
            num_elements = self.num_elements
            all_elements = self.all_elements
        else:
            num_elements = self.num_grouped_elements
            all_elements = self.all_grouped_elements

        conc = np.zeros(num_elements, dtype=int)
        for x in range(num_elements):
            element = all_elements[x]
            conc[x] = len([a for a in atoms if a.symbol == element])

        # determine the dimensions of the concentration matrix
        # then, search to see if there is a match
        conc_shape = self.conc_matrix.shape
        if self.conc_matrix.ndim == 1:
            if np.array_equal(conc, self.conc_matrix):
                return True
        elif self.conc_matrix.ndim == 2:
            for x in range(conc_shape[0]):
                if np.array_equal(conc, self.conc_matrix[x]):
                    return True
        else:
            for x in range(conc_shape[0]):
                for y in range(conc_shape[1]):
                    if np.array_equal(conc, self.conc_matrix[x][y]):
                        return True
        return False

    def _group_index_by_basis_group(self):
        index_by_grouped_basis = []
        for group in self.grouped_basis:
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
        location = []
        for unique_name in self.unique_cluster_names:
            cluster_size = int(unique_name[1])
            for symm_indx, names_by_size in enumerate(self.cluster_names):
                try:
                    indx = names_by_size[cluster_size].index(unique_name)
                    location.append([symm_indx, cluster_size, indx])
                    break
                except ValueError:
                    continue

        cluster_atoms = []
        center = np.sum(self.atoms.cell, axis=0) / 2
        for i, loc in enumerate(location):
            ref_indx = self.ref_index_trans_symm[loc[0]]
            name = self.unique_cluster_names[i]
            size = loc[1]
            if size == 0:
                continue
            elif size == 1:
                keep_indx = [ref_indx]
            else:
                keep_indx = [ref_indx]
                keep_indx.extend(self.cluster_indx[loc[0]][loc[1]][loc[2]][0])
            equiv = self.cluster_eq_sites[loc[0]][loc[1]][loc[2]]
            try:
                order = self.cluster_order[loc[0]][loc[1]][loc[2]][0]
            except:
                order = None
            atoms = self.atoms.copy()
            if order is not None:
                keep_indx = [keep_indx[indx] for indx in order]

            for tag, indx in enumerate(keep_indx):
                atoms[indx].tag = tag
            if equiv:
                for group in equiv:
                    for i in range(1, len(group)):
                        atoms[keep_indx[group[i]]].tag = \
                            atoms[keep_indx[group[0]]].tag

            ref_pos = atoms[ref_indx].position

            if len(keep_indx) >= 2:
                atoms = create_cluster(atoms, keep_indx)
            else:
                atoms = atoms[keep_indx]
                atoms.center()
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
        if self.grouped_basis:
            basis_elements = self.grouped_basis_elements
            num_basis = self.num_grouped_basis
        else:
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
        class_types = ['BulkCrystal', 'BulkSpacegroup']
        if type(self).__name__ not in class_types:
            raise NotImplementedError('Class {}'.format(type(self).__name__)
                                      + 'is not supported.')

        import json
        if type(self).__name__ == 'BulkCrystal':
            self.kwargs['classtype'] = 'BulkCrystal'
        else:
            self.kwargs['classtype'] = 'BulkSpacegroup'
        # Write keyword arguments necessary for initializing the class
        with open(filename, 'w') as outfile:
            json.dump(self.kwargs, outfile, indent=2)

    def _check_equiv_sites(self):
        """Check that the equivalent sites array is valid"""

        for cluster_same_symm in self.cluster_eq_sites:
            for cluster_same_size in cluster_same_symm:
                for cluster_cat in cluster_same_size:
                    if cluster_cat is None:
                        continue

                    if not cluster_cat:
                        continue

                    # Flatten the list of categories
                    flat = []
                    for equiv in cluster_cat:
                        flat += equiv
                    flat.sort()
                    flat_unique = list(set(flat))
                    flat_unique.sort()

                    if flat != flat_unique:
                        msg = "Same site are in multiple equivalent groups.\n"
                        msg += "This is should never happen.\n"
                        msg += "Equiv sites list {}\n".format(flat)
                        msg += "Equiv sites list unique {}".format(flat_unique)
                        raise ValueError(msg)
