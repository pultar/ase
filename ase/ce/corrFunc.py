"""Module for calculating correlation functions."""
import numpy as np
from itertools import product
from ase.atoms import Atoms
from ase.ce import BulkCrystal, BulkSpacegroup
from ase.ce.tools import wrap_and_sort_by_position, dec_string, equivalent_deco
from ase.db import connect


class CorrFunction(object):
    """Calculate the correlation function.

    Arguments
    =========
    setting: settings object
    """

    def __init__(self, setting):
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            raise TypeError("setting must be BulkCrystal or BulkSpacegroup "
                            "object")
        self.setting = setting

        self.index_by_trans_symm = setting.index_by_trans_symm
        self.num_trans_symm = setting.num_trans_symm
        self.ref_index_trans_symm = setting.ref_index_trans_symm

    def get_c1(self, atoms, dec):
        """Get correlation function for single-body clusters."""
        c1 = 0
        for element, spin in self.setting.basis_functions[dec].items():
            num_element = len([a for a in atoms if a.symbol == element])
            c1 += num_element * spin
        c1 /= float(len(atoms))
        return c1

    def get_cf(self, atoms, return_type='dict'):
        """Calculate correlation function for all possible clusters.

        Arguments:
        =========
        atoms: Atoms object

        return_type: str
            -'dict' (default): returns a dictionary (e.g., {'name': cf_value})
            -'tuple': returns a list of tuples (e.g., [('name', cf_value)])
            -'array': NumPy array containing *only* the correlation function
                      values in the same order as the order in
                      "setting.full_cluster_names"
        """
        if isinstance(atoms, Atoms):
            atoms = self.check_and_convert_cell_size(atoms.copy())
        else:
            raise TypeError('atoms must be an Atoms object')

        bf_list = list(range(len(self.setting.basis_functions)))
        cf = {}
        num_excluded_symmetry = 0
        # ----------------------------------------------------
        # Compute correlation function up the max_cluster_size
        # ----------------------------------------------------
        # loop though all cluster sizes
        for n in range(self.setting.max_cluster_size + 1):
            comb = list(product(bf_list, repeat=n))
            if n == 0:
                cf['c0'] = 1.
                continue
            if n == 1:
                for dec in comb:
                    cf['c1_{}'.format(dec[0])] = self.get_c1(atoms, dec[0])
                continue

            unique_name_list = [i for i in self.setting.unique_cluster_names
                                if int(i[1]) == n]
            # loop though all names of cluster with size n
            for unique_name in unique_name_list:


                # loop through all possible decoration numbers
                for dec in comb:
                    sp = 0.
                    count = 0

                    # need to perform for each symmetry inequivalent sites
                    for symm in range(self.num_trans_symm):
                        name_list = self.setting.cluster_names[symm][n]
                        try:
                            n_indx = name_list.index(unique_name)
                        except ValueError:
                            continue

                        indices = self.setting.cluster_indx[symm][n][n_indx]
                        indx_order = self.setting.cluster_order[symm][n][n_indx]
                        eq_sites = self.setting.cluster_eq_sites[symm][n][n_indx]

                        # Get decoration number as a string
                        dec_str = dec_string(dec, eq_sites)
                        cf_name = '{}_{}'.format(unique_name, dec_str)
                        if cf_name in cf.keys():
                            # Skip this because it has already been taken into
                            # account
                            num_excluded_symmetry += 1
                            continue

                        sp_temp, count_temp = self._spin_product(
                            atoms, indices, indx_order, eq_sites, symm, dec)
                        sp += sp_temp
                        count += count_temp

                    if count > 0:
                        cf_temp = sp / count
                        cf[cf_name] = cf_temp

        # print("Number of CFs skipped because of symmetry: {}".format(
        #     num_excluded_symmetry))
        if return_type == 'dict':
            pass
        elif return_type == 'tuple':
            cf = list(cf.items())
        elif return_type == 'array':
            cf = np.array([cf[x] for x in self.setting.full_cluster_names],
                          dtype=float)
        return cf

    def get_cf_by_cluster_names(self, atoms, cluster_names,
                                return_type='dict'):
        """Calculate correlation functions of the specified clusters.

        Arguments
        =========
        atoms: Atoms object

        cluster_names: list
            names (str) of the clusters for which the correlation functions are
            calculated for the structure provided in atoms

        return_type: str
            -'dict' (default): returns a dictionary (e.g., {'name': cf_value})
            -'tuple': returns a list of tuples (e.g., [('name', cf_value)])
            -'array': NumPy array containing *only* the correlation function
                      values in the same order as the order provided in the
                      "cluster_names"
        """
        if isinstance(atoms, Atoms):
            atoms = self.check_and_convert_cell_size(atoms.copy())
        else:
            raise TypeError('atoms must be Atoms object')
        cf = {}

        for name in cluster_names:
            if name == 'c0':
                cf[name] = 1.
                continue
            prefix = name.rpartition('_')[0]
            dec = name.rpartition('_')[-1]
            dec_list = [int(i) for i in dec]
            # find c{num} in cluster type
            n = int(prefix[1])

            if n == 1:
                cf[name] = self.get_c1(atoms, int(dec))
                continue

            sp = 0.
            count = 0
            # loop through the symmetry inequivalent groups
            for symm in range(self.num_trans_symm):
                # find the type of cluster based on the index of the original
                # settings.cluster_names nested list (unflattened)
                try:
                    name_indx = self.setting.cluster_names[symm][n].index(
                        prefix)
                except ValueError:
                    continue
                indices = self.setting.cluster_indx[symm][n][name_indx]
                indx_order = self.setting.cluster_order[symm][n][name_indx]
                eq_sites = self.setting.cluster_eq_sites[symm][n][name_indx]

                sp_temp, count_temp = self._spin_product(
                    atoms, indices, indx_order, eq_sites, symm, dec_list)
                sp += sp_temp
                count += count_temp
            cf_temp = sp / count
            cf['{}_{}'.format(prefix, dec)] = cf_temp

        if return_type == 'dict':
            pass
        elif return_type == 'tuple':
            cf = list(cf.items())
        elif return_type == 'array':
            cf = np.array([cf[x] for x in cluster_names], dtype=float)
        return cf

    def reconfig_db_entries(self, select_cond=None, reset=True):
        """Reconfigure the correlation function values of the entries in DB.

        Arguments
        =========
        select_cond: list
            -None (default): select every item in DB except for 'information'
            -else: select based on additional condictions provided

        reset: bool
            -True: removes all the key-value pairs that describe correlation
                   functions.
            -False: leaves the existing correlation functions in the key-Value
                    pairs, but overwrites the ones that exists in the current
                    setting.
        """
        db = connect(self.setting.db_name)
        select = [('name', '!=', 'information')]
        if select_cond is not None:
            for cond in select_cond:
                select.append(cond)

        # get how many entries need to be reconfigured
        row_ids = [row.id for row in db.select(select)]
        num_reconf = len(row_ids)
        print('{} entries will be reconfigured'.format(num_reconf))

        count = 0
        for row_id in row_ids:
            row = db.get(id=row_id)
            kvp = row.key_value_pairs

            # delete existing CF values
            if reset:
                keys = []
                for key in kvp.keys():
                    if key.startswith(('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
                                       'c6', 'c7', 'c8', 'c9')):
                        keys.append(key)
                db.update(row_id, delete_keys=keys)

            # get new CF based on setting
            atoms = wrap_and_sort_by_position(row.toatoms())
            cf = self.get_cf(atoms, return_type='dict')
            db.update(row_id, **cf)
            count += 1
            print('updated {} of {} entries'.format(count, num_reconf))

    def _spin_product(self, atoms, indx_list, indx_order, eq_sites, symm_group,
                      deco):
        """Get spin product of a given cluster.

        Arguments
        =========
        atoms: Atoms object

        indx_list: list
            A nested list where indices of the atoms that consistute a cluster
            are grouped together.

        indx_order: list
            A nested list of how the indices in "indx_list" should be ordered.
            The indices of atoms are sorted in a decrease order of internal
            distances to other members of the cluster.

        eq_sites: list
            A nested list that groups the equivalent atoms in a cluster. Atoms
            are classified as equivalent when they are inditinguishable based
            on the geometry of the cluster
            (e.g., equilateral triangles have 3 indistinguishable points.)

        symm_group: int
            The number that indicates which translational symmetry group that
            the reference atom in the cluster belongs to.

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        sp = 0.
        count = 0

        # spin product of each atom in the symmetry equivalent group
        indices_of_symm_group = self.index_by_trans_symm[symm_group]
        ref_indx_grp = indices_of_symm_group[0]
        for ref_indx in indices_of_symm_group:
            sp_temp, count_temp = \
                self._sp_same_shape_deco_for_ref_indx(atoms, ref_indx,
                                                      indx_list, indx_order,
                                                      eq_sites, ref_indx_grp,
                                                      deco)
            sp += sp_temp
            count += count_temp
        return sp, count

    def _sp_same_shape_deco_for_ref_indx(self, atoms, ref_indx, indx_list,
                                         indx_order, eq_sites, ref_indx_grp,
                                         deco):
        """Compute sp of cluster with same shape and deco for given ref atom.

        Arguments
        =========
        atoms: Atoms object

        ref_indx: int
            Index of the atom used as a reference to get clusters.

        indx_list: list
            A nested list where indices of the atoms that consistute a cluster
            are grouped together.

        indx_order: list
            A nested list of how the indices in "indx_list" should be ordered.
            The indices of atoms are sorted in a decrease order of internal
            distances to other members of the cluster.

        eq_sites: list
            A nested list that groups the equivalent atoms in a cluster. Atoms
            are classified as equivalent when they are inditinguishable based
            on the geometry of the cluster
            (e.g., equilateral triangles have 3 indistinguishable points.)

        ref_indx_grp: int
            Index of the reference atom used for the translational symmetry
            group.

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        count = 0
        sp = 0.0
        for cluster_indices, order in zip(indx_list, indx_order):
            temp_sp, temp_cnt = \
                self._spin_product_one_cluster(atoms, ref_indx,
                                               cluster_indices, order,
                                               eq_sites, ref_indx_grp, deco)
            sp += temp_sp
            count += temp_cnt
        return sp, count

    def _spin_product_one_cluster(self, atoms, ref_indx, cluster_indices,
                                  order, eq_sites, ref_indx_grp, deco):
        """Compute spin product for one cluster (same shape, deco, ref_indx).

        Arguments
        =========
        atoms: Atoms object

        ref_indx: int
            Index of the atom used as a reference to get clusters.

        cluster_indices: list
            A list where indices of the atoms that consistute a cluster.

        order: list
            A list of how the indices in "cluster_indices" should be ordered.
            The indices of atoms are sorted in a decrease order of internal
            distances to other members of the cluster.

        eq_sites: list
            A list that groups the equivalent atoms in a cluster. Atoms are
            classified as equivalent when they are inditinguishable based on
            the geometry of the cluster.
            (e.g., equilateral triangles have 3 indistinguishable points.)

        ref_indx_grp: int
            Index of the reference atom used for the translational symmetry
            group.

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        bf = self.setting.basis_functions
        count = 0
        sp = 0.0
        indices = [ref_indx_grp] + cluster_indices
        sorted_indices = [indices[indx] for indx in order]
        # Average over decoration numbers of equivalent sites
        equiv_deco = equivalent_deco(deco, eq_sites)
        for dec in equiv_deco:
            sp_temp = 1.0
            # loop through indices of atoms in each cluster
            for i, indx in enumerate(sorted_indices):
                trans_indx = self.setting.trans_matrix[ref_indx][indx]
                sp_temp *= bf[dec[i]][atoms[trans_indx].symbol]
            sp += sp_temp
            count += 1
        return sp, count

    def check_and_convert_cell_size(self, atoms, return_ratio=False):
        """Check the size of provided cell and convert in necessary.

        If the size of the provided cell is the same as the size of the
        template stored in the database. If it either (1) has the same size or
        (2) can make the same size by simple multiplication (supercell), the
        cell with the same size is returned after it is sorted by the position
        and wrapped. If not, it raises an error.
        """
        cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
        try:
            row = connect(self.setting.db_name).get(name='information')
            template = row.toatoms()
        except BaseException:
            raise IOError("Cannot retrieve the information template from the "
                          "database")
        template_lengths = template.get_cell_lengths_and_angles()[:3]

        if np.allclose(cell_lengths, template_lengths):
            atoms = wrap_and_sort_by_position(atoms)
            int_ratios = np.array([1, 1, 1])
        else:
            ratios = template_lengths / cell_lengths
            int_ratios = ratios.round(decimals=0).astype(int)
            if np.allclose(ratios, int_ratios):
                atoms = wrap_and_sort_by_position(atoms * int_ratios)
            else:
                raise TypeError("Cannot make the passed atoms to the specified"
                                " size of {}".format(self.setting.size))
        if return_ratio:
            return atoms, int_ratios
        return atoms
