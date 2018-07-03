"""Module for calculating correlation functions."""
from itertools import combinations_with_replacement, permutations
import numpy as np
from ase.atoms import Atoms
from ase.ce import BulkCrystal, BulkSpacegroup
from ase.ce.tools import wrap_and_sort_by_position
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
        # ----------------------------------------------------
        # Compute correlation function up the max_cluster_size
        # ----------------------------------------------------
        # loop though all cluster sizes
        for n in range(self.setting.max_cluster_size + 1):
            comb = list(combinations_with_replacement(bf_list, r=n))
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
                            name_indx = name_list.index(unique_name)
                        except ValueError:
                            continue

                        indices = self.setting.cluster_indx[symm][n][name_indx]

                        sp_temp, count_temp = \
                            self._spin_product(atoms, indices, symm, dec)
                        sp += sp_temp
                        count += count_temp

                    cf_temp = sp / count
                    # make decoration number into string
                    dec_string = ''.join(str(i) for i in dec)
                    cf['{}_{}'.format(unique_name, dec_string)] = cf_temp

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
        # natoms = len(atoms)
        # bf_list = list(range(len(self.setting.basis_functions)))
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
                    name_indx = self.setting.cluster_names[symm][n].index(prefix)
                except ValueError:
                    continue
                indices = self.setting.cluster_indx[symm][n][name_indx]
                sp_temp, count_temp = \
                    self._spin_product(atoms, indices, symm, dec_list)
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
                for key, value in kvp.items():
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

    def _spin_product(self, atoms, indx_list, symm_group, deco):
        bf = self.setting.basis_functions
        sp = 0.
        count = 0
        # spin product of each atom in the symmetry equivalent group
        indices_of_symm_group = self.index_by_trans_symm[symm_group]

        perm = list(permutations(deco, len(deco)))
        perm = np.unique(perm, axis=0)

        # loop through each permutation of decoration numbers
        for dec in perm:
            # loop through each symmetry equivalent atom
            for ref_indx in indices_of_symm_group:
                ref_spin = bf[dec[0]][atoms[ref_indx].symbol]
                # loop through each cluster
                for cluster_indices in indx_list:
                    sp_temp = ref_spin
                    # loop through indices of atoms in each cluster
                    for i, indx in enumerate(cluster_indices):
                        trans_indx = self.setting.trans_matrix[ref_indx][indx]
                        sp_temp *= bf[dec[i + 1]][atoms[trans_indx].symbol]
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
        except:
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
