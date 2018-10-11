"""Module for calculating correlation functions."""
import numpy as np
from ase.atoms import Atoms
from ase.clease import CEBulk, CECrystal
from ase.clease.tools import wrap_and_sort_by_position, equivalent_deco
from ase.db import connect
import multiprocessing as mp

# workers can not be a member of CorrFunction since CorrFunctions is passed
# as argument to the map function. Hence, we leave it as a global variable,
# but it is initialized wheh the CorrFunction object is initialized.
workers = None


class CorrFunction(object):
    """Calculate the correlation function.

    Arguments
    =========
    setting: settings object

    parallel: bool
        specify whether or not to use the parallel processing for ``get_cf''
        method.

    num_core: int or "all"
        specify the number of cores to use for parallelization.

    """

    def __init__(self, setting, parallel=False, num_core="all"):
        if not isinstance(setting, (CEBulk, CECrystal)):
            raise TypeError("setting must be CEBulk or CECrystal "
                            "object")
        self.setting = setting

        self.index_by_trans_symm = setting.index_by_trans_symm
        self.num_trans_symm = setting.num_trans_symm
        self.ref_index_trans_symm = setting.ref_index_trans_symm
        self.parallel = parallel
        self.num_core = num_core
        if parallel:
            global workers
            if self.num_core == "all":
                num_proc = int(mp.cpu_count()/2)
            else:
                num_proc = int(self.num_core)
            if workers is None:
                workers = mp.Pool(num_proc)

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
        cf['c0'] = 1.0

        # Update singlets
        for dec in bf_list:
            cf['c1_{}'.format(dec)] = self.get_c1(atoms, dec)

        cnames = self.setting.cluster_names
        if self.parallel:
            args = [(self, atoms, name) for name in cnames]
            res = workers.map(get_cf_parallel, args)
            cf = {}
            for r in res:
                cf[r[0][0]] = r[0][1]
            if return_type == 'tuple':
                return list(cf.items())
            elif return_type == 'array':
                cf = np.array([cf[x] for x in cnames], dtype=float)
            return cf
        return self.get_cf_by_cluster_names(atoms, cnames, return_type)

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
            # for symm in range(self.num_trans_symm):
            for cluster_set in self.setting.cluster_info:
                if prefix not in cluster_set.keys():
                    continue
                cluster = cluster_set[prefix]
                sp_temp, count_temp = \
                    self._spin_product(atoms, cluster, dec_list)
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

    def _spin_product(self, atoms, cluster, deco):
        """Get spin product of a given cluster.

        Arguments
        =========
        atoms: Atoms object

        cluster: dict
            A dictionary containing all necessary information about the
            family of cluster (i.e., list of indices, order, equivalent sites,
            symmetry group, etc.).

        deco: tuple
            Decoration number that specifies which basis function should be
            used for getting the spin variable of each atom.
        """
        sp = 0.
        count = 0

        # spin product of each atom in the symmetry equivalent group
        indices_of_symm_group = self.index_by_trans_symm[cluster["symm_group"]]
        ref_indx_grp = indices_of_symm_group[0]

        for ref_indx in indices_of_symm_group:
            sp_temp, count_temp = self._sp_same_shape_deco_for_ref_indx(
                atoms, ref_indx, cluster, ref_indx_grp, deco)
            sp += sp_temp
            count += count_temp
        return sp, count

    def _sp_same_shape_deco_for_ref_indx(self, atoms, ref_indx, cluster,
                                         ref_indx_grp, deco):
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
        for cluster_indices, order in zip(cluster["indices"],
                                          cluster["order"]):
            temp_sp, temp_cnt = \
                self._spin_product_one_cluster(atoms, ref_indx,
                                               cluster_indices, order,
                                               cluster["equiv_sites"],
                                               ref_indx_grp, deco)
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
        indices = [ref_indx_grp] + list(cluster_indices)
        sorted_indices = [indices[indx] for indx in order]
        # Average over decoration numbers of equivalent sites
        eq_sites = list(eq_sites)
        equiv_deco = equivalent_deco(deco, eq_sites)
        for dec in equiv_deco:
            sp_temp = 1.0
            # loop through indices of atoms in each cluster
            for i, indx in enumerate(sorted_indices):
                trans_indx = self.setting.trans_matrix[ref_indx][indx]
                sp_temp *= bf[dec[i]][atoms[trans_indx].symbol]
            sp += sp_temp
            count += 1
        num_equiv = float(len(equiv_deco))
        return sp/num_equiv, count/num_equiv

    def check_and_convert_cell_size(self, atoms):
        """Check the size of provided cell and convert in necessary.

        If the size of the provided cell is the same as the size of the
        template stored in the database. If it either (1) has the same size or
        (2) can make the same size by simple multiplication (supercell), the
        cell with the same size is returned after it is sorted by the position
        and wrapped. If not, it raises an error.
        """
        cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
        db = connect(self.setting.db_name)
        found_matching_template = False
        for row in db.select(name='template'):
            template = row.toatoms()
            template_lengths = template.get_cell_lengths_and_angles()[:3]

            if np.allclose(cell_lengths, template_lengths):
                atoms = wrap_and_sort_by_position(atoms)
                found_matching_template = True
                break
            #
            ratios = template_lengths / cell_lengths
            int_ratios = ratios.round(decimals=0).astype(int)
            if np.allclose(ratios, int_ratios):
                atoms = wrap_and_sort_by_position(atoms * int_ratios)
                found_matching_template = True
                break

        if not found_matching_template:
            raise TypeError("Cannot find the template atoms that matches the "
                            "size of the passed atoms")
        return atoms


def get_cf_parallel(args):
    cf = args[0]
    atoms = args[1]
    name = args[2]
    return cf.get_cf_by_cluster_names(atoms, [name], return_type="tuple")
