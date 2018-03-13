import math
from copy import deepcopy
from itertools import permutations
from random import choice, getrandbits
import numpy as np
from numpy.linalg import inv
from ase.ce import BulkCrystal, BulkSpacegroup, CorrFunction, Evaluate
from ase.ce.tools import wrap_and_sort_by_position


class ProbeStructure(object):
    """Generate probe structures based on simulated annealing according to the
    recipe in PRB 80, 165122 (2009)."""
    def __init__(self, setting, atoms, struct_per_gen, init_temp=None,
                 final_temp=None, num_temp=5, num_steps=10000):
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            raise TypeError("setting must be BulkCrystal or BulkSpacegroup "
                            "object")
        self.setting = setting
        self.trans_matrix = setting.trans_matrix

        self.cluster_names = self.setting.full_cluster_names
        self.corrFunc = CorrFunction(setting)
        self.cfm = Evaluate(setting, self.cluster_names).full_cf_matrix

        if self.setting.in_conc_matrix(atoms):
            self.init = wrap_and_sort_by_position(atoms)
        else:
            raise ValueError("concentration of the elements in the provided"
                             " atoms cannot be found in the conc_matrix")

        if init_temp is None or final_temp is None:
            self.init_temp, self.final_temp = self._determine_temps()
        else:
            self.init_temp = init_temp
            self.final_temp = final_temp
        self.num_temp = num_temp
        self.num_steps = num_steps

        if self.init_temp <= self.final_temp:
            raise ValueError("Initial temperature must be higher than final"
                             " temperature")

    def generate(self):
        """Generate a probe structure according to PRB 80, 165122 (2009)
        """
        # Start
        old = self.init.copy()
        o_cf = self.corrFunc.get_cf_by_cluster_names(old, self.cluster_names,
                                                     return_type='array')
        o_cfm = np.vstack((self.cfm, o_cf))
        o_mv = self._get_mean_variance(o_cfm)

        temps = np.logspace(math.log10(self.init_temp),
                            math.log10(self.final_temp),
                            self.num_temp)
        steps_per_temp = int(self.num_steps / self.num_temp)
        count = 0
        for temp in temps:
            for _ in range(steps_per_temp):
                if bool(getrandbits(1)):
                    new, n_cf = self._change_element_type(old, o_cf)
                else:
                    new, n_cf = self._swap_two_atoms(old, o_cf)
                n_cfm = np.vstack((self.cfm, n_cf))
                n_mv = self._get_mean_variance(n_cfm)
                accept = np.exp((o_mv - n_mv) / temp) > np.random.uniform()
                count += 1
                # print(count, accept, o_mv - n_mv, temp)
                if accept:
                    old = new.copy()
                    o_cf = np.copy(n_cf)
                    o_mv = n_mv

        # Check to see if the cf is indeed preserved
        final_cf = self.corrFunc.get_cf(old, return_type='array')
        if not np.allclose(final_cf, o_cf):
            raise ValueError("The correlation function changed after simulated"
                             " annealing")
        return old, o_cf

    def _determine_temps(self):
        print("Temperature range not given. "
              "Determining the range automatically.")
        old = self.init.copy()
        o_cf = self.corrFunc.get_cf_by_cluster_names(old, self.cluster_names,
                                                     return_type='array')
        o_cfm = np.vstack((self.cfm, o_cf))
        o_mv = self._get_mean_variance(o_cfm)
        diffs = []
        for _ in range(50):
            if bool(getrandbits(1)):
                new, n_cf = self._change_element_type(old, o_cf)
            else:
                new, n_cf = self._swap_two_atoms(old, o_cf)
            n_cfm = np.vstack((self.cfm, n_cf))
            n_mv = self._get_mean_variance(n_cfm)
            diffs.append(abs(o_mv - n_mv))
            # update old
            old = new.copy()
            o_cf = np.copy(n_cf)
            o_mv = n_mv

        avg_diff = sum(diffs) / len(diffs)
        init_temp = 10 * avg_diff
        final_temp = 0.1 * avg_diff
        print('init_temp= {}, final_temp= {}'.format(init_temp, final_temp))
        return init_temp, final_temp

    def _swap_two_atoms(self, atoms, cf):
        """
        Swaps two randomly chosen atoms.
        """
        atoms = atoms.copy()
        cf = deepcopy(cf)
        natoms = len(atoms)
        indx = np.zeros(2, dtype=int)
        symbol = [None] * 2
        # pick fist atom and determine its symbol and type
        indx[0] = choice(range(natoms))
        symbol[0] = atoms[indx[0]].symbol
        for site in range(self.setting.num_basis):
            if symbol[0] in self.setting.basis_elements[site]:
                break
        # pick second atom that is not the same element, but occupies the
        # same site.
        while True:
            indx[1] = choice(range(natoms))
            symbol[1] = atoms[indx[1]].symbol
            if symbol[1] == symbol[0]:
                continue
            if symbol[1] in self.setting.basis_elements[site]:
                break
        # swap two atoms
        atoms, cf = self._change_element_type(atoms, cf, indx[0], symbol[1])
        atoms, cf = self._change_element_type(atoms, cf, indx[1], symbol[0])
        return atoms, cf

    def _change_element_type(self, atoms, cf, index=None, rplc_element=None):
        """Change the type of element for the atom with a given index.
        If index and replacing element types are not specified, they are
        randomly generated.
        """
        old = atoms.copy()
        new = atoms.copy()
        natoms = len(new)
        # ------------------------------------------------------
        # Change the type of element for a given index if given.
        # If index not given, pick a random index
        # ------------------------------------------------------
        while True:
            # pick an atom and determine its symbol
            if index is None:
                indx = choice(range(natoms))
            else:
                indx = index
            old_symbol = new[indx].symbol
            # determine its basis
            for site in range(self.setting.num_basis):
                if old_symbol in self.setting.basis_elements[site]:
                    break
            # change element type
            if rplc_element is None:
                new_symbol = choice(self.setting.basis_elements[site])
                if new_symbol != old_symbol:
                    new[indx].symbol = new_symbol
                    if self.setting.in_conc_matrix(new):
                        break
                    new[indx].symbol = old_symbol
            else:
                new_symbol = rplc_element
                new[indx].symbol = rplc_element
                break

        return self._track_cf(old, new, cf, indx)

    def _track_cf(self, old, new, cf, index):
        """Track the changes of correlation function"""
        # change size of the cell if needed (e.g., max_cluster_dia > min(lat))
        # also, get the size ratios (multiplicaion factor for making supercell)
        old_sc = self.corrFunc.check_and_convert_cell_size(old.copy())
        new_sc, ratios = \
            self.corrFunc.check_and_convert_cell_size(new.copy(), True)
        scale = np.prod(ratios)
        cf = np.copy(cf)

        # need to find the corresponding index in the supercell
        pos = new[index].position
        for i, atom in enumerate(new_sc):
            if np.allclose(pos, atom.position):
                index = i
                break

        for i, name in enumerate(self.cluster_names):
            n = int(name[1])
            if n == 0:
                continue
            # Find the type of cluster and its decoration numbers
            prefix = name.rpartition('_')[0]
            dec_str = name.rpartition('_')[-1]
            dec = [int(x) for x in dec_str]
            if n == 1:
                cf[i] = self.corrFunc.get_c1(new_sc, int(dec_str))
                continue

            # Get the total count
            count = 0
            for symm in range(self.setting.num_trans_symm):
                name_indx = self.setting.cluster_names[symm][n].index(prefix)
                indices = self.setting.cluster_indx[symm][n][name_indx]
                clusters_per_atom = len(indices)
                atoms_per_symm = len(self.setting.index_by_trans_symm[symm])
                count += clusters_per_atom * atoms_per_symm
                # Find which symmetry group the given atom (index) belongs to
                if index in self.setting.index_by_trans_symm[symm]:
                    symm_group = symm

            # set name_indx and indices that compose a cluster
            name_indx = self.setting.cluster_names[symm_group][n].index(prefix)
            indices = self.setting.cluster_indx[symm_group][n][name_indx]

            t_indices = self._translate_indx(index, indices)
            cf_tot = cf[i] * count
            cf_old = self._cf_by_indx(old_sc, index, t_indices, dec)
            cf_new = self._cf_by_indx(new_sc, index, t_indices, dec)

            # if there is only one symm equiv site, the changes can be just
            # multiplied by *n*
            if self.setting.num_trans_symm == 1:
                sp = cf_tot + scale * n * (cf_new - cf_old)
                cf[i] = sp / count
            else:
                sp = cf_tot + scale * (cf_new - cf_old)
                # find the members of cluster
                members = np.unique(t_indices)

                for nindx in members:
                    # only count correlation function of the clusters that
                    # contain the changed atom
                    for symm in range(self.setting.num_trans_symm):
                        if nindx in self.setting.index_by_trans_symm[symm]:
                            symm_group = symm
                            break
                    indices = self.setting.cluster_indx[symm][n][name_indx]
                    # tl = self._translate_indx(nindx, indices)
                    # trans_list = tl[~np.all(tl != indx, axis=1)]
                    t_indices = []
                    for item in self._translate_indx(nindx, indices):
                        if index in item:
                            t_indices.append(item)

                    cf_old = self._cf_by_indx(old_sc, nindx, t_indices, dec)
                    cf_new = self._cf_by_indx(new_sc, nindx, t_indices, dec)
                    sp += scale * (cf_new - cf_old)
                cf[i] = sp / count

        return new, cf

    def _translate_indx(self, ref_indx, indx_list):
        tlist = deepcopy(indx_list)
        for i in range(len(indx_list)):
            for j in range(len(indx_list[i])):
                tlist[i][j] = self.trans_matrix[ref_indx, indx_list[i][j]]
        return tlist

    def _cf_by_indx(self, atoms, ref_indx, trans_indices, deco):
        """Calculate the spin product of the cluster that starts with the
        ref_indx.
        """
        bf = self.setting.basis_functions
        sp = 0.
        perm = list(permutations(deco, len(deco)))
        perm = np.unique(perm, axis=0)
        for dec in perm:
            for indices in trans_indices:
                sp_temp = bf[dec[0]][atoms[ref_indx].symbol]
                for i, indx in enumerate(indices):
                    sp_temp *= bf[dec[i + 1]][atoms[indx].symbol]
                sp += sp_temp
        sp /= len(perm)
        return sp

    def _get_mean_variance_full(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        mv = 0.
        for x in range(cfm.shape[0]):
            mv += cfm[x].dot(prec).dot(cfm[x].T)
        mv = mv / cfm.shape[0]
        return mv

    def _get_mean_variance(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        sigma = np.cov(cfm.T)
        mu = np.mean(cfm, axis=0)
        mv = np.trace(prec.dot(sigma)) + mu.dot(prec).dot(mu.T)
        return mv
