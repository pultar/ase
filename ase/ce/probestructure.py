import os
import math
from copy import deepcopy
from itertools import permutations
from random import choice, getrandbits
import numpy as np
from numpy.linalg import inv
from ase.db import connect
from ase.ce import BulkCrystal, BulkSpacegroup, CorrFunction
from ase.ce.tools import wrap_and_sort_by_position, reduce_matrix


class ProbeStructure(object):
    """Generate probe structures.

    Based on simulated annealing according to the recipe in
    PRB 80, 165122 (2009).

    Arguments:
    =========
    setting: BulkCrystal or BulkSapcegroup object

    atoms: Atoms object
        initial structure to start the simulated annealing

    struct_per_gen: int
        number of structures to be generated per generation

    init_temp: int or float
        initial temperature (does not represent *physical* temperature)

    final_temp: int or float
        final temperature (does not represent *physical* temperature)

    num_temp: int
        number of temperatures to be used in simulated annealing

    num_steps: int
        number of steps in simulated annealing

    approx_mean_var: bool
        whether or not to use a spherical and isotropical distribution
        approximation scheme for determining the mean variance.
        -'True': Assume a spherical and isotropical distribution of
                 structures in the configurational space.
                 Corresponds to eq.4 in PRB 80, 165122 (2009)
        -'False': Use sigma and mu of eq.3 in PRB 80, 165122 (2009)
                  to characterize the distribution of structures in
                  population.
                  Requires pre-sampling of random structures before
                  generating probe structures.
                  Reads sigma and mu from 'probe_structure-sigma_mu.npz' file.
    """

    def __init__(self, setting, atoms, struct_per_gen, init_temp=None,
                 final_temp=None, num_temp=5, num_steps=10000,
                 approx_mean_var=False):
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            raise TypeError("setting must be BulkCrystal or BulkSpacegroup "
                            "object")
        self.setting = setting
        self.trans_matrix = setting.trans_matrix

        self.cluster_names = self.setting.full_cluster_names
        self.corrFunc = CorrFunction(setting)
        self.cfm = self._get_full_cf_matrix()

        if self.setting.in_conc_matrix(atoms):
            self.init = wrap_and_sort_by_position(atoms)
        else:
            raise ValueError("concentration of the elements in the provided"
                             " atoms cannot be found in the conc_matrix")

        self.approx_mean_var = approx_mean_var
        fname = 'probe_structure-sigma_mu.npz'
        if not approx_mean_var:
            if os.path.isfile(fname):
                data = np.load(fname)
                self.sigma = data['sigma']
                self.mu = data['mu']
            else:
                raise IOError("'{}' not found.".format(fname))

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
        """Generate a probe structure according to PRB 80, 165122 (2009)."""
        # Start
        old = self.init.copy()
        o_cf = self.corrFunc.get_cf_by_cluster_names(old, self.cluster_names,
                                                     return_type='array')
        o_cfm = np.vstack((self.cfm, o_cf))
        if self.approx_mean_var:
            o_mv = mean_variance_approx(o_cfm)
        else:
            o_mv = mean_variance(o_cfm, self.sigma, self.mu)

        temps = np.logspace(math.log10(self.init_temp),
                            math.log10(self.final_temp),
                            self.num_temp)
        steps_per_temp = int(self.num_steps / self.num_temp)
        count = 0
        for temp in temps:
            for _ in range(steps_per_temp):
                if bool(getrandbits(1)):
                    if self._has_more_than_one_conc():
                        new, n_cf = self._change_element_type(old, o_cf)
                    else:
                        if self._is_swappable(old):
                            new, n_cf = self._swap_two_atoms(old, o_cf)
                        else:
                            msg = 'Atoms has only one concentration value '
                            msg += 'and not swappable.'
                            raise RuntimeError(msg)
                else:
                    if self._is_swappable(old):
                        new, n_cf = self._swap_two_atoms(old, o_cf)
                    else:
                        new, n_cf = self._change_element_type(old, o_cf)
                n_cfm = np.vstack((self.cfm, n_cf))
                if self.approx_mean_var:
                    n_mv = mean_variance_approx(n_cfm)
                else:
                    n_mv = mean_variance(n_cfm, self.sigma, self.mu)
                accept = np.exp((o_mv - n_mv) / temp) > np.random.uniform()
                count += 1
                # print(count, accept)
                if accept:
                    old = new.copy()
                    o_cf = np.copy(n_cf)
                    o_mv = n_mv

        # Check to see if the cf is indeed preserved
        final_cf = self.corrFunc.get_cf(old, return_type='array')
        if not np.allclose(final_cf, o_cf):
            msg = 'The correlation function changed after simulated annealing'
            raise ValueError(msg)
        return old, o_cf

    def _determine_temps(self):
        print("Temperature range not given. "
              "Determining the range automatically.")
        old = self.init.copy()
        o_cf = self.corrFunc.get_cf_by_cluster_names(old, self.cluster_names,
                                                     return_type='array')
        o_cfm = np.vstack((self.cfm, o_cf))
        if self.approx_mean_var:
            o_mv = mean_variance_approx(o_cfm)
        else:
            o_mv = mean_variance(o_cfm, self.sigma, self.mu)
        diffs = []
        for _ in range(100):
            if bool(getrandbits(1)):
                # Change element Type
                if self._has_more_than_one_conc():
                    new, n_cf = self._change_element_type(old, o_cf)
                else:
                    if self._is_swappable(old):
                        new, n_cf = self._swap_two_atoms(old, o_cf)
                    else:
                        raise RuntimeError('Atoms has only one concentration '
                                           + 'value and not swappable.')
            else:
                # Swap two atoms
                if self._is_swappable(old):
                    new, n_cf = self._swap_two_atoms(old, o_cf)
                else:
                    new, n_cf = self._change_element_type(old, o_cf)
            n_cfm = np.vstack((self.cfm, n_cf))
            if self.approx_mean_var:
                n_mv = mean_variance_approx(n_cfm)
            else:
                n_mv = mean_variance(n_cfm, self.sigma, self.mu)
            diffs.append(abs(o_mv - n_mv))
            # update old
            old = new.copy()
            o_cf = np.copy(n_cf)
            o_mv = n_mv

        avg_diff = sum(diffs) / len(diffs)
        init_temp = 10 * avg_diff
        final_temp = 0.01 * avg_diff
        print('init_temp= {}, final_temp= {}'.format(init_temp, final_temp))
        return init_temp, final_temp

    def _swap_two_atoms(self, atoms, cf):
        """Swap two randomly chosen atoms."""
        atoms = atoms.copy()
        cf = deepcopy(cf)
        indx = np.zeros(2, dtype=int)
        symbol = [None] * 2

        # determine if the basis is grouped
        if self.setting.grouped_basis is None:
            basis_elements = self.setting.basis_elements
            num_basis = self.setting.num_basis
            index_by_basis = self.setting.index_by_basis
        else:
            basis_elements = self.setting.grouped_basis_elements
            num_basis = self.setting.num_grouped_basis
            index_by_basis = self.setting.index_by_grouped_basis

        # pick fist atom and determine its symbol and type
        while True:
            basis = choice(range(num_basis))
            # a basis with only 1 type of element should not be chosen
            if len(basis_elements[basis]) < 2:
                continue
            indx[0] = choice(index_by_basis[basis])
            symbol[0] = atoms[indx[0]].symbol
            break
        # pick second atom that is not the same element, but occupies the
        # same site.
        while True:
            indx[1] = choice(index_by_basis[basis])
            symbol[1] = atoms[indx[1]].symbol
            if symbol[1] == symbol[0]:
                continue
            if symbol[1] in basis_elements[basis]:
                break
        # swap two atoms
        atoms, cf = self._change_element_type(atoms, cf, indx[0], symbol[1])
        atoms, cf = self._change_element_type(atoms, cf, indx[1], symbol[0])
        return atoms, cf

    def _has_more_than_one_conc(self):
        if len(self.setting.conc_matrix) > 1 and \
           self.setting.conc_matrix.ndim > 1:
            return True
        return False

    def _is_swappable(self, atoms):
        # determine if the basis is grouped
        if self.setting.grouped_basis is None:
            basis_elements = self.setting.basis_elements
            num_basis = self.setting.num_basis
        else:
            basis_elements = self.setting.grouped_basis_elements
            num_basis = self.setting.num_grouped_basis

        for i in range(num_basis - 1, -1, -1):
            # delete basis with only one element type
            if len(basis_elements[i]) < 2:
                num_basis -= 1
                continue

            # delete basis if atoms object has only one element type
            existing_elements = 0
            for element in basis_elements[i]:
                num = len([a.index for a in atoms if a.symbol == element])
                if num > 0:
                    existing_elements += 1
            if existing_elements < 2:
                num_basis -= 1

        if num_basis > 0:
            return True
        return False

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
        """Track the changes of correlation function."""
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

        # scan through each cluster name
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

            # Find which symmetry group the given atom (index) belongs to
            for symm in range(self.setting.num_trans_symm):
                if index in self.setting.index_by_trans_symm[symm]:
                    sg = symm

            # set name_indx and indices that compose a cluster
            try:
                name_indx = self.setting.cluster_names[sg][n].index(prefix)
            # ValueError means that the cluster name (prefix) was not
            # found in the symmetry group of the index --> this cluster is
            # not altered.
            except ValueError:
                continue
            indices = self.setting.cluster_indx[sg][n][name_indx]

            # Get the total count
            count = 0
            for symm in range(self.setting.num_trans_symm):
                try:
                    nindx = self.setting.cluster_names[symm][n].index(prefix)
                except ValueError:
                    continue

                clusters_per_atom = len(self.setting.cluster_indx[symm][n][nindx])
                atoms_per_symm = len(self.setting.index_by_trans_symm[symm])
                count += clusters_per_atom * atoms_per_symm

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
                    sg = None
                    # only count correlation function of the clusters that
                    # contain the changed atom
                    for symm in range(self.setting.num_trans_symm):
                        if nindx in self.setting.index_by_trans_symm[symm]:
                            sg = symm
                            break

                    name_indx = self.setting.cluster_names[sg][n].index(prefix)
                    indices = self.setting.cluster_indx[sg][n][name_indx]
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
        """Calculate spin product of the cluster that starts with ref_indx."""
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

    def _get_full_cf_matrix(self):
        """Get correlation function of every entry in DB."""
        cfm = []
        db = connect(self.setting.db_name)
        for row in db.select([('name', '!=', 'information')]):
            cfm.append([row[x] for x in self.cluster_names])
        cfm = np.array(cfm, dtype=float)
        return cfm

def mean_variance_full(cfm):
    prec = precision_matrix(cfm)
    mv = 0.
    for x in range(cfm.shape[0]):
        mv += cfm[x].dot(prec).dot(cfm[x].T)
    mv = mv / cfm.shape[0]
    return mv

def mean_variance(cfm, sigma, mu):
    prec = precision_matrix(cfm)
    return np.trace(prec.dot(sigma)) + mu.dot(prec).dot(mu.T)

def mean_variance_approx(cfm):
    prec = precision_matrix(cfm)
    return np.trace(prec)

def precision_matrix(cfm):
    try:
        prec = inv(cfm.T.dot(cfm))
    # if inverting matrix leads to a singular matrix, reduce the matrix
    except np.linalg.linalg.LinAlgError:
        cfm = reduce_matrix(cfm)
        prec = inv(cfm.T.dot(cfm))
    return prec
