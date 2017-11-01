import numpy as np
import math
from ase.ce.settings import BulkCrystal
from ase.ce.corrFunc import CorrFunction
from ase.ce.evaluate import Evaluate
from ase.ce.tools import reduce_matrix
from ase.units import kB
from random import choice, getrandbits
from numpy.linalg import inv
from copy import deepcopy
from itertools import product

class SimulatedAnnealing(object):
    """
    Class for Simulated Annealing.
    Used for (1) Generating probe structures (generate_probe_structure)
             (2) Finding Emin structure of for a given concentration
                 (generate_Emin_structure).
    """
    def __init__(self, BC, init_atoms, struct_per_gen,
                 cluster_name_eci_tuple=None, init_temp=None, final_temp=None,
                 num_temp=5, num_steps=10000):
        if type(BC) is not BulkCrystal:
            raise TypeError("Passed object should be BulkCrystal type")
        self.BC = BC
        self.num_sites = BC.num_sites
        self.site_elements = BC.site_elements
        self.all_elements = BC.all_elements
        self.num_elements = BC.num_elements
        self.orig_names = BC.cluster_names
        self.spin_dict = BC.spin_dict
        self.basis_functions = BC.basis_functions
        self.conc_matrix = BC.conc_matrix
        self.trans_matrix = BC.trans_matrix
        self.db_name = BC.db_name

        if cluster_name_eci_tuple is None:
            self.cluster_names = self._get_full_cluster_names()
            self.eci = None
        else:
            self.cluster_names = [tup[0] for tup in cluster_name_eci_tuple]
            self.eci = [tup[1] for tup in cluster_name_eci_tuple]

        self.cluster_dist = BC.cluster_dist
        self.cluster_indx = BC.cluster_indx
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.num_temp = num_temp
        self.num_steps = num_steps
        self.corrFunc = CorrFunction(BC)

        # sanity checks
        if self._in_conc_matrix(init_atoms):
            self.init = self.corrFunc.check_and_convert_cell_size(init_atoms)
        else:
            raise ValueError("concentration of the elements in the provided"
                             " atoms cannot be found in the conc_matrix")
        if init_temp is None or final_temp is None:
            raise ValueError("Initial and final temperatures must be"
                             " specified for simulated annealing")
        if init_temp <= final_temp:
            raise ValueError("Initial temperature must be higher than final"
                             " temperature")

    def generate_probe_structure(self):
        """
        Generate a probe structure according to PRB 80, 165122 (2009)
        """
        evaluate = Evaluate(self.BC, self.cluster_names)
        cfm = evaluate.full_cf_matrix
        # Start
        atoms = self.init.copy()
        cf1d = self.corrFunc.get_cf_by_cluster_names(atoms,
                                                     self.cluster_names)
        cf1 = np.array([cf1d[x] for x in self.cluster_names], dtype=float)
        cfm1 = np.vstack((cfm, cf1))
        cfm1 = reduce_matrix(cfm1)
        mv1 = self._get_mean_variance_Seko(cfm1)

        kTs = kB *np.logspace(math.log10(self.init_temp),
                              math.log10(self.final_temp),
                              self.num_temp)
        steps_per_temp = self.num_steps/self.num_temp
        for kT in kTs:
            for _ in range(steps_per_temp):
                if bool(getrandbits(1)):
                    atoms2, cf2d = self._change_element_type(atoms, cf1d)
                else:
                    atoms2, cf2d = self._swap_two_atoms(atoms, cf1d)
                cf2 = np.array([cf2d[x] for x in self.cluster_names], dtype=float)
                cfm2 = np.vstack((cfm, cf2))
                cfm2 = reduce_matrix(cfm2)
                mv2 = self._get_mean_variance_Seko(cfm2)
                accept = np.exp((mv1 - mv2) / kT) > np.random.uniform()
                if accept:
                    atoms = atoms2.copy()
                    cf1 = np.copy(cf2)
                    cf1d = deepcopy(cf2d)
                    mv1 = deepcopy(mv2)

        # Check to see if the cf is indeed preserved
        cfd_new =self.corrFunc.get_cf(atoms)
        if not self._two_dicts_equal(cf1d, cfd_new):
            raise ValueError("The correlation function changed after simulated"
                             " annealing")
        return atoms, cf1d

    def generate_Emin_structure(self):
        if self.eci is None:
            raise ValueError("ECI values need to be provided to generate Emin"
                             " structures")
        atoms = self.init.copy()
        cfd1 = self.corrFunc.get_cf_by_cluster_names(atoms,
                                                     self.cluster_names)
        cf1 = np.array([cfd1[x] for x in self.cluster_names], dtype=float)
        e1 = cf1.dot(self.eci)

        kTs = kB *np.logspace(math.log10(self.init_temp),
                              math.log10(self.final_temp),
                              self.num_temp)
        steps_per_temp = self.num_steps/self.num_temp

        for kT in kTs:
            for _ in range(steps_per_temp):
                atoms2, cfd2 = self._swap_two_atoms(atoms, cfd1)
                cf2 = np.array([cfd2[x] for x in self.cluster_names],
                               dtype=float)
                e2 = cf2.dot(self.eci)
                accept = np.exp((e1 - e2)/kT) > np.random.uniform()
                if accept:
                    atoms = atoms2.copy()
                    cf1 = np.copy(cf2)
                    cfd1 = deepcopy(cfd2)
                    e1 = deepcopy(e2)
        #Check to see if the cf is indeed preserved
        cfd_new =self.corrFunc.get_cf_by_cluster_names(atoms,
                                                       self.cluster_names)
        if not self._two_dicts_equal(cfd1, cfd_new):
            raise ValueError("The correlation function changed after"
                             " simulated annealing")
        cfd1['Epred'] = e1
        return atoms, cfd1

    def _get_full_cluster_names(self):
        """
        Returns the all possible cluster names.
        """
        full_names = []
        full_names.append(self.orig_names[0][0])
        for k in range(1,len(self.orig_names)):
            cases = (self.num_elements-1)**k
            for name in self.orig_names[k][:]:
                for i in range(1,cases+1):
                    full_names.append('{}_{}'.format(name,i))
        return full_names

    def _in_conc_matrix(self, atoms):
        """
        Checks to see if the passed atoms object has allowed concentration by
        checking the concentration matrix. Returns boolean.
        """
        # determine the concentration of the given atoms
        conc = np.zeros(self.num_elements, dtype=int)
        for x in range(self.num_elements):
            element = self.all_elements[x]
            num_element = len([a for a in atoms if a.symbol == element])
            conc[x] = num_element

        # determine the dimensions of the concentration matrix
        # then, search to see if there is a match
        conc_shape = self.conc_matrix.shape
        if len(conc_shape) == 2:
            for x in range(conc_shape[0]):
                if np.array_equal(conc, self.conc_matrix[x]):
                    return True
        else:
            for x in range(conc_shape[0]):
                for y in range(conc_shape[1]):
                    if np.array_equal(conc, self.conc_matrix[x][y]):
                        return True
        return False

    def _change_element_type(self, atoms, cf, index=None, rplc_element=None):
        """
        Changes the type of element for the atom with a given index.
        If index and replacing element types are not specified, they are
        randomly generated.
        """
        orig_atoms = atoms.copy()
        new_atoms = atoms.copy()
        cf = deepcopy(cf)
        natoms = len(new_atoms)
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
            old_symbol = new_atoms[indx].symbol
            # determine its site type
            for site in range(self.num_sites):
                if old_symbol in self.site_elements[site]:
                    break
            # change element type
            if rplc_element is None:
                new_symbol = choice(self.site_elements[site])
                if new_symbol != old_symbol:
                    new_atoms[indx].symbol = new_symbol
                    if self._in_conc_matrix(new_atoms):
                        break
                    new_atoms[indx].symbol = old_symbol
            else:
                new_symbol = rplc_element
                new_atoms[indx].symbol = rplc_element
                break

        # -----------------------------------------
        # Track the changes of correlation function
        # -----------------------------------------
        bf_list = list(range(len(self.basis_functions)))
        for name in self.cluster_names:
            if name == 'c0':
                continue
            elif name.startswith('c1'):
                dec = int(name[-1]) - 1
                cf[name] = self.corrFunc.get_c1(new_atoms, dec)
                continue
            # Find the type of cluster and its decoration numbers
            prefix = name.rpartition('_')[0]
            dec = int(name.rpartition('_')[-1]) - 1
            for n in range(2, len(self.orig_names)):
                try:
                    ctype = self.orig_names[n].index(prefix)
                    num = n
                    break
                except ValueError:
                    continue
            perm = list(product(bf_list, repeat=num))
            indx_list = self.cluster_indx[num][ctype]
            trans_list = self._translate_indx(indx_list, indx)
            count = len(indx_list) * natoms
            sp_tot = cf[name] * count
            sp_old = self._sp_by_indx(orig_atoms, trans_list, indx, perm[dec])
            sp_new = self._sp_by_indx(new_atoms, trans_list, indx, perm[dec])

            # If the decoration number of all the elements in the cluster is
            # the same, it suffices to change only the correlation function of
            # the given index. If not, it is also necessary to change the
            # correlation function of its members' clusters which contain
            # the element changed.
            if len(set(perm[dec])) == 1:
                sp = sp_tot + (num)*(sp_new - sp_old)
            else:
                sp = sp_tot + sp_new - sp_old
                members = np.unique(trans_list)
                for nindx in members:
                    # only count correlation function of the clusters that
                    # contain the changed atom
                    tl = self._translate_indx(indx_list, nindx)
                    trans_list = tl[~np.all(tl != indx, axis=1)]
                    sp_old = self._sp_by_indx(orig_atoms, trans_list, nindx,
                                              perm[dec])
                    sp_new = self._sp_by_indx(new_atoms, trans_list, nindx,
                                             perm[dec])
                    sp += (sp_new - sp_old)
            cf[name] = sp/count
        return new_atoms, cf

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
        for site in range(self.num_sites):
            if symbol[0] in self.site_elements[site]:
                break
        # pick second atom that is not the same element, but occupies the
        # same site.
        while True:
            indx[1] = choice(range(natoms))
            symbol[1] = atoms[indx[1]].symbol
            if symbol[1] == symbol[0]:
                continue
            if symbol[1] in self.site_elements[site]:
                break
        # swap two atoms
        atoms, cf = self._change_element_type(atoms, cf, indx[0], symbol[1])
        atoms, cf = self._change_element_type(atoms, cf, indx[1], symbol[0])
        return atoms, cf

    def _translate_indx(self, indx_list, ref_indx):
        tlist = deepcopy(indx_list)
        for i in range(len(indx_list)):
            for j in range(len(indx_list[i])):
                tlist[i][j] = self.trans_matrix[ref_indx, indx_list[i][j]]
        return tlist

    def _sp_by_indx(self, atoms, indx_list, ref_indx, dec):
        """
        Calculates the spin product of the cluster that starts with
        the ref_indx.
        """
        bf = self.basis_functions
        sp = 0.
        for i in range(len(indx_list)):
            sp_temp = bf[dec[0]][atoms[ref_indx].symbol]
            for j, indx in enumerate(indx_list[i][:]):
                sp_temp *= bf[dec[j+1]][atoms[indx].symbol]
            sp += sp_temp
        return sp

    def _get_mean_variance(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        mv = 0.
        for x in range(cfm.shape[0]):
            mv += cfm[x].dot(prec).dot(cfm[x].T)
        mv = mv / cfm.shape[0]
        return mv

    def _get_mean_variance_Seko(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        sigma = np.cov(cfm.T)
        mu = np.mean(cfm, axis=0)
        mv = np.trace(prec.dot(sigma)) + mu.dot(prec).dot(mu.T)
        return mv

    def _two_dicts_equal(self, dict1, dict2):
        keys1 = list(dict1.keys())
        keys2 = list(dict2.keys())
        if set(keys1) != set(keys2):
            print("Two dictionaries have different keys.")
            return False

        for key in keys1:
            if abs(dict1[key] - dict2[key]) > 1e-6:
                return False
        return True
