import numpy as np
import math
from ase.ce.settings import BulkCrystal
from ase.ce.corrFunc import CorrFunction
from ase.ce.evaluate import Evaluate
from ase.ce.tools import reduce_matrix
from ase.units import kB
from random import choice, getrandbits
from ase.visualize import view
from numpy.linalg import inv, cond
from copy import deepcopy



class SimulatedAnnealing(object):
    """
    look into basin hopping
    actions can be a swap_positions, swap_element
    """

    def __init__(self, BC, initial_atoms, struct_per_gen, cluster_names=None,
                 init_temp=None, final_temp=None, num_temp=5, num_steps=10000,
                 eci=None):
        self.num_sites = BC.num_sites
        self.site_elements = BC.site_elements
        self.all_elements = BC.all_elements
        self.num_elements = BC.num_elements
        self.spin_dict = BC.spin_dict
        self.conc_matrix = BC.conc_matrix
        self.trans_matrix = BC.trans_matrix
        if self.in_conc_matrix(initial_atoms):
            self.init = initial_atoms
        else:
            raise ValueError("concentration of the elements in the provided"
                             " atoms cannot be found in the conc_matrix")
        self.db_name = BC.db_name
        if cluster_names is None:
            # if no names are provided, just use flattened cluster_names
            self.provided_cnames = [i for sub in BC.cluster_names for i in sub]
        else:
            self.provided_cnames = cluster_names
        self.cluster_names = BC.cluster_names
        self.cluster_dist = BC.cluster_dist
        self.cluster_indx = BC.cluster_indx
        if init_temp is None or final_temp is None:
            raise ValueError("Initial and final temperatures must be specified for"
                             " simulated annealing")
        if init_temp <= final_temp:
            raise ValueError("Initial temperature must be higher than final"
                             " temperature")
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.num_temp = num_temp
        self.num_steps = num_steps

        self.corrFunc = CorrFunction(BC)
        self.eci = eci
        print(len(self.provided_cnames))

    def in_conc_matrix(self, atoms):
        # determine the concentration of the given atoms
        conc = np.zeros(self.num_elements, dtype=int)
        for x in range(self.num_elements):
            element = self.all_elements[x]
            num_element = len([a for a in atoms if a.symbol == element])
            conc[x] = num_element

        # This is a temporary line
        if len([a for a in atoms if a.symbol == 'Li']) < 15:
            return False
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

    def change_element_type(self, atoms, cf, index=None, rplc_element=None):
        # --------------------------------------------------------------------
        # Choose which element to swap. Keep track of atomic index, old symbol
        # and new symbol.
        # --------------------------------------------------------------------
        atoms = atoms.copy()
        cf = deepcopy(cf)
        natoms = len(atoms)
        while True:
            # pick an atom and determine its symbol
            if index is None:
                indx = choice(range(natoms))
            else:
                indx = index
            old_symbol = atoms[indx].symbol
            # determine its site type
            for site in range(self.num_sites):
                if old_symbol in self.site_elements[site]:
                    break
            # change element type
            if rplc_element is None:
                new_symbol = choice(self.site_elements[site])
                if new_symbol != old_symbol:
                    atoms[indx].symbol = new_symbol
                    if self.in_conc_matrix(atoms):
                        break
                    atoms[indx].symbol = old_symbol
            else:
                new_symbol = rplc_element
                atoms[indx].symbol = rplc_element
                break
        # -------------------------------------------------------------------
        # Updating correlation function may require the use of supercell. The
        # index of atom needs to be checked again after size conversion. Then,
        # the change in correlation function is tracked.
        # -------------------------------------------------------------------
        position = atoms[indx].position
        new_atoms = self.corrFunc.check_and_convert_cell_size(atoms.copy())
        nnatoms = len(new_atoms)
        scale = nnatoms / natoms
        # get the index of the atom with the same position
        for atom in new_atoms:
            if np.allclose(atom.position, position):
                nindx = atom.index
                break
        for name in self.provided_cnames:
            # print name, self.spin_dict
            if name == 'c0':
                continue
            elif name == 'c1':
                cf[name] = self.corrFunc.get_c1(new_atoms)
                continue
            # find the location of the name in cluster_names
            for n in range(2, len(self.cluster_names)):
                try:
                    typ = self.cluster_names[n].index(name)
                    num = n
                    break
                except ValueError:
                    continue

            tot_count = len(self.cluster_indx[num][typ]) * nnatoms
            sp_tot = cf[name] * tot_count
            indx_list = self.cluster_indx[num][typ]
            # count = len(indx_list)
            sp_old = self.sp_with_indx(new_atoms, indx_list, nindx, old_symbol)
            sp_new = self.sp_with_indx(new_atoms, indx_list, nindx, new_symbol)
            sp = (sp_tot + (num * scale) * (sp_new - sp_old)) / tot_count
            cf[name] = sp
        return atoms, cf

    def swap_two_atoms(self, atoms, cf):
        # --------------------------------------------------------------------
        # Choose which two element to swap. Keep track of their atomic indices
        # and symbols.
        # --------------------------------------------------------------------
        atoms = atoms.copy()
        cf = deepcopy(cf)
        natoms = len(atoms)
        indx = np.zeros(2, dtype=int)
        symbol = [None] * 2
        # pick fist atom and determine its symbol and type
        #indx[0] = choice(range(natoms))
        #--------
        # to be removed
        #-------
        indx_list = [a.index for a in atoms if a.symbol != 'V']
        indx[0] = choice(indx_list)

        symbol[0] = atoms[indx[0]].symbol
        for site in range(self.num_sites):
            if symbol[0] in self.site_elements[site]:
                break
        # pick second atom that is not the same element, but occupies the
        # same site.
        while True:
            #indx[1] = choice(range(natoms))
            #------
            # to be removed
            #------
            indx[1] = choice(indx_list)
            
            symbol[1] = atoms[indx[1]].symbol
            if symbol[1] == symbol[0]:
                continue
            if symbol[1] in self.site_elements[site]:
                break
        # swap two atoms
        atoms, cf = self.change_element_type(atoms, cf, indx[0], symbol[1])
        atoms, cf = self.change_element_type(atoms, cf, indx[1], symbol[0])

        return atoms, cf

    def sp_with_indx(self, atoms, indx_list, indx, symbol):
        ref_spin = self.spin_dict[symbol]
        sp = 0.
        for cluster in range(len(indx_list)):
            sp_temp = ref_spin
            for i in indx_list[cluster][:]:
                trans_indx = self.trans_matrix[indx, i]
                sp_temp *= self.spin_dict[atoms[trans_indx].symbol]
            sp += sp_temp
        return sp

    def get_mean_variance(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        mv = 0.
        for x in range(cfm.shape[0]):
            mv += cfm[x].dot(prec).dot(cfm[x].T)
        mv = mv / cfm.shape[0]
        return mv

    def get_mean_variance_Seko(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        sigma = np.cov(cfm.T)
        mu = np.mean(cfm, axis=0)
        mv = np.trace(prec.dot(sigma)) + mu.dot(prec).dot(mu.T)
        return mv

    def get_mean_variance_approx(self, cfm):
        prec = inv(cfm.T.dot(cfm))
        return np.trace(prec)

    def generate_probe_structure(self):
        """
        Generate a probe structure according to PRB 80, 165122 (2009)
        """
        evaluate = Evaluate(self.db_name, self.provided_cnames)
        cfm = evaluate.full_cf_matrix
        # Start
        atoms = self.init.copy()
        cf1d = self.corrFunc.get_cf_by_cluster_names(atoms,
                                                     self.provided_cnames)
        cf1 = np.array([cf1d[x] for x in self.provided_cnames], dtype=float)
        cfm1 = np.vstack((cfm, cf1))
        cfm1 = reduce_matrix(cfm1)
        mv1 = self.get_mean_variance_Seko(cfm1)
        # mv1s = self.get_mean_variance_Seko(cfm1)

        kTs = kB *np.logspace(math.log10(self.init_temp), math.log10(self.final_temp),
                              self.num_temp)
        steps_per_temp = self.num_steps/self.num_temp
        for kT in kTs:
            for _ in range(steps_per_temp):
                if bool(getrandbits(1)):
                    atoms2, cf2d = self.change_element_type(atoms, cf1d)
                else:
                    atoms2, cf2d = self.swap_two_atoms(atoms, cf1d)
                cf2 = np.array([cf2d[x] for x in self.provided_cnames],
                               dtype=float)
                cfm2 = np.vstack((cfm, cf2))
                cfm2 = reduce_matrix(cfm2)
                mv2 = self.get_mean_variance_Seko(cfm2)

                # mv2s = self.get_mean_variance_Seko(cfm2)
                accept = np.exp((mv1 - mv2) / kT) > np.random.uniform()
                # print mv1 - mv2, np.exp((mv1 - mv2) / kT)
                if accept:
                    atoms = atoms2.copy()
                    cf1 = np.copy(cf2)
                    cf1d = deepcopy(cf2d)
                    mv1 = deepcopy(mv2)
                    # mv1s = deepcopy(mv2s)

        # Check to see if the cf is indeed preserved
        cfd_new =self.corrFunc.get_cf(atoms)
        if not self.two_dicts_equal(cf1d, cfd_new):
            raise ValueError("The correlation function changed after simulated annealing")

        return atoms, cf1d

    def generate_Emin_structure(self):
        if self.eci is None:
            raise ValueError("ECI values need to be provided to generate Emin"
                             " structures")
        atoms = self.init.copy()
        cfd1 = self.corrFunc.get_cf_by_cluster_names(atoms,
                                                     self.provided_cnames)
        cf1 = np.array([cfd1[x] for x in self.provided_cnames], dtype=float)
        energy1 = cf1.dot(self.eci)

        kTs = kB *np.logspace(math.log10(self.init_temp), math.log10(self.final_temp),
                              self.num_temp)
        steps_per_temp = self.num_steps/self.num_temp
        for kT in kTs:
            for _ in range(steps_per_temp):
                atoms2, cfd2 = self.swap_two_atoms(atoms, cfd1)
                cf2 = np.array([cfd2[x] for x in self.provided_cnames],
                               dtype=float)
                energy2 = cf2.dot(self.eci)
                accept = np.exp((energy1 - energy2) / kT) > np.random.uniform()
                if accept:
                    print(accept, energy1, energy2)
                    atoms = atoms2.copy()
                    cf1 = np.copy(cf2)
                    cfd1 = deepcopy(cfd2)
                    energy1 = energy2

        # Check to see if the cf is indeed preserved
        cfd_new =self.corrFunc.get_cf_by_cluster_names(atoms,
                                                       self.provided_cnames)
        if not self.two_dicts_equal(cfd1, cfd_new):
            raise ValueError("The correlation function changed after simulated annealing")
        cfd1['Epred'] = energy1

        return atoms, cfd1




    def two_dicts_equal(self, dict1, dict2):
        keys1 = list(dict1.keys())
        keys2 = list(dict2.keys())
        if set(keys1) != set(keys2):
            print("Two dictionaries have different keys.")
            return False

        for key in keys1:
            if abs(dict1[key] - dict2[key]) > 1e-6:
                print("values are different\n"
                      "key:{}, dict1:{} dict2:{}".format(key, dict1[key], dict2[key]))
                return False
        return True
