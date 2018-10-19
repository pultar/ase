"""Module for generating probe structures."""
import os
import math
from random import choice, getrandbits
from copy import deepcopy
import numpy as np
from numpy.linalg import inv, pinv
from ase.db import connect
from ase.clease import CEBulk, CECrystal, CorrFunction
from ase.clease.tools import wrap_and_sort_by_position
from ase.calculators.clease import Clease
from ase.units import kB
import time


# class ProbeStructure(object):
class StructureGenerator(object):
    """Generate probe structures.

    Based on simulated annealing according to the recipe in
    PRB 80, 165122 (2009).

    Arguments:
    =========
    setting: CEBulk or BulkSapcegroup object

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
                 final_temp=None, num_temp=5, num_steps_per_temp=10000):
        if not isinstance(setting, (CEBulk, CECrystal)):
            raise TypeError("setting must be CEBulk or CECrystal "
                            "object")
        
        self.setting = setting
        self.trans_matrix = setting.trans_matrix
        self.cluster_names = self.setting.cluster_names
        self.corrFunc = CorrFunction(setting)
        self.cfm = self._get_full_cf_matrix()

        if self.setting.in_conc_matrix(atoms):
            if len(atoms) != len(setting.atoms_with_given_dim):
                raise ValueError("Passed Atoms has a wrong size.")
            self.supercell, self.periodic_indices, self.index_by_basis = \
                self._build_supercell(wrap_and_sort_by_position(atoms.copy()))
        else:
            raise ValueError("concentration of the elements in the provided"
                             " atoms cannot be found in the conc_matrix")

        # eci set to 1 to ensure that all correlation functions are included 
        # but the energy produced from this should never be used 
        eci = {name: 1. for name in self.cluster_names}
        self.calc = Clease(self.setting, cluster_name_eci=eci)
        self.supercell.set_calculator(self.calc)
        self.output_every = 30
        self.init_temp = init_temp
        self.final_temp = final_temp
        self.temp = self.init_temp
        self.num_temp = num_temp
        self.num_steps_per_temp = num_steps_per_temp
        self.alter_composition = True

    def _build_supercell(self, atoms):
        for atom in atoms:
            atom.tag = atom.index

        natoms = len(atoms)
        atoms *= self.setting.supercell_scale_factor
        atoms = wrap_and_sort_by_position(atoms)

        periodic_indices = []
        for tag in range(natoms):
            periodic_indices.append([a.index for a in atoms if a.tag == tag])

        if self.setting.grouped_basis is None:
            tag_by_basis = self.setting.index_by_basis
        else:
            tag_by_basis = self.setting.index_by_grouped_basis

        index_by_basis = []
        for basis in tag_by_basis:
            basis_elements = []
            for atom in atoms:
                if atom.tag in basis:
                    basis_elements.append(atom.index)
            index_by_basis.append(basis_elements)

        return atoms, periodic_indices, index_by_basis

    def _supercell2unitcell(self):
        atoms = self.setting.atoms_with_given_dim.copy()
        for a in atoms:
            a.symbol = self.supercell[self.periodic_indices[a.index][0]].symbol
        return atoms

    def _reset(self):
        pass

    def generate(self):
        """Generate a probe structure according to PRB 80, 165122 (2009)."""
        # Start
        self._reset()
        if self.init_temp is None or self.final_temp is None:
            self.init_temp, self.final_temp = self._determine_temps()
        
        if self.init_temp <= self.final_temp:
            raise ValueError("Initial temperature must be higher than final"
                             " temperature")
        self._reset()

        temps = np.logspace(math.log10(self.init_temp),
                            math.log10(self.final_temp),
                            self.num_temp)
        now = time.time()
        for temp in temps:
            self.temp = temp
            num_accepted = 0
            count = 0
            while count < self.num_steps_per_temp:
                count += 1
                if time.time() - now > self.output_every:
                    acc_rate = float(num_accepted)/count
                    print("Temp: {}. {} of {}. Acc. rate: {}"
                          "".format(temp, count, self.num_steps_per_temp,
                                     acc_rate))
                    now = time.time()

                if bool(getrandbits(1)) and self.alter_composition:
                    # Change element Type
                    if self._has_more_than_one_conc():
                        self._change_element_type()
                    else:
                        continue
                else:
                    indx = self._swap_two_atoms()
                    if self.supercell[indx[0]].symbol == \
                            self.supercell[indx[1]].symbol:
                        continue
                self.supercell.get_potential_energy()

                if self._accept():
                    num_accepted += 1
                else:
                    self.calc.restore(self.supercell)

        self._check_consistency()
        return self._optimal_structure()
        #return self._supercell2unitcell(), self.calc.get_cf_dict()

    def _accept(self):
        raise NotImplementedError('This should be implemented in the inherited '
                                  'class.')
    
    def _estimate_temp_range(self):
        raise NotImplementedError('This should be implemented in the inherited '
                                  'class.')
    
    def _optimal_structure(self):
        raise NotImplementedError("This shoud be implemented in the inherited "
                                  "class.")

    def _determine_temps(self):
        print("Temperature range not given. "
              "Determining the range automatically.")
        self._reset()
        count = 0
        max_count = 100
        now = time.time()
        # To avoid errors, just set the temperature to 
        # an arbitrary file
        self.temp = 10000000.0
        while count < max_count:
            if time.time() - now > self.output_every:
                print("Progress ({}%)".format(100*count/max_count))
                now = time.time()

            if bool(getrandbits(1)) and self.alter_composition:
                # Change element Type
                if self._has_more_than_one_conc():
                    self._change_element_type()
                    count += 1
                else:
                    continue
            else:
                indx = self._swap_two_atoms()
                if self.supercell[indx[0]].symbol == \
                        self.supercell[indx[1]].symbol:
                    continue
                count += 1
            self.supercell.get_potential_energy()
            

            # By calling accept statistics on the correlation
            # function and variance will be collected
            self._accept()
        init_temp, final_temp = self._estimate_temp_range()
        self.temp = init_temp
        print('init_temp= {}, final_temp= {}'.format(init_temp, final_temp))
        return init_temp, final_temp

    def _swap_two_atoms(self):
        """Swap two randomly chosen atoms."""
        indx = np.zeros(2, dtype=int)
        symbol = [None] * 2

        # determine if the basis is grouped
        if self.setting.grouped_basis is None:
            basis_elements = self.setting.basis_elements
            num_basis = self.setting.num_basis
        else:
            basis_elements = self.setting.grouped_basis_elements
            num_basis = self.setting.num_grouped_basis

        # pick fist atom and determine its symbol and type
        while True:
            basis = choice(range(num_basis))
            # a basis with only 1 type of element should not be chosen
            if len(basis_elements[basis]) < 2:
                continue
            indx[0] = choice(self.index_by_basis[basis])
            symbol[0] = self.supercell[indx[0]].symbol
            break
        # pick second atom that occupies the same basis.
        while True:
            indx[1] = choice(self.index_by_basis[basis])
            symbol[1] = self.supercell[indx[1]].symbol
            if symbol[1] in basis_elements[basis]:
                break

        # Swap two elements
        self.supercell[indx[0]].symbol = symbol[1]
        self.supercell[indx[1]].symbol = symbol[0]

        # find which index it should be in unit cell
        for i in indx:
            unit_cell_indx = self.supercell[i].tag
            for grp_index in self.periodic_indices[unit_cell_indx]:
                self.supercell[grp_index].symbol = self.supercell[i].symbol
        return indx

    def _has_more_than_one_conc(self):
        if len(self.setting.conc_matrix) > 1 \
                and self.setting.conc_matrix.ndim > 1:
            return True
        return False

    def _change_element_type(self):
        """Change the type of element for the atom with a given index.

        If index and replacing element types are not specified, they are
        randomly generated.
        """
        if self.setting.grouped_basis is None:
            basis_elements = self.setting.basis_elements
            num_basis = self.setting.num_basis
        else:
            basis_elements = self.setting.grouped_basis_elements
            num_basis = self.setting.num_grouped_basis
        # ------------------------------------------------------
        # Change the type of element for a given index if given.
        # If index not given, pick a random index
        # ------------------------------------------------------
        while True:
            basis = choice(range(num_basis))
            # a basis with only 1 type of element should not be chosen
            if len(basis_elements[basis]) < 2:
                continue

            indx = choice(self.index_by_basis[basis])
            old_symbol = self.supercell[indx].symbol

            # change element type
            new_symbol = choice(basis_elements[basis])
            if new_symbol != old_symbol:
                self.supercell[indx].symbol = new_symbol

                # Update all periodic images
                unit_cell_indx = self.supercell[indx].tag
                for grp_index in self.periodic_indices[unit_cell_indx]:
                    self.supercell[grp_index].symbol \
                        = self.supercell[indx].symbol

                if self.setting.in_conc_matrix(self._supercell2unitcell()):
                    break
                self.supercell[indx].symbol = old_symbol

                # Revert all periodic images
                for grp_index in self.periodic_indices[unit_cell_indx]:
                    self.supercell[grp_index].symbol \
                        = self.supercell[indx].symbol

    def _check_consistency(self):
        # Check to see if the cf is indeed preserved
        final_cf = \
            self.corrFunc.get_cf_by_cluster_names(self.supercell,
                                                  self.calc.cluster_names,
                                                  return_type='array')
        # for i in range(len(self.calc.cluster_names)):
        #     print(self.calc.cluster_names[i], final_cf[i] - self.calc.cf[i])
        if not np.allclose(final_cf, self.calc.cf):
            msg = 'The correlation function changed after simulated annealing'
            raise ValueError(msg)

        for grp in self.periodic_indices:
            ref_symbol = self.supercell[grp[0]].symbol
            for indx in grp[1:]:
                assert self.supercell[indx].symbol == ref_symbol

    def _get_full_cf_matrix(self):
        """Get correlation function of every entry in DB."""
        cfm = []
        db = connect(self.setting.db_name)
        for row in db.select([('name', '!=', 'information')]):
            cfm.append([row[x] for x in self.cluster_names])
        cfm = np.array(cfm, dtype=float)
        return cfm


class ProbeStructure(StructureGenerator):
    def __init__(self, setting, atoms, struct_per_gen, init_temp=None,
                 final_temp=None, num_temp=5, num_steps_per_temp=1000,
                 approx_mean_var=False):

        StructureGenerator.__init__(self, setting, atoms, struct_per_gen, 
                    init_temp, final_temp, num_temp, num_steps_per_temp)
        self.o_cf = self.calc.cf
        self.o_cfm = np.vstack((self.cfm, self.o_cf))
        self.approx_mean_var = approx_mean_var
        fname = 'probe_structure-sigma_mu.npz'
        if not approx_mean_var:
            if os.path.isfile(fname):
                data = np.load(fname)
                self.sigma = data['sigma']
                self.mu = data['mu']
                self.o_mv = mean_variance(self.o_cfm, self.sigma, self.mu)
            else:
                raise IOError("'{}' not found.".format(fname))
        else:
            self.o_mv = mean_variance_approx(self.o_cfm)
        self.avg_mv = 0.0
        self.num_steps = 0
        self.avg_diff = 0.0

        self.min_mv = None
        self.min_mv_atoms = None
        self.min_mv_cf = None

    def _accept(self):
        """Accept the last change."""
        cfm = np.vstack((self.cfm, self.calc.cf))
        if self.approx_mean_var:
            n_mv = mean_variance_approx(cfm)
        else:
            n_mv = mean_variance(cfm, self.sigma, self.mu)

        # Always accept the first move
        if self.min_mv_atoms is None:
            self.min_mv_atoms = self._supercell2unitcell()
            self.min_mv = n_mv
            self.min_mv_cf = deepcopy(self.calc.cf)
            return True

        if n_mv < self.o_mv:
            self.min_mv_atoms = self._supercell2unitcell()
            self.min_mv = n_mv
            self.min_mv_cf = deepcopy(self.calc.cf)

        if self.o_mv > n_mv:
            accept_move = True
        else:
            accept_move = np.exp((self.o_mv-n_mv)/self.temp) > np.random.uniform()
        
        self.avg_diff += abs(n_mv - self.o_mv)
        if accept_move:
            self.o_mv = n_mv

        self.avg_mv += self.o_mv
        self.num_steps += 1
        return accept_move

    def _estimate_temp_range(self):
        if self.num_steps == 0:
            return 100000.0, 1.0
        avg_diff = self.avg_diff/self.num_steps
        init_temp = 10 * avg_diff
        final_temp = 0.01 * avg_diff
        return init_temp, final_temp

    def _optimal_structure(self):
        return self.min_mv_atoms, self.min_mv_cf

    @property
    def avg_mean_variance(self):
        if self.num_steps == 0:
            return 0.0
        return self.avg_mv/self.num_steps

    def _reset(self):
        self.avg_mv = 0.0
        self.avg_diff = 0.0
        self.num_steps = 0
        

class EminStructure(StructureGenerator):
    def __init__(self, setting, atoms, struct_per_gen, init_temp=2000,
                 final_temp=10, num_temp=10, num_steps_per_temp=10000, 
                 cluster_name_eci=None):
        StructureGenerator.__init__(self, setting, atoms, struct_per_gen,
                                    init_temp, final_temp, num_temp, 
                                    num_steps_per_temp)
        self.alter_composition = False
        self.calc = Clease(self.setting, cluster_name_eci=cluster_name_eci)
        self.old_energy = None
        self.min_energy = None
        self.min_energy_atoms = None
        self.min_energy_cf = None

    def _accept(self):
        """Accept the last change."""
        new_energy = self.calc.energy
        if self.old_energy is None:
            self.old_energy = new_energy
            self.min_energy = new_energy
            self.min_energy_atoms = self._supercell2unitcell()
            self.min_energy_cf = deepcopy(self.calc.cf)
            return True
    
        if new_energy < self.min_energy:
            self.min_energy = new_energy
            self.min_energy_atoms = self._supercell2unitcell()
            self.min_energy_cf = deepcopy(self.calc.cf)

        if new_energy < self.old_energy:
            self.old_energy = new_energy
            return True
        
        diff = new_energy - self.old_energy
        kT = kB*self.temp
        accept_move = np.random.uniform() < np.exp(-diff/kT)

        if accept_move:
            self.old_energy = new_energy
        return accept_move

    def _optimal_structure(self):
        return self.min_energy_atoms, self.min_energy_cf

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
        prec = pinv(cfm.T.dot(cfm))
    return prec
