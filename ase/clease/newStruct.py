"""Module for generating new structures for training."""
import os
import random
import numpy as np
from random import shuffle
from copy import deepcopy
from fractions import gcd
from functools import reduce

from ase.io import read
from ase.db import connect
from ase.atoms import Atoms
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from ase.calculators.singlepoint import SinglePointCalculator

from ase.clease import CEBulk, CECrystal, CorrFunction
from ase.clease.structure_generator import ProbeStructure, EminStructure
from ase.clease.tools import wrap_and_sort_by_position

max_attempt = 10


class MaxAttemptReachedError(Exception):
    """Raised when number of try reaches 10."""

    pass


# class GenerateStructures(object):
class NewStructures(object):
    """Generate new structure in Atoms object format.

    Arguments:
    =========
    setting: CEBulk or BulkSapcegroup object

    generation_number: int
        a generation number to be assigned to the newly generated structure.

    struct_per_gen: int
        number of structures to generate per generation.
    """

    def __init__(self, setting, generation_number=None, struct_per_gen=5):
        if not isinstance(setting, (CEBulk, CECrystal)):
            msg = "setting must be CEBulk or CECrystal object"
            raise TypeError(msg)
        self.setting = setting
        self.db = connect(setting.db_name)
        self.corrfunc = CorrFunction(self.setting)
        if generation_number is None:
            self.gen = self._determine_gen_number()
        else:
            self.gen = generation_number

        self.struct_per_gen = struct_per_gen

    def generate_probe_structure(self, init_temp=None, final_temp=None,
                                 num_temp=5, num_steps_per_temp=1000,
                                 approx_mean_var=False, num_samples_var=10000):
        """Generate a probe structure according to PRB 80, 165122 (2009).

        Arguments:
        =========
        init_temp: int or float
            initial temperature (does not represent *physical* temperature)

        final_temp: int or float
            final temperature (does not represent *physical* temperature)

        num_temp: int
            number of temperatures to be used in simulated annealing

        num_steps_per_temp: int
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
                      sigma and mu are generated and stored in
                      'probe_structure-sigma_mu.npz' file.

        num_samples_var: int
            Number of samples to be used in determining signam and mu.
            Only used when approx_mean_var is True.

        Note: init_temp and final_temp are automatically generated if either
              one of the two is not specified.
        """
        if not approx_mean_var:
            # check to see if there are files containing mu and sigma values
            if not os.path.isfile('probe_structure-sigma_mu.npz'):
                self._generate_sigma_mu(num_samples_var)

        print("Generate {} probe structures.".format(self.struct_per_gen))
        num_attempt = 0
        while True:
            self.setting.set_new_template()
            # Break out of the loop if reached struct_per_gen
            num_struct = len([row.id for row in
                              self.db.select(gen=self.gen)])
            if num_struct >= self.struct_per_gen:
                break
            
            atoms = self._get_struct_at_conc(conc_type='random')
            formula_unit = self._get_formula_unit(atoms)
            if self._exists_in_db(atoms, formula_unit):
                num_attempt += 1
                continue

            print('Generating {} out of {} structures.'
                  .format(num_struct + 1, self.struct_per_gen))
            ps = ProbeStructure(self.setting, atoms, self.struct_per_gen,
                                init_temp, final_temp, num_temp, 
                                num_steps_per_temp, approx_mean_var)
            atoms, cf = ps.generate()
            formula_unit = self._get_formula_unit(atoms)
            if self._exists_in_db(atoms, formula_unit):
                print('generated structure is already in DB.')
                print('generating again...')
                num_attempt += 1
                continue
            else:
                num_attempt = 0

            kvp = self._get_kvp(atoms, cf, formula_unit)
            self.db.write(atoms, kvp)

            if num_attempt >= max_attempt:
                msg = "Could not generate probe structure in 10 attempts."
                raise MaxAttemptReachedError(msg)

    def generate_Emin_structure(self, atoms=None, init_temp=2000, final_temp=1, 
                                num_temp=10, num_steps_per_temp=1000, 
                                cluster_names_eci=None):
        """Generate Emin structure.

        Arguments:
        =========
        atoms: Atoms object
            Atoms object with the desired composition of the new structure.
            A random composition is selected atoms=None.

        init_temp: int or float
            initial temperature (does not represent *physical* temperature)

        final_temp: int or float
            final temperature (does not represent *physical* temperature)

        num_temp: int
            number of temperatures to be used in simulated annealing

        num_steps_per_temp: int
            number of steps in simulated annealing
        """

        print("Generate {} Emin structures.".format(self.struct_per_gen))
        
        num_attempt = 0
        while True:
            # Break out of the loop if reached struct_per_gen
            if atoms is None:
                self.setting.set_new_template()
                atoms = self._get_struct_at_conc(conc_type='random')
            else:
                atoms = wrap_and_sort_by_position(atoms)
                num_struct = len([row.id for row in
                                self.db.select(gen=self.gen)])
                if num_struct >= self.struct_per_gen:
                    break

            print('Generating {} out of {} structures.'
                  .format(num_struct + 1, self.struct_per_gen))
            es = EminStructure(self.setting, atoms, self.struct_per_gen,
                               init_temp, final_temp, num_temp, 
                               num_steps_per_temp, cluster_names_eci)
            atoms, cf = es.generate()
            formula_unit = self._get_formula_unit(atoms)

            if self._exists_in_db(atoms, formula_unit):
                print('generated structure is already in DB.')
                print('generating again...')
                num_attempt += 1
                continue
            else:
                num_attempt = 0
        
            print('Structure with E = {} generated.'.format(es.min_energy))
            kvp = self._get_kvp(atoms, cf, formula_unit)
            self.db.write(atoms, kvp)

            if num_attempt >= max_attempt:
                msg = "Could not generate probe structure in 10 attempts."
                raise MaxAttemptReachedError(msg)

    def generate_initial_pool(self):
        """Generate initial pool of random structures."""

        print("Generating initial pool consisting of one structure per "
              "concentration where the number of an element is at max/min")
        
        for indx in range(self.setting.concentration.num_concs):
            for option in ["min", "max"]:
                atoms = self._get_struct_at_conc(conc_type=option, index=indx)
                atoms = wrap_and_sort_by_position(atoms)
                formula_unit = self._get_formula_unit(atoms)

                if not self._exists_in_db(atoms, formula_unit):
                    kvp = self.corrfunc.get_cf(atoms)
                    kvp = self._get_kvp(atoms, kvp, formula_unit)
                    self.db.write(atoms, kvp)
         

    def _get_struct_at_conc(self, conc_type='random', index=0):
        """Generate a structure at a concentration specified.

        Arguments:
        =========
        conc_type: str
            One of 'min', 'max' and 'random'
        
        index: int
            index of the flattened basis_element array to specify which element
            to be maximized/minimized
        """
        conc = self.setting.concentration
        if conc_type == 'min':
            x = conc.get_conc_min_component(index)
        elif conc_type == 'max':
            x = conc.get_conc_max_component(index)
        else:
            x = conc.get_random_concentration()
        
        num_atoms_in_basis = [len(indices) for indices  
                              in self.setting.index_by_basis]
        num_atoms_to_insert = conc.conc_in_int(num_atoms_in_basis, x)
        atoms = self._random_struct_at_conc(num_atoms_to_insert)
        return wrap_and_sort_by_position(atoms)

    def insert_structure(self, init_struct=None, final_struct=None, name=None):
        """Insert a user-supplied structure to the database.

        Arguments:
        =========
        init_struct: .xyz, .cif or .traj file
            *Unrelaxed* initial structure.

        final_struct: .traj file
            Final structure that contains the energy.
            Needs to also supply init_struct in order to use the final_struct.

        name: str
            Name of the DB entry if non-default name is to be used.
            If *None*, default naming convention will be used.
        """
        if init_struct is None:
            raise TypeError('init_struct must be provided')
            
        if name is not None:
            num = sum(1 for _ in self.db.select(['name', '=', name]))
            if num > 0:
                raise ValueError("Name: {} already exists in DB!".format(name))

        if isinstance(init_struct, Atoms):
            init = wrap_and_sort_by_position(init_struct)
        else:
            init = wrap_and_sort_by_position(read(init_struct))

        formula_unit = self._get_formula_unit(init)
        if self._exists_in_db(init, formula_unit):
            raise RuntimeError('supplied structure already exists in DB')

        cf = self.corrfunc.get_cf(init)
        kvp = self._get_kvp(init, cf, formula_unit)

        if name is not None:
            kvp['name'] = name

        kvp['converged'] = True
        kvp['started'] = ''
        kvp['queued'] = ''
        kvp['struct_type'] = 'initial'
        uid_init = self.db.write(init, key_value_pairs=kvp)

        if final_struct is not None:
            if isinstance(final_struct, Atoms):
                final = final_struct
            else:
                final = read(final_struct)
            kvp_final = {'struct_type': 'final', 'name': kvp['name']}
            uid = self.db.write(final, kvp_final)
            self.db.update(uid_init, final_struct_id=uid)

    def _exists_in_db(self, atoms, formula_unit=None):
        """Check to see if the passed atoms already exists in DB.

        To reduce the number of assessments for symmetry equivalence,
        check is only performed with the entries with the same concentration
        value.

        Return *True* if there is a symmetry-equivalent structure in DB,
        return *False* otherwise.

        Arguments:
        =========
        atoms: Atoms object

        formula_unit: str
            reduced formula unit of the passed Atoms object
        """
        cond = []
        if formula_unit is not None:
            cond = [("formula_unit", "=", formula_unit)]

        to_prim = True
        try:
            __import__('spglib')
        except Exception:
            msg = "Warning! Setting to_primitive=False because spglib "
            msg += "is missing!"
            print(msg)
            to_prim = False

        symmcheck = SymmetryEquivalenceCheck(angle_tol=1.0, ltol=0.05,
                                             stol=0.05, scale_volume=True,
                                             to_primitive=to_prim)
        atoms_in_db = []
        for row in self.db.select(cond):
            atoms_in_db.append(row.toatoms())
        return symmcheck.compare(atoms.copy(), atoms_in_db)

    def _get_kvp(self, atoms, kvp, formula_unit=None):
        """Get key-value pairs of the passed Atoms object.

        Append terms (started, gen, converged, started, queued, name, conc)
        to key-value pairs and return it.

        Arguments:
        =========
        atoms: Atoms object

        kvp: dict
            key-value pairs (correlation functions of the passed atoms)

        formula_unit: str
            reduced formula unit of the passed Atoms object
        """
        if formula_unit is None:
            raise ValueError("Formula unit not specified!")                        
        kvp['gen'] = self.gen
        kvp['converged'] = False
        kvp['started'] = False
        kvp['queued'] = False

        count = 0
        for _ in self.db.select(formula_unit=formula_unit):
            count += 1
        kvp['name'] = formula_unit+"_{}".format(count)
        kvp['formula_unit'] = formula_unit
        return kvp

    def _get_formula_unit(self, atoms):
        """Generates a reduced formula unit for the structure."""
        atom_count = []
        all_nums = []
        for group in self.setting.index_by_basis:
            new_count = {}
            for indx in group:
                symbol = atoms[indx].symbol
                if symbol not in new_count.keys():
                    new_count[symbol] = 1
                else:
                    new_count[symbol] += 1
            atom_count.append(new_count)
            all_nums += [v for k, v in new_count.items()]
        gcdp = reduce(lambda x, y: gcd(x, y), all_nums)
        fu = ""
        for i, count in enumerate(atom_count):
            keys = list(count.keys())
            keys.sort()
            for k in keys:
                fu += "{}{}".format(k, int(count[k]/gcdp))
            if i < len(atom_count)-1:
                fu += "_"
        return fu

    def _random_struct_at_conc(self, num_atoms_to_insert):
        """Generate a random structure."""
        randomized_indices = []
        for indices in self.setting.index_by_basis:
            randomized_indices.append(deepcopy(indices))
            shuffle(randomized_indices[-1])

        # Insert the number of atoms
        basis_elem = self.setting.concentration.basis_elements
        assert len(randomized_indices) == len(basis_elem)
        atoms = self.setting.atoms_with_given_dim.copy()

        current_conc = 0
        for basis in range(len(randomized_indices)):
            current_indx = 0
            for symb in basis_elem[basis]:
                for _ in range(num_atoms_to_insert[current_conc]):
                    atoms[randomized_indices[basis][current_indx]].symbol = symb
                    current_indx += 1
                current_conc += 1
        assert current_indx == len(atoms)
        return atoms

    def _determine_gen_number(self):
        """Determine generation number based on the values in DB."""
        try:
            gens = [row.get('gen') for row in self.db.select()]
            gens = [i for i in gens if i is not None]
            gen = max(gens) + 1
        except ValueError:
            gen = 0
        return gen

    def _generate_sigma_mu(self, num_samples_var):
        print('===========================================================\n'
              'Determining sigma and mu value for assessing mean variance.\n'
              'May take a long time depending on the number of samples \n'
              'specified in the *num_samples_var* argument.\n'
              '===========================================================')
        count = 0
        cfm = np.zeros((num_samples_var, len(self.setting.cluster_names)),
                       dtype=float)
        while count < num_samples_var:
            atoms = self._get_struct_at_conc(conc_type='random')
            cfm[count] = self.corrfunc.get_cf(atoms, 'array')
            count += 1
            print('sampling {} ouf of {}'.format(count, num_samples_var))

        sigma = np.cov(cfm.T)
        mu = np.mean(cfm, axis=0)
        np.savez('probe_structure-sigma_mu.npz', sigma=sigma, mu=mu)
