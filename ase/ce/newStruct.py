"""Module for generating new structures for training."""
import os
import random
import numpy as np

from ase.ce import BulkCrystal, BulkSpacegroup, CorrFunction
from ase.ce.probestructure import ProbeStructure
from ase.ce.tools import wrap_and_sort_by_position
from ase.ce.structure_comparator import SymmetryEquivalenceCheck
from ase.ce.structure_comparator import SpgLibNotFoundError
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read
from ase.db import connect

max_attempt = 10


class MaxAttemptReachedError(Exception):
    """Raised when number of try reaches 10."""

    pass


class GenerateStructures(object):
    """Generate new structure in Atoms object format.

    Arguments:
    =========
    setting: BulkCrystal or BulkSapcegroup object

    generation_number: int
        a generation number to be assigned to the newly generated structure.

    struct_per_gen: int
        number of structures to generate per generation.
    """

    def __init__(self, setting, generation_number=None, struct_per_gen=None):
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            msg = "setting must be BulkCrystal or BulkSpacegroup object"
            raise TypeError(msg)
        self.setting = setting
        self.db = connect(setting.db_name)
        self.corrfunc = CorrFunction(self.setting)
        self.conc_matrix = self.setting.conc_matrix
        self.atoms = setting.atoms_with_given_dim
        if generation_number is None:
            self.gen = self._determine_gen_number()
        else:
            self.gen = generation_number

        # If the number of structures to include per generation is not given,
        # use one structure per computable concentration value
        # Note: May not be the most practical solution for some cases
        if struct_per_gen is None:
            if self.setting.num_conc_var == 1:
                self.struct_per_gen = self.conc_matrix.shape[0]
            else:
                self.struct_per_gen = self.conc_matrix.shape[0] *\
                    self.conc_matrix.shape[1]
        else:
            self.struct_per_gen = struct_per_gen

    def generate_probe_structure(self, init_temp=None, final_temp=None,
                                 num_temp=5, num_steps=10000,
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
            # Break out of the loop if reached struct_per_gen
            num_struct = len([row.id for row in
                              self.db.select(gen=self.gen)])
            if num_struct >= self.struct_per_gen:
                break

            # special case where there is only 1 concentration value
            if self.conc_matrix.ndim == 1:
                conc_ratio = np.copy(self.conc_matrix)
                conc_value = [1.0, None]
            else:
                conc_ratio, conc_value = self._get_random_conc()
            atoms = self._random_struct_at_conc(conc_ratio, conc_value[0],
                                                conc_value[1], False)
            # selected one of the "ghost" concentration with negative values
            if atoms is None:
                continue

            atoms = wrap_and_sort_by_position(atoms)
            print('Generating {} out of {} structures.'
                  .format(num_struct + 1, self.struct_per_gen))
            ps = ProbeStructure(self.setting, atoms, self.struct_per_gen,
                                init_temp, final_temp, num_temp, num_steps,
                                approx_mean_var)
            atoms, cf = ps.generate()
            conc = self._find_concentration(atoms)
            if self._exists_in_db(atoms, conc[0], conc[1]):
                print('generated structure is already in DB.')
                print('generating again...')
                num_attempt += 1
                continue
            else:
                num_attempt = 0

            kvp = self._get_kvp(atoms, cf, conc[0], conc[1])
            self.db.write(atoms, kvp)

            if num_attempt >= max_attempt:
                msg = "Could not generate probe structure in 10 attempts."
                raise MaxAttemptReachedError(msg)

    def generate_initial_pool(self):
        """Generate initial pool of random structures.

        If struct_per_gen is not specified, one structure is generated for each
        concentration value realizable by the given cell size.

        If structure_per_gen is not the same as the number of possible
        concentration values, a random structure is generated at a random
        concentration value in each iteration step.
        """
        print("Generating initial pool consisting of "
              "{} structures".format(self.struct_per_gen))

        # Special case where there is only 1 concentration value
        if self.conc_matrix.ndim == 1:
            conc1 = 1.0
            x = 0
            while x < self.struct_per_gen:
                atoms = self._random_struct_at_conc(self.conc_matrix, conc1)
                atoms = wrap_and_sort_by_position(atoms)
                kvp = self.corrfunc.get_cf(atoms)
                kvp = self._get_kvp(atoms, kvp, conc1)
                self.db.write(atoms, kvp)
                x += 1
            return True

        # Case 1: 1 conc variable, one struct per concentration
        if (self.setting.num_conc_var == 1
                and self.struct_per_gen == self.conc_matrix.shape[0]):
            for x in range(self.conc_matrix.shape[0]):
                conc1 = float(x) / max(self.conc_matrix.shape[0] - 1, 1)
                atoms = self._random_struct_at_conc(self.conc_matrix[x], conc1)
                if atoms is None:
                    continue
                atoms = wrap_and_sort_by_position(atoms)
                kvp = self.corrfunc.get_cf(atoms)
                kvp = self._get_kvp(atoms, kvp, conc1)
                self.db.write(atoms, kvp)

        # Case 2: 2 conc variable, one struct per concentration
        elif (self.setting.num_conc_var == 2
              and self.struct_per_gen == self.conc_matrix.shape[0]
              * self.conc_matrix.shape[1]):
            for x in range(self.conc_matrix.shape[0]):
                conc1 = float(x) / max(self.conc_matrix.shape[0] - 1, 1)
                for y in range(self.conc_matrix.shape[1]):
                    conc2 = float(y) / max(self.conc_matrix.shape[1] - 1, 1)
                    atoms = self._random_struct_at_conc(self.conc_matrix[x][y],
                                                        conc1, conc2)
                    if atoms is None:
                        continue
                    atoms = wrap_and_sort_by_position(atoms)
                    kvp = self.corrfunc.get_cf(atoms)
                    kvp = self._get_kvp(atoms, kvp, conc1, conc2)
                    self.db.write(atoms, kvp)

        # Case 3: 1 conc variable, user specified number of structures
        elif self.setting.num_conc_var == 1:
            num_conc1 = self.conc_matrix.shape[0]
            conc1_opt = range(num_conc1)
            x = 0
            num_attempt = 0
            while x < self.struct_per_gen:
                a = random.choice(conc1_opt)
                conc1 = float(a) / max(num_conc1 - 1, 1)
                atoms = self._random_struct_at_conc(self.conc_matrix[a], conc1)
                if atoms is None:
                    continue
                atoms = wrap_and_sort_by_position(atoms)
                num_attempt += 1
                if not self._exists_in_db(atoms, conc1):
                    kvp = self.corrfunc.get_cf(atoms)
                    kvp = self._get_kvp(atoms, kvp, conc1)
                    self.db.write(atoms, kvp)
                    x += 1
                    num_attempt = 0
                if num_attempt >= max_attempt:
                    msg = "Could not find initial structure in 10 attempts."
                    raise MaxAttemptReachedError(msg)

        # Case 4: 2 conc variable, user specified number of structures
        else:
            num_conc1, num_conc2 = self.conc_matrix.shape[:2]
            conc1_opt = range(num_conc1)
            conc2_opt = range(num_conc2)
            x = 0
            num_attempt = 0
            while x < self.struct_per_gen:
                a = [random.choice(conc1_opt), random.choice(conc2_opt)]
                conc1 = float(a[0]) / max(num_conc1 - 1, 1)
                conc2 = float(a[1]) / max(num_conc2 - 1, 1)
                atoms = \
                    self._random_struct_at_conc(self.conc_matrix[a[0]][a[1]],
                                                conc1, conc2)
                if atoms is None:
                    continue
                atoms = wrap_and_sort_by_position(atoms)
                num_attempt += 1
                if not self._exists_in_db(atoms, conc1, conc2):
                    kvp = self.corrfunc.get_cf(atoms)
                    kvp = self._get_kvp(atoms, kvp, conc1, conc2)
                    self.db.write(atoms, kvp)
                    x += 1
                    num_attempt = 0
                if num_attempt >= max_attempt:
                    msg = "Could not find initial structure in 10 attempts."
                    raise MaxAttemptReachedError(msg)

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

        if isinstance(init_struct, Atoms):
            init = wrap_and_sort_by_position(init_struct)
        else:
            init = wrap_and_sort_by_position(read(init_struct))

        conc = self._find_concentration(init)
        if self._exists_in_db(init, conc[0], conc[1]):
            raise RuntimeError('supplied structure already exists in DB')

        cf = self.corrfunc.get_cf(init)
        kvp = self._get_kvp(init, cf, conc[0], conc[1])

        if name is not None:
            kvp['name'] = name

        if final_struct is not None:
            if isinstance(final_struct, Atoms):
                energy = final_struct.get_potential_energy()
            else:
                energy = read(final_struct).get_potential_energy()
            calc = SinglePointCalculator(init, energy=energy)
            init.set_calculator(calc)
            kvp['converged'] = True
            kvp['started'] = ''
            kvp['queued'] = ''

        self.db.write(init, key_value_pairs=kvp)

    def _find_concentration(self, atoms):
        """Find the concentration value(s) of the passed atoms object."""
        if self.setting.grouped_basis is None:
            num_elements = self.setting.num_elements
            all_elements = self.setting.all_elements
        else:
            num_elements = self.setting.num_grouped_elements
            all_elements = self.setting.all_grouped_elements
        # determine the concentration of the given atoms
        conc_ratio = np.zeros(num_elements, dtype=int)
        for i, element in enumerate(all_elements):
            conc_ratio[i] = len([a for a in atoms if a.symbol == element])

        if self.setting.num_conc_var == 1:
            num_conc1 = self.conc_matrix.shape[0]
            for x in range(num_conc1):
                if np.array_equal(conc_ratio, self.conc_matrix[x]):
                    break
            conc = [round(float(x) / max(num_conc1 - 1, 1), 3), None]
        else:
            num_conc1, num_conc2 = self.conc_matrix.shape[:2]
            for x in range(num_conc1):
                for y in range(num_conc2):
                    if np.array_equal(conc_ratio, self.conc_matrix[x][y]):
                        break
                else:
                    continue
                break
            conc = [round(float(x) / max(num_conc1 - 1, 1), 3),
                    round(float(y) / max(num_conc2 - 1, 1), 3)]

        return conc

    def _exists_in_db(self, atoms, conc1=None, conc2=None):
        """Check to see if the passed atoms already exists in DB.

        To reduce the number of assessments for symmetry equivalence,
        check is only performed with the entries with the same concentration
        value.

        Return *True* if there is a symmetry-equivalent structure in DB,
        return *False* otherwise.

        Arguments:
        =========
        atoms: Atoms object

        conc1: float
            concentration value

        conc2: float
            secondary concentration value if more than one concentration
            variables are needed.
        """
        if conc1 is None:
            raise ValueError('conc1 needs to be defined')
        conc1 = round(conc1, 3)
        cond = [('conc1', '=', conc1)]
        if conc2 is not None:
            conc2 = round(conc2, 3)
            cond.append(('conc2', '=', conc2))
        # find if there is a match
        match = False
        to_prim = True
        try:
            import spglib
        except ImportError:
            msg = "Warning! Setting to_primitive=False because spglib "
            msg += "is missing!"
            print(msg)
            to_prim = False

        symmcheck = SymmetryEquivalenceCheck(angle_tol=1.0, ltol=0.05,
                                             stol=0.05, scale_volume=True,
                                             to_primitive=to_prim)
        for row in self.db.select(cond):
            atoms2 = row.toatoms()
            match = symmcheck.compare(atoms, atoms2)
            if match:
                break
        return match

    def _get_kvp(self, atoms, kvp, conc1=None, conc2=None):
        """Get key-value pairs of the passed Atoms object.

        Append terms (started, gen, converged, started, queued, name, conc)
        to key-value pairs and return it.

        Arguments:
        =========
        atoms: Atoms object

        kvp: dict
            key-value pairs (correlation functions of the passed atoms)

        conc1: float
            concentration value

        conc2: float
            secondary concentration value if more than one concentration
            variables are needed.
        """
        if conc1 is None:
            raise ValueError('conc1 needs to be defined')
        conc1 = round(conc1, 3)
        kvp['conc1'] = conc1
        kvp['gen'] = self.gen
        kvp['converged'] = False
        kvp['started'] = False
        kvp['queued'] = False

        # Determine name
        if conc2 is None:
            n = len([row.id for row in self.db.select(conc1=conc1)])
            kvp['name'] = 'conc_{:.3f}_{}'.format(conc1, n)
        else:
            conc2 = round(conc2, 3)
            kvp['conc2'] = conc2
            n = len([row.id for row in self.db.select(conc1=conc1,
                                                      conc2=conc2)])
            kvp['name'] = 'conc_{:.3f}_{:.3f}_{}'.format(conc1, conc2, n)
        return kvp

    def _get_random_conc(self):
        """Pick a random concentration value."""
        num_attempt = 0
        while True:
            if self.setting.num_conc_var == 1:
                conc_index = [random.choice(range(self.conc_matrix.shape[0])),
                              None]
                conc_ratio = self.conc_matrix[conc_index[0]]
            else:
                conc_index = [random.choice(range(self.conc_matrix.shape[0])),
                              random.choice(range(self.conc_matrix.shape[1]))]
                conc_ratio = self.conc_matrix[conc_index[0]][conc_index[1]]

            # loop until no atom has a negative number count
            if min(conc_ratio) >= 0:
                break
            num_attempt += 1
            if num_attempt >= max_attempt:
                msg = "Could not find random concentration."
                raise MaxAttemptReachedError(msg)

        conc_value = [None, None]
        for i, index in enumerate(conc_index):
            if index is None:
                continue
            conc_value[i] = float(conc_index[i]) /\
                max(self.conc_matrix.shape[i] - 1, 1)

        return conc_ratio, conc_value

    def _random_struct_at_conc(self, conc_ratio, conc1=None, conc2=None,
                               unique=True):
        """Generate a random structure that does not already exist in DB."""
        # convert the conc_ratio into the same format as basis_elements if
        # setting.group_basis is None
        # Else, convert the conc_ratio the format of
        # setting.grouped_basis_elements
        tmp = list(conc_ratio)
        conc_ratio = []
        if self.setting.grouped_basis is None:
            num_basis = self.setting.num_basis
            basis_elements = self.setting.basis_elements
        else:
            num_basis = self.setting.num_grouped_basis
            basis_elements = self.setting.grouped_basis_elements
        for site in range(num_basis):
            length = len(basis_elements[site])
            conc_ratio.append(tmp[:length])
            del tmp[:length]

        # 1D np array to easily count how many types of species are present
        flat_conc_ratio = np.array([x for sub in conc_ratio for x in sub])
        if min(flat_conc_ratio) < 0:
            return None

        in_DB = True
        num_attempt = 0
        while in_DB:
            atoms = self.atoms.copy()
            for site in range(num_basis):
                indx = [a.index for a in atoms
                        if a.symbol == basis_elements[site][0]]
                if len(indx) != sum(conc_ratio[site]):
                    raise ValueError("number of atoms to be replaced in the "
                                     "Atoms object does not match the value "
                                     "in conc_ratio")
                for i in range(1, len(conc_ratio[site])):
                    indx = self._replace_rnd(indx,
                                             basis_elements[site][i],
                                             conc_ratio[site][i], atoms)

            if not unique:
                break

            in_DB = self._exists_in_db(atoms, conc1=conc1, conc2=conc2)
            # special case where only one species is present
            # break out of the loop and return atoms=None
            if in_DB and np.count_nonzero(flat_conc_ratio) == 1:
                atoms = None
                in_DB = False

            num_attempt += 1
            if num_attempt >= max_attempt:
                msg = "Could not find random structure at given concentration."
                raise MaxAttemptReachedError(msg)

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

    def _replace_rnd(self, indices, element, n_replace, atoms):
        if len(indices) < n_replace:
            raise ValueError("# of elements to replace > # of elements")
        replace = []
        for x in range(n_replace):
            selected = random.choice(indices)
            while selected in replace:
                selected = random.choice(indices)
            replace.append(selected)
        for atom in atoms:
            if atom.index in replace:
                atom.symbol = element

        indices = [x for x in indices if x not in replace]
        return indices

    def _generate_sigma_mu(self, num_samples_var):
        print('===========================================================\n'
              'Determining sigma and mu value for assessing mean variance.\n'
              'May take a long time depending on the number of samples \n'
              'specified in the *num_samples_var* argument.\n'
              '===========================================================')
        count = 0
        cfm = np.zeros((num_samples_var, len(self.setting.full_cluster_names)),
                       dtype=float)
        while count < num_samples_var:
            # special case where there is only 1 concentration value
            if self.conc_matrix.ndim == 1:
                conc_ratio = np.copy(self.conc_matrix)
                conc_value = [1.0, None]
            else:
                conc_ratio, conc_value = self._get_random_conc()
            atoms = self._random_struct_at_conc(conc_ratio, conc_value[0],
                                                conc_value[1], False)
            cfm[count] = self.corrfunc.get_cf(atoms, 'array')
            count += 1
            print('sampling {} ouf of {}'.format(count, num_samples_var))

        sigma = np.cov(cfm.T)
        mu = np.mean(cfm, axis=0)
        np.savez('probe_structure-sigma_mu.npz', sigma=sigma, mu=mu)
