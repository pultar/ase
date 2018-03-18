import random
import numpy as np

# dependence on PyMatGen to be removed
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor

from ase.ce import BulkCrystal, BulkSpacegroup, CorrFunction
from ase.ce.probestructure import ProbeStructure
from ase.ce.tools import wrap_and_sort_by_position
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read


class GenerateStructures(object):
    """
    Class that generates atoms object.
    """

    def __init__(self, setting, generation_number=None, struct_per_gen=None):
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            raise TypeError("setting must be BulkCrystal or BulkSpacegroup "
                            "object")
        self.setting = setting
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

        # Generate initial pool of structures for generation 0
        # if self.gen == 0:
            # self.generate_initial_pool()

    def generate_probe_structure(self, init_temp=None, final_temp=None,
                                 num_temp=5, num_steps=10000):
        """Generate a probe structure according to PRB 80, 165122 (2009)."""
        print("Generate {} probe structures.".format(self.struct_per_gen))
        while True:
            # Break out of the loop if reached struct_per_gen
            num_struct = len([row.id for row in
                              self.setting.db.select(gen=self.gen)])
            if num_struct >= self.struct_per_gen:
                break

            # Pick an initial random structure
            if self.setting.num_conc_var == 1:
                num_conc = self.conc_matrix.shape[0]
                conc = random.choice(range(num_conc))
                conc1 = float(conc) / (self.conc_matrix.shape[0] - 1)
                print('conc1 = {}'.format(conc1))
                atoms = self.random_struct(self.conc_matrix[conc], conc1)
            else:
                num_conc1, num_conc2 = self.conc_matrix.shape[:2]
                conc = [random.choice(range(num_conc1)),
                        random.choice(range(num_conc2))]
                conc1 = float(conc[0]) / (self.conc_matrix.shape[0] - 1)
                conc2 = float(conc[1]) / (self.conc_matrix.shape[1] - 1)
                atoms = self.random_struct(self.conc_matrix[conc[0]][conc[1]],
                                           conc1, conc2)

            # selected one of the "ghost" concentration with negative values
            if atoms is None:
                continue

            atoms = wrap_and_sort_by_position(atoms)
            print('Generating {} out of {}'.format(num_struct + 1,
                                                   self.struct_per_gen) +
                  ' structures')
            ps = ProbeStructure(self.setting, atoms, self.struct_per_gen,
                                init_temp=init_temp, final_temp=final_temp,
                                num_temp=num_temp, num_steps=num_steps)
            atoms, cf_array = ps.generate()
            conc = self._find_concentration(atoms)
            if self._exists_in_db(atoms, conc[0], conc[1]):
                continue
            # convert cf array to dictionary
            cf = {}
            for i, name in enumerate(self.setting.full_cluster_names):
                cf[name] = cf_array[i]

            kvp = self.get_kvp(atoms, cf, conc[0], conc[1])
            self.setting.db.write(atoms, kvp)

    def generate_initial_pool(self):
        """Generates the initial pool"""
        print("Generating initial pool consisting of "
              "{} structures".format(self.struct_per_gen))

        # Case 1: 1 conc variable, one struct per concentration
        if (self.setting.num_conc_var == 1 and
            self.struct_per_gen == self.conc_matrix.shape[0]):
            for x in range(self.conc_matrix.shape[0]):
                conc1 = float(x)/(self.conc_matrix.shape[0] - 1)
                atoms = self.random_struct(self.conc_matrix[x], conc1)
                if atoms is None:
                    continue
                atoms = wrap_and_sort_by_position(atoms)
                kvp = CorrFunction(self.setting).get_cf(atoms)
                kvp = self.get_kvp(atoms, kvp, conc1)
                self.setting.db.write(atoms, kvp)

        # Case 2: 2 conc variable, one struct per concentration
        elif (self.setting.num_conc_var == 2 and
              self.struct_per_gen == self.conc_matrix.shape[0] *
              self.conc_matrix.shape[1]):
            for x in range(self.conc_matrix.shape[0]):
                conc1 = float(x)/(self.conc_matrix.shape[0] - 1)
                for y in range(self.conc_matrix.shape[1]):
                    conc2 = float(y)/(self.conc_matrix.shape[1] - 1)
                    atoms = self.random_struct(self.conc_matrix[x][y],
                                               conc1, conc2)
                    if atoms is None:
                        continue
                    atoms = wrap_and_sort_by_position(atoms)
                    kvp = CorrFunction(self.setting).get_cf(atoms)
                    kvp = self.get_kvp(atoms, kvp, conc1, conc2)
                    self.setting.db.write(atoms, kvp)

        # Case 3: 1 conc variable, user specified number of structures
        elif self.setting.num_conc_var == 1:
            num_conc1 = self.conc_matrix.shape[0]
            conc1_opt = range(num_conc1)
            x = 0
            while x < self.struct_per_gen:
                a = random.choice(conc1_opt)
                conc1 = float(a)/(num_conc1 - 1)
                atoms = self.random_struct(self.conc_matrix[a], conc1)
                if atoms is None:
                    continue
                atoms = wrap_and_sort_by_position(atoms)
                kvp = CorrFunction(self.setting).get_cf(atoms)
                kvp = self.get_kvp(atoms, kvp, conc1)
                self.setting.db.write(atoms, kvp)
                x += 1

        # Case 4: 2 conc variable, user specified number of structures
        else:
            num_conc1, num_conc2 = self.conc_matrix.shape[:2]
            conc1_opt = range(num_conc1)
            conc2_opt = range(num_conc2)
            x = 0
            while x < self.struct_per_gen:
                a = [random.choice(conc1_opt), random.choice(conc2_opt)]
                conc1 = float(a[0])/(num_conc1 - 1)
                conc2 = float(a[1])/(num_conc2 - 1)
                atoms = self.random_struct(self.conc_matrix[a[0]][a[1]],
                                           conc1, conc2)
                if atoms is None:
                    continue
                atoms = wrap_and_sort_by_position(atoms)
                kvp = CorrFunction(self.setting).get_cf(atoms)
                kvp = self.get_kvp(atoms, kvp, conc1, conc2)
                self.setting.db.write(atoms, kvp)
                x += 1
        return True

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

        init = wrap_and_sort_by_position(read(init_struct))
        conc = self._find_concentration(init)
        if self._exists_in_db(init, conc[0], conc[1]):
            raise RuntimeError('supplied structure already exists in DB')

        cf = CorrFunction(self.setting).get_cf(init)
        kvp = self.get_kvp(init, cf, conc[0], conc[1])

        if name is not None:
            kvp['name'] = name

        if final_struct is not None:
            energy = read(final_struct).get_potential_energy()
            calc= SinglePointCalculator(init, energy=energy)
            init.set_calculator(calc)
            kvp['converged'] = True
            kvp['started'] = ''
            kvp['queued'] = ''

        self.setting.db.write(init, key_value_pairs=kvp)

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
            conc = [round(float(x)/(num_conc1 - 1), 3), None]
        else:
            num_conc1, num_conc2 = self.conc_matrix.shape[:2]
            for x in range(num_conc1):
                for y in range(num_conc2):
                    if np.array_equal(conc_ratio, self.conc_matrix[x][y]):
                        break
                else:
                    continue
                break
            conc = [round(float(x)/(num_conc1 - 1), 3),
                    round(float(y)/(num_conc2 - 1), 3)]

        return conc

    def _exists_in_db(self, atoms, conc1=None, conc2=None):
        """Checks to see if the passed atoms already exists in DB.
        Return True if so, False otherwise.
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
        m = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=5,
                             primitive_cell=True, scale=True)
        s1 = AseAtomsAdaptor.get_structure(atoms)

        for row in self.setting.db.select(cond):
            atoms2 = row.toatoms()
            s2 = AseAtomsAdaptor.get_structure(atoms2)
            match = m.fit(s1, s2)
            if match:
                break
            # else:
            #     print('match = {}'.format(match))
        return match

    def get_kvp(self, atoms, kvp, conc1=None, conc2=None):
        """
        Receive atoms object, its correlation function (passed kvp) and
        value(s) of the concentration(s). Append key terms (conc, started, etc.)
        to kvp and returns it.
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
            n = len([row.id for row in self.setting.db.select(conc1=conc1)])
            kvp['name'] = 'conc_{:.3f}_{}'.format(conc1, n)
        else:
            conc2 = round(conc2, 3)
            kvp['conc2'] = conc2
            n = len([row.id for row in self.setting.db.select(conc1=conc1,
                                                              conc2=conc2)])
            kvp['name'] = 'conc_{:.3f}_{:.3f}_{}'.format(conc1, conc2, n)
        return kvp

    def random_struct(self, conc_ratio, conc1=None, conc2=None):
        """
        Generate a random structure that does not already exist in DB
        """
        # convert the conc_ratio into the same format as basis_elements if
        # setting.num_basis = 1 or setting.group_basis is None
        # Else, convert the conc_ratio the format of
        # setting.grouped_basis_elements
        if self.setting.num_basis == 1:
            conc_ratio = [list(conc_ratio)]
            num_basis = self.setting.num_basis
            basis_elements = self.setting.basis_elements
        else:
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
        while in_DB:
            atoms = self.atoms.copy()
            for site in range(num_basis):
                indx = [a.index for a in atoms if a.symbol ==
                        basis_elements[site][0]]
                if len(indx) != sum(conc_ratio[site]):
                    raise ValueError("number of atoms to be replaced in the "
                                     "Atoms object does not match the value "
                                     "in conc_ratio")
                for i in range(1, len(conc_ratio[site])):
                    indx = self._replace_rnd(indx,
                                             basis_elements[site][i],
                                             conc_ratio[site][i], atoms)
            in_DB = self._exists_in_db(atoms, conc1=conc1, conc2=conc2)
            # special case where only one species is present
            # break out of the loop and return atoms=None
            if in_DB and np.count_nonzero(flat_conc_ratio) == 1:
                atoms = None
                in_DB = False
        return atoms

    def _determine_gen_number(self):
        try:
            gens = [row.get('gen') for row in self.setting.db.select()]
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
