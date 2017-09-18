import numpy as np
from numpy.linalg import inv
import os
import random
from ase.ce.settings import BulkCrystal
from ase.ce.corrFunc import CorrFunction
from ase.ce.evaluate import Evaluate
from ase.ce.simulatedannealing import SimulatedAnnealing
from ase.ce.tools import wrap_and_sort_by_position, index_by_position
from ase.db import connect
from pymatgen.analysis.structure_matcher import StructureMatcher
from pmg import atoms_to_structure


class GenerateStructures(object):

    def __init__(self, BC, gen=None, struct_per_gen=None):
        if type(BC) is not BulkCrystal:
            raise TypeError("Passed object should be BulkCrystal type")
        self.BC = BC
        self.site_elements = BC.site_elements
        self.all_elements = BC.all_elements
        self.num_sites = len(self.site_elements)
        self.num_conc_var = BC.num_conc_var
        self.conc_matrix = BC.conc_matrix
        self.atoms = BC.atoms_with_given_dim()
        self.db_name = BC.db_name
        self.db = connect(self.db_name)
        if not os.path.exists(self.db_name):
            raise ValueError("DB file {} does not exist".format(self.db_name))
        if gen is None:
            self.gen = self.determine_gen_number()
        else:
            self.gen = gen

        # If the number of structures to include per generation is not given,
        # use one structure per computable concentration value
        # Note: May not be the most practical solution
        if struct_per_gen is None:
            if self.num_conc_var == 1:
                self.struct_per_gen = self.conc_matrix.shape[0]
            else:
                self.struct_per_gen = self.conc_matrix.shape[0] *\
                    self.conc_matrix.shape[1]
        else:
            self.struct_per_gen = struct_per_gen

        # Generate initial pool of structures for generation 0
        if self.gen == 0:
            self.generate_initial_pool()

    def determine_gen_number(self):
        try:
            gens = [row.gen for row in self.db.select()]
            gen = max(gens) + 1
        except AttributeError:
            gen = 0
        return gen

    def generate(self):
        """
        Generate a probe structure according to PRB 80, 165122 (2009)
        """
        while len([row.id for row in self.db.select(gen=self.gen)]) < self.struct_per_gen:
            # Pick an initial random structure
            if self.num_conc_var == 1:
                num_conc = self.conc_matrix.shape[0]
                conc = random.choice(range(num_conc))
                atoms = self.random_struct(self.conc_matrix[conc])
            else:
                num_conc1, num_conc2 = self.conc_matrix.shape[:2]
                conc = [random.choice(range(num_conc1)),
                        random.choice(range(num_conc2))]
                # to be removed
                while conc[0] < 15:
                    conc = [random.choice(range(num_conc1)),
                            random.choice(range(num_conc2))]
                atoms = self.random_struct(self.conc_matrix[conc[0]][conc[1]])

            atoms = wrap_and_sort_by_position(atoms)
            sa = SimulatedAnnealing(self.BC, atoms, self.struct_per_gen,
                                    init_temp=0.001, final_temp=0.00001, num_temp=5,
                                    num_steps=10000)
            atoms, cf = sa.generate_probe_structure()
            conc = self.find_concentration(atoms)
            if self.exists_in_db(atoms, conc[0], conc[1]):
                continue
            keys = self.get_key_values(atoms, cf, conc[0], conc[1], self.gen)
            self.db.write(atoms, keys)
            print(keys)
        return True

    def generate_initial_pool(self):
        if (self.num_conc_var == 1 and
                self.struct_per_gen == self.conc_matrix.shape[0]):
            pass

        elif (self.num_conc_var == 2 and
              self.struct_per_gen == self.conc_matrix.shape[0] *
              self.conc_matrix.shape[1]):
            for _1 in range(self.conc_matrix.shape[0]):
                conc1 = float(_1) / (self.conc_matrix.shape[0] - 1)
                for _2 in range(self.conc_matrix.shape[1]):
                    conc2 = float(_2) / (self.conc_matrix.shape[1] - 1)
                    atoms = self.random_struct(self.conc_matrix[_1][_2])
                    atoms = wrap_and_sort_by_position(atoms)
                    keys = CorrFunction(self.BC).get_cf(atoms)
                    keys = self.get_key_values(atoms, keys, conc1, conc2, gen=0)
                    self.db.write(atoms, keys)

        elif self.num_conc_var == 1:
            pass

        else:
            num_conc1, num_conc2 = self.conc_matrix.shape[:2]
            conc1_opt = range(num_conc1)
            conc2_opt = range(num_conc2)
            for x in range(self.struct_per_gen):
                a = [random.choice(conc1_opt), random.choice(conc2_opt)]
                conc1 = float(a[0]) / (num_conc1 - 1)
                conc2 = float(a[1]) / (num_conc2 - 1)
                while True:
                    atoms = self.random_struct(self.conc_matrix[a[0]][a[1]])
                    atoms = wrap_and_sort_by_position(atoms)
                    print('{} - check db: {} {}'.format(x, conc1, conc2))
                    if not self.exists_in_db(atoms, conc1, conc2):
                        break
                keys = CorrFunction(self.BC).get_cf(atoms)
                keys = self.get_key_values(atoms, keys, conc1, conc2, gen=0)
                self.db.write(atoms, keys)
        return True

    def find_concentration(self, atoms):
        conc_ratio = np.zeros(len(self.all_elements), dtype=int)
        for i, element in enumerate(self.all_elements):
            conc_ratio[i] = len([a for a in atoms if a.symbol == element])

        if self.num_conc_var == 1:
            pass
        else:
            num_conc1, num_conc2 = self.conc_matrix.shape[:2]
            for x in range(num_conc1):
                for y in range(num_conc2):
                    if np.array_equal(conc_ratio, self.conc_matrix[x][y]):
                        break
                else:
                    continue
                break
            conc1 = float(x) / (num_conc1 - 1)
            conc2 = float(y) / (num_conc2 - 1)
            conc = [conc1, conc2]

        return conc

    def exists_in_db(self, atoms, conc1=None, conc2=None):
        if conc1 is None:
            raise ValueError('conc1 needs to be defined')
        conc1 = round(conc1, 3)
        if conc2 is not None:
            conc2 = round(conc2, 3)
        # find if there is a match
        match = False
        m = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=5,
                             primitive_cell=True, scale=True)
        s1 = atoms_to_structure(atoms)
        for row in self.db.select(conc1=conc1, conc2=conc2):
            atoms2 = row.toatoms()
            s2 = atoms_to_structure(atoms2)
            match = m.fit(s1, s2)
            if match:
                break
        return match

    def get_key_values(self, atoms, keys, conc1=None, conc2=None, gen=None):
        if conc1 is None:
            raise ValueError('conc1 needs to be defined')
        if gen is None:
            raise ValueError('gen needs to be defined')
        # keys = CorrFunction(self.BC).get_cf(atoms)
        conc1 = round(conc1, 3)
        keys['conc1'] = conc1
        if conc2 is None:
            keys['conc2'] = conc2
        else:
            conc2 = round(conc2, 3)
            keys['conc2'] = conc2
        keys['gen'] = gen
        keys['converged'] = False
        keys['started'] = False
        keys['queued'] = False
        # Determine name
        n = len([row.id for row in self.db.select(conc1=conc1, conc2=conc2)])
        if conc2 is None:
            keys['name'] = 'conc_{:.3f}_{}'.format(conc1, n)
        else:
            keys['name'] = 'conc_{:.3f}_{:.3f}_{}'.format(conc1, conc2, n)
        return keys

    def random_struct(self, conc_ratio):
        atoms = self.atoms.copy()
        # convert the conc_ratio into the same format as site_elements
        if self.num_sites == 1:
            conc_ratio = [list(conc_ratio)]
        else:
            temp = list(conc_ratio)
            conc_ratio = []
            for site in range(self.num_sites):
                l = len(self.site_elements[site])
                conc_ratio.append(temp[:l])
                del temp[:l]

        for site in range(self.num_sites):
            indx = [a.index for a in atoms if a.symbol ==
                    self.site_elements[site][0]]
            if len(indx) != sum(conc_ratio[site]):
                raise ValueError("number of atoms to be replaced in the Atoms"
                                 " object does not match the value in"
                                 " conc_ratio")
            for i in range(1, len(conc_ratio[site])):
                indx = self._replace_rnd(indx, self.site_elements[site][i],
                                         conc_ratio[site][i], atoms)
        return atoms

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
        # remove "selected" from "indices"
        indices = [x for x in indices if x not in replace]
        return indices
