"""Calculator for Cluster Expansion."""
import sys
from itertools import product
from copy import deepcopy
import numpy as np
from ase.utils import basestring
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.ce.corrFunc import CorrFunction
from ase.ce.settings import BulkCrystal


class ClusterExpansion(Calculator):
    """Class for calculating energy using Cluster Expansion.

    Arguments
    =========
    settings: object that contains CE settings (e.g., BulkCrystal)

    init_atoms: Atoms object containing the initial structure

    cluster_name_eci: dictionary of list of tuples containing
                      cluster names and ECI

    init_cf: (optional) correlation function of init_cf

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.
    """

    name = 'ClusterExpansion'
    implemented_properties = ['energy']

    def __init__(self, settings, init_atoms, cluster_name_eci=None,
                 init_cf=None, logfile=None):
        Calculator.__init__(self)

        if not isinstance(settings, BulkCrystal):
            raise TypeError("settings should be BulkCrystal object")
        self.settings = settings
        self.CF = CorrFunction(settings)

        # check cluster_name_eci and separate them out
        if isinstance(cluster_name_eci, list) and \
           (all(isinstance(i, tuple) for i in cluster_name_eci) or
            all(isinstance(i, list) for i in cluster_name_eci)):
            self.cluster_names = [tup[0] for tup in cluster_name_eci]
            self.eci = np.array([tup[1] for tup in cluster_name_eci])
        elif isinstance(cluster_name_eci, dict):
            self.cluster_names = []
            self.eci = []
            for cluster_name, eci in cluster_name_eci.items():
                self.cluster_names.append(cluster_name)
                self.eci.append(eci)
            self.eci = np.array(self.eci)
        else:
            raise TypeError("'cluster_name_eci' needs to be either (1) a list "
                            "of tuples or (2) a dictionary. They can be "
                            "retrieved by 'get_cluster_name_eci_tuple' or "
                            "'get_cluster_name_eci_dict' methods in Evaluate "
                            "class")

        # calculate init_cf or convert init_cf to array
        if init_cf is None:
            self.cf = init_cf
        elif isinstance(init_cf, list):
            if all(isinstance(i, (tuple, list)) for i in init_cf):
                # cluster_names = [tup[0] for tup in init_cf]
                # if all()
                self.cf = np.array([tup[1] for tup in init_cf], dtype=float)
            else:
                self.cf = np.array(init_cf, dtype=float)
        elif isinstance(init_cf, dict):
            self.cf = np.array([init_cf[x] for x in self.cluster_names],
                               dtype=float)
        else:
            raise TypeError("'init_cf' needs to be either (1) a list "
                            "of tuples, (2) a dictionary, or (3) numpy array "
                            "containing correlation function in the same order "
                            "as the 'cluster_name_eci'.")

        if (self.cf is not None and len(self.eci) != len(self.cf)):
            raise ValueError('length of provided ECIs and correlation '
                             'functions do not match')

        # logfile
        if isinstance(logfile, basestring):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

        # reference atoms for calculating the cf and energy for new atoms
        self.ref_atoms = None
        self.ref_cf = None
        self.ref_energy = None
        # old atoms for the case where one needs to revert to the previous
        # structure (e.g., Monte Carlo Simulation)
        self.old_atoms = None
        self.old_cf = None
        self.old_energy = None

    # def calculate(self, atoms, accept=True):
    def calculate(self, atoms, properties, system_changes):
        """Calculate the energy of the passed atoms object.

        If accept=True, the most recently used atoms object is used as a
        reference structure to calculate the energy of the passed atoms.
        Returns energy.
        """
        self._check_atoms(atoms)
        Calculator.calculate(self, atoms)
        self.update_energy()
        self.results['energy'] = self.energy
        self.log()
        if self.old_atoms is None:
            # first calculation
            self._copy_current_to_old()
        else:
            self._copy_ref_to_old()
        self._copy_current_to_ref()

        return self.energy

    def _copy_current_to_ref(self):
        self.ref_atoms = self.atoms.copy()
        self.ref_cf = deepcopy(self.cf)

    def _copy_current_to_old(self):
        self.old_atoms = self.atoms.copy()
        self.old_cf = deepcopy(self.cf)

    def _copy_ref_to_old(self):
        self.old_atoms = self.ref_atoms.copy()
        self.old_cf = deepcopy(self.ref_cf)

    def restore(self):
        print('restore')
        self.ref_atoms = self.old_atoms.copy()
        self.ref_cf = deepcopy(self.old_cf)

    def update_energy(self):
        """Update correlation function and get new energy."""
        # this is the first run
        if self.ref_atoms is None:
            print('first time')
            if self.cf is None:
                self.cf = self.CF.get_cf_by_cluster_names(self.atoms,
                                                          self.cluster_names,
                                                          return_type='array')
        else:
            self.update_cf()
        self.energy = self.eci.dot(self.cf) * len(self.atoms)
        return self.energy

    @property
    def indices_of_changed_atoms(self):
        """Returns the indices of atoms that have been changed."""
        o_numbers = self.ref_atoms.numbers
        n_numbers = self.atoms.numbers
        check = (n_numbers == o_numbers)
        changed = np.argwhere(check == 0)[:, 0]
        print(changed)
        return np.unique(changed)

    def _symbol_by_index(self, indx):
        return [self.ref_atoms[indx].symbol, self.atoms[indx].symbol]

    def update_cf(self):
        swapped_indices = self.indices_of_changed_atoms

        # if swapped_indices is empty, no changes are made to the atoms
        if len(swapped_indices) == 0:
            return deepcopy(self.old_cf)

        bf_list = list(range(len(self.settings.basis_functions)))
        self.cf = deepcopy(self.old_cf)

        for indx in swapped_indices:
            for i, name in enumerate(self.cluster_names):
                if name == 'c0':
                    continue

                dec = int(name.rpartition('_')[-1]) - 1
                prefix = (name.rpartition('_')[0])

                if prefix == 'c1':
                    self.cf[i] = self.CF.get_c1(self.atoms, dec)
                    continue
                # find c{num} in cluster type
                num = int(prefix[1])
                # find the type of cluster based on the index of the original
                # settings.cluster_names nested list (unflattened)
                ctype = self.settings.cluster_names[num].index(prefix)

                perm = list(product(bf_list, repeat=num))
                i_list = self.settings.cluster_indx[num][ctype]
                t_list = self._translated_indx(indx, i_list)

                # ----------------------------------
                # This only works for a single basis
                # ----------------------------------
                count = len(i_list) * len(self.atoms)
                cf_tot = self.cf[i] * count
                cf_change = self._cf_change_by_indx(indx, t_list, perm[dec])

                # If the decoration number of all the elements in the cluster
                # is the same, it suffices to change only the correlation
                # function of the given index. If not, it is also necessary to
                # change the correlation function of its members' clusters
                # which contain the element changed.
                # ------------------------------------------------------------
                # Check the order in which decoration numbers are assigned. It
                # is possible that correct ordering might solve the problem.
                # ------------------------------------------------------------
                if len(set(perm[dec])) == 1:
                    self.cf[i] = cf_tot + (num * cf_change)
                else:
                    self.cf[i] = cf_tot + cf_change
                    members = np.unique(t_list)
                    for mindx in members:
                        # only count correlation function of the clusters that
                        # contain the changed atom
                        t_l = self._translated_indx(mindx, i_list)
                        t_list = t_l[~np.all(t_l != indx, axis=1)]
                        cf_change = self._cf_change_by_indx(mindx, t_list,
                                                            perm[dec])
                        self.cf[i] += cf_change
                self.cf[i] = self.cf[i] / count
        return True

    def _cf_change_by_indx(self, ref_indx, trans_list, dec):
        """
        Calculates the change in correlation function due to change in element
        type for atom with index = ref_indx. Passed trans_list refers to the
        indices of atoms that constitute the cluster (after translation).
        """
        symbol = self._symbol_by_index(ref_indx)
        b_f = self.settings.basis_functions
        delta_cf = 0.
        for cluster_indx in trans_list:
            cf_new = b_f[dec[0]][symbol[1]]
            cf_ref = b_f[dec[0]][symbol[0]]
            for j, indx in enumerate(cluster_indx):
                cf_new *= b_f[dec[j + 1]][self.atoms[indx].symbol]
                cf_ref *= b_f[dec[j + 1]][self.ref_atoms[indx].symbol]
            delta_cf += cf_new - cf_ref
        return delta_cf

    def _translated_indx(self, ref_indx, indx_list):
        tlist = deepcopy(indx_list)
        for i, cluster_indx in enumerate(indx_list):
            for j, indx in enumerate(cluster_indx):
                # ----------------------------------
                # This only works for a single basis
                # ----------------------------------
                tlist[i][j] = self.settings.trans_matrix[ref_indx, indx]
        return tlist

    def _check_atoms(self, atoms):
        """Check to see if the passed atoms argument is Atoms object with
        the same number of atoms and positions as the previous one."""
        if not isinstance(atoms, Atoms):
            raise TypeError('Passed argument is not Atoms object')
        if self.old_atoms is None:
            if self.settings.in_conc_matrix(atoms):
                self.atoms = self.CF.check_and_convert_cell_size(atoms)
            else:
                raise ValueError("Provides atoms object does not seem valid "
                                 "based on concentration matrix. Please check "
                                 "that the passed atoms and settings are "
                                 "consistent.")
        else:
            if len(self.ref_atoms) != len(atoms):
                raise ValueError('Passed atoms does not have the same size '
                                 'as previous atoms')
            elif not np.allclose(self.ref_atoms.positions, atoms.positions):
                raise ValueError('Atomic postions of the passed atoms are '
                                 'different from init_atoms')

    def log(self):
        """Write energy to log file"""
        if self.logfile is None:
            return True
        self.logfile.write('{}\n'.format(self.energy))
        self.logfile.flush()
