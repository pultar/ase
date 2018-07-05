"""Calculator for Cluster Expansion."""
import sys
from copy import deepcopy
import numpy as np
from ase.utils import basestring
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.ce import CorrFunction, BulkCrystal, BulkSpacegroup
from ase.ce.corrFunc import equivalent_deco


class MovedIgnoredAtomError(Exception):
    """Raised when ignored atoms is moved."""
    pass


class ClusterExpansion(Calculator):
    """Class for calculating energy using Cluster Expansion.

    Arguments
    =========
    setting: BulkCrystal or BulkSapcegroup object

    cluster_name_eci: dictionary of list of tuples containing
                      cluster names and ECI

    init_cf: (optional) correlation function of init_cf

    logfile: file object or str
        If *logfile* is a string, a file with that name will be opened.
        Use '-' for stdout.
    """

    name = 'ClusterExpansion'
    implemented_properties = ['energy']

    def __init__(self, setting, cluster_name_eci=None, init_cf=None,
                 logfile=None):
        Calculator.__init__(self)

        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            msg = "setting must be BulkCrystal or BulkSpacegroup object."
            raise TypeError(msg)
        self.setting = setting
        self.CF = CorrFunction(setting)

        # check cluster_name_eci and separate them out
        if isinstance(cluster_name_eci, list) and \
           (all(isinstance(i, tuple) for i in cluster_name_eci)
                or all(isinstance(i, list) for i in cluster_name_eci)):
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
            msg = "'cluster_name_eci' needs to be either (1) a list of tuples "
            msg += "or (2) a dictionary.\n They can be etrieved by "
            msg += "'get_cluster_name_eci' method in Evaluate class."
            raise TypeError(msg)

        # calculate init_cf or convert init_cf to array
        if init_cf is None:
            self.cf = init_cf
        elif isinstance(init_cf, list):
            if all(isinstance(i, (tuple, list)) for i in init_cf):
                cluster_names = [tup[0] for tup in init_cf]
                # cluster_name_eci and init_cf in the same order
                if cluster_names == self.cluster_names:
                    self.cf = np.array([tup[1] for tup in init_cf],
                                       dtype=float)
                # not in the same order
                else:
                    self.cf = []
                    for name in self.cluster_names:
                        indx = cluster_names.index(name)
                        self.cf.append(init_cf[indx][1])
                    self.cf = np.array(self.cf, dtype=float)
            else:
                self.cf = np.array(init_cf, dtype=float)
        elif isinstance(init_cf, dict):
            self.cf = np.array([init_cf[x] for x in self.cluster_names],
                               dtype=float)
        else:
            raise TypeError("'init_cf' needs to be either (1) a list "
                            "of tuples, (2) a dictionary, or (3) numpy array "
                            "containing correlation function in the same "
                            "order as the 'cluster_name_eci'.")

        if self.cf is not None and len(self.eci) != len(self.cf):
            raise ValueError('length of provided ECIs and correlation '
                             'functions do not match')

        # logfile
        if isinstance(logfile, basestring):
            if logfile == '-':
                logfile = sys.stdout
            else:
                logfile = open(logfile, 'a')
        self.logfile = logfile

        self.energy = None
        # reference atoms for calculating the cf and energy for new atoms
        self.ref_atoms = None
        self.ref_cf = None
        # old atoms for the case where one needs to revert to the previous
        # structure (e.g., Monte Carlo Simulation)
        self.old_atoms = None
        self.old_cf = None

    def set_atoms(self, atoms):
        self.atoms = atoms.copy()
        if self.cf is None:
            self.cf = self.CF.get_cf_by_cluster_names(self.atoms,
                                                      self.cluster_names,
                                                      return_type='array')
        self._copy_current_to_ref()
        self._copy_current_to_old()

        if len(self.setting.atoms) != len(atoms):
            msg = "Passed Atoms object and setting.atoms should have "
            msg += "same number of atoms."
            raise ValueError(msg)
        if not np.allclose(atoms.positions, self.setting.atoms.positions):
            msg = "atomic positions of all atoms in the passed Atoms "
            msg += "object and setting.atoms should be the same. "
            raise ValueError(msg)


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

    def restore(self, atoms):
        """Restore the old atoms and correlation functions to the reference."""
        self.ref_atoms = self.old_atoms.copy()
        self.ref_cf = deepcopy(self.old_cf)
        self.cf = deepcopy(self.old_cf)
        atoms.numbers = self.old_atoms.numbers

    def update_energy(self):
        """Update correlation function and get new energy."""
        self.update_cf()
        self.energy = self.eci.dot(self.cf) * len(self.atoms)
        return self.energy

    @property
    def indices_of_changed_atoms(self):
        """Return the indices of atoms that have been changed."""
        o_numbers = self.ref_atoms.numbers
        n_numbers = self.atoms.numbers
        check = (n_numbers == o_numbers)
        changed = np.argwhere(check == 0)[:, 0]
        changed = np.unique(changed)
        for index in changed:
            if self.atoms[index].symbol in self.setting.background_symbol:
                raise MovedIgnoredAtomError("Atom with index {} ".format(index)
                                            + "is a background atom.")

        return np.unique(changed)

    def get_cf_dict(self):
        """Return the correlation functions as a dict"""
        return dict(self.get_cf_list_tup())

    def get_cf_list_tup(self):
        """Return the correlation function as a list of tuples"""
        return zip(self.cluster_names, self.cf)

    def _symbol_by_index(self, indx):
        return [self.ref_atoms[indx].symbol, self.atoms[indx].symbol]

    def update_cf(self):
        """Update correlation function based on the reference value."""
        swapped_indices = self.indices_of_changed_atoms
        # print("swapped: {}".format(swapped_indices))
        self.cf = deepcopy(self.ref_cf)
        new_symbs = {}
        # Reset the atoms object
        for indx in swapped_indices:
            new_symbs[indx] = self.atoms[indx].symbol
            self.atoms[indx].symbol = self.ref_atoms[indx].symbol
            
        for indx in swapped_indices:
            # Swap one index at the time
            self.atoms[indx].symbol = new_symbs[indx]
            for i, name in enumerate(self.cluster_names):
                # find c{num} in cluster type
                n = int(name[1])
                if n == 0:
                    continue

                prefix = (name.rpartition('_')[0])
                dec_str = name.rpartition('_')[-1]
                dec = [int(x) for x in dec_str]

                if n == 1:
                    self.cf[i] = self.CF.get_c1(self.atoms, int(dec_str))
                    continue

                # Find which symmetry group the given atom (index) belongs to
                for symm in range(self.setting.num_trans_symm):
                    if indx in self.setting.index_by_trans_symm[symm]:
                        sg = symm
                        break

                # set name_indx and indices that compose a cluster
                try:
                    name_indx = self.setting.cluster_names[sg][n].index(prefix)
                # ValueError means that the cluster name (prefix) was not
                # found in the symmetry group of the index --> this cluster is
                # not altered.
                except ValueError:
                    continue
                indices = self.setting.cluster_indx[sg][n][name_indx]
                order = self.setting.cluster_order[sg][n][name_indx]
                equiv_sites = self.setting.cluster_eq_sites[sg][n][name_indx]

                # Get the total count
                count = 0
                for symm in range(self.setting.num_trans_symm):
                    try:
                        nindx = \
                            self.setting.cluster_names[symm][n].index(prefix)
                    except ValueError:
                        continue

                    clusters_per_atom = \
                        len(self.setting.cluster_indx[symm][n][nindx])
                    atoms_per_symm = \
                        len(self.setting.index_by_trans_symm[symm])
                    count += clusters_per_atom * atoms_per_symm

                t_indices = self._translate_indx(indx, indices)
                cf_tot = self.cf[i] * count
                cf_change = self._cf_change_by_indx(indx, t_indices, order,
                                                    equiv_sites, dec)
                self.cf[i] = (cf_tot + (n * cf_change)) / count



    def _cf_change_by_indx(self, ref_indx, trans_list, indx_order, eq_sites,
                           deco):
        """Calculate the change in correlation function based on atomic index.

        This method tracks changes in correaltion function due to change in
        element type for atom with index = ref_indx. Passed trans_list refers
        to the indices of atoms that constitute the cluster
        (after translation).
        """
        symbol = self._symbol_by_index(ref_indx)
        b_f = self.setting.basis_functions
        delta_cf = 0.
        equiv_deco = equivalent_deco(deco, eq_sites)
        for dec in equiv_deco:
            for cluster_indx, order in zip(trans_list, indx_order):
                # NOTE: Here cluster_indx is already translated!
                indices = [ref_indx] + cluster_indx
                indices = [indices[indx] for indx in order]
                cf_new = 1.0
                cf_ref = 1.0
                for j, indx in enumerate(indices):
                    if indx == ref_indx:
                        cf_new *= b_f[dec[j]][symbol[1]]
                        cf_ref *= b_f[dec[j]][symbol[0]]
                    else:
                        cf_new *= b_f[dec[j]][self.atoms[indx].symbol]
                        cf_ref *= b_f[dec[j]][self.atoms[indx].symbol]
                delta_cf += (cf_new - cf_ref)/len(equiv_deco)
        return delta_cf

    def _translate_indx(self, ref_indx, indx_list):
        tlist = deepcopy(indx_list)
        for i in range(len(indx_list)):
            for j in range(len(indx_list[i])):
                tlist[i][j] = \
                    self.setting.trans_matrix[ref_indx][indx_list[i][j]]
        return tlist

    def _check_atoms(self, atoms):
        """Check to see if the passed atoms argument valid.

        This method checks that:
            - atoms argument is Atoms object,
            - atoms has the same size and atomic positions as
                (1) setting.atoms,
                (2) reference Atoms object.
        """
        if not isinstance(atoms, Atoms):
            raise TypeError('Passed argument is not Atoms object')
        if len(self.ref_atoms) != len(atoms):
            raise ValueError('Passed atoms does not have the same size '
                             'as previous atoms')
        if not np.allclose(self.ref_atoms.positions, atoms.positions):
            raise ValueError('Atomic postions of the passed atoms are '
                             'different from init_atoms')

    def log(self):
        """Write energy to log file."""
        if self.logfile is None:
            return True
        self.logfile.write('{}\n'.format(self.energy))
        self.logfile.flush()
