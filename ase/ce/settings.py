import os
import numpy as np
from ase.build import bulk
from itertools import combinations
from ase.db import connect
from ase.ce.tools import wrap_and_sort_by_position, index_by_position

class BulkCrystal(object):
    """
    Class that stores the necessary information about rocksalt structures.
    """
    def __init__(self, crystalstructure=None, alat=None, cell_dim=None,
                 num_sites=None, site_elements=None, conc_args=None, 
                 db_name=None, min_cluster_size=0, max_cluster_size=4,
                 max_cluster_dia=None, reconf_db=False):
        """
        crystalstructure: name of the crystal structure (e.g., 'fcc', 'sc')
        alat: lattice constant
        cell_dim: size of the supercell (e.g., [2, 2, 2] for 2x2x2 cell)
        num_sites: number of inequivalent sites
        site_elements: types of elements to reside in each inequivalent sites.
                (note) even if there is only one site, keep it in the list form
                       like [['Cu', 'Au']]
        conc_args: dictionary containing ratios of the elements for different
                   concentrations.
        db_name: name of the database file
        max_cluster_size: maximum size (number of atoms in a cluster)
        max_cluster_dia: maximum diameter of cluster (in unit of alat)
        reconf_db: boolean variable to indicate wheter or not to rebuild the 
                   information entry of the database
        """
        # ----------------------
        # Perform sanity checks
        # ----------------------
        # check for allowed structure types
        structures = ['sc', 'fcc', 'bcc', 'hcp', 'diamond', 'zincblende',
                      'rocksalt', 'cesiumchloride', 'fluorite', 'wurtzite']
        if crystalstructure is None:
            raise ValueError("Please specify 'crystalstructure' (e.g., 'fcc')")
        if crystalstructure not in structures:
            raise TypeError('Provided crystal structure is not supported.\n'
                            'The supported types are: {}'.format(structures))

        if type(alat) is not int and type(alat) is not float:
            raise TypeError("'alat' must be either int or float type.")
        if len(cell_dim) != 3:
            raise ValueError("Size of the cell needs to be specified for all"
                             " dimensions.")
        if len(site_elements) != num_sites:
            raise ValueError("list of elements is needed for each site")

        # check for concentration ratios 
        conc_names = ['conc_ratio_min_1', 'conc_ratio_min_2',
                      'conc_ratio_max_1', 'conc_ratio_max_2']
        conc_ratio_min_1 = None; conc_ratio_min_2 = None
        conc_ratio_max_1 = None; conc_ratio_max_2 = None
        for ratio_type, ratio in conc_args.items():
            if ratio_type not in conc_names:
                raise NameError('The name {} is not supported. \n'
                                'The allowed names are {}'\
                                .format(ratio_type, conc_names))
            if ratio_type == conc_names[0]:
                conc_ratio_min_1 = ratio
            elif ratio_type == conc_names[1]:
                conc_ratio_min_2 = ratio
            elif ratio_type == conc_names[2]:
                conc_ratio_max_1 = ratio
            else:
                conc_ratio_max_2 = ratio

        if (conc_ratio_min_1 is None or conc_ratio_max_1 is None):
            raise ValueError('Both min and max concentration ratios need be'
                             ' specified')
        if (conc_ratio_min_2 is not None and conc_ratio_max_2 is not None):
            num_conc_var = 2
        elif (conc_ratio_min_1 is None or conc_ratio_max_1 is None):
            raise ValueError('Both min and max concentration ratios need be'
                             ' specified')
        else:
            num_conc_var = 1

        # check dimensions of the element list and concentration ratio lists
        if num_sites == 1:
            if not(len(site_elements) == len(conc_ratio_min_1) ==
                   len(conc_ratio_max_1)):
                raise ValueError('lengths of the site_elements and conc_ratio'
                                 ' lists are not the same')
            if num_conc_var == 2:
                if not(len(site_elements) == len(conc_ratio_min_2) ==
                       len(conc_ratio_max_2)):
                    raise ValueError('lengths of the site_elements and'
                                     ' conc_ratio lists are not the same')
        else:
            element_size = [len(row) for row in site_elements]
            min_1_size = [len(row) for row in conc_ratio_min_1]
            max_1_size = [len(row) for row in conc_ratio_max_1]
            if not (element_size == min_1_size == max_1_size):
                raise ValueError('lengths of the site_elements and conc_ratio'
                                 ' lists are not the same')
            if num_conc_var == 2:
                min_2_size = [len(row) for row in conc_ratio_min_2]
                max_2_size = [len(row) for row in conc_ratio_max_2]
                if not (element_size == min_2_size == max_2_size):
                    raise ValueError('lengths of the site_elements and'
                                     ' conc_ratio lists are not the same')

        # -------------------------------
        # Passed tests. Assign parameters
        # -------------------------------
        self.crystalstructure = crystalstructure
        self.alat = float(alat)
        self.cell_dim = cell_dim
        self.site_elements = site_elements
        self.all_elements = [item for row in site_elements for item in row]
        self.num_elements = len(self.all_elements)
        self.num_conc_var = num_conc_var
        self.num_sites = num_sites
        self.conc_ratio_min_1 = conc_ratio_min_1
        self.conc_ratio_max_1 = conc_ratio_max_1
        if num_conc_var == 2:
            self.conc_ratio_min_2 = conc_ratio_min_2
            self.conc_ratio_max_2 = conc_ratio_max_2
        self.db_name = db_name
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.spin_dict = self.get_spin_values()
        self.basis_functions = self.get_basis_functions()
        self.atoms = self.create_atoms(max_cluster_dia)
        self.dist_matrix = self.create_distance_matrix()

        self.db = connect(db_name)
        if not os.path.exists(db_name):
            self.store_data()
        elif reconf_db:
            ids = [row.id for row in self.db.select(name='information')]
            self.db.delete(ids)
            self.store_data()
        else:
            try:
                row = self.db.get(name='information')
                self.cluster_names = row.data.cluster_names
                self.cluster_dist = row.data.cluster_dist
                self.cluster_indx = row.data.cluster_indx
                self.trans_matrix = row.data.trans_matrix
                self.site_elements = row.data.site_elements
                self.conc_matrix = row.data.conc_matrix
            except:
                ids = [row.id for row in self.db.select(name='information')]
                self.db.delete(ids)
                self.store_data()


    def atoms_with_given_dim(self):
        """
        Creates a template atoms object.
        """
        if self.crystalstructure == 'rocksalt':
            atoms = bulk('{}{}'.format(self.site_elements[0][0],
                                       self.site_elements[1][0]),
                         'rocksalt', a=self.alat, cubic=False)*self.cell_dim
        elif self.crystalstructure == 'fcc' or self.crystalstructure == 'bcc'\
             or self.crystalstructure == 'sc':
            atoms = bulk('{}'.format(self.site_elements[0][0]), 
                         '{}'.format(self.crystalstructure),
                         a=self.alat, cubic=False)*self.cell_dim
        else:
            # Not implemented yet.
            raise ValueError("Not implemented. Sorry~!")
        atoms = wrap_and_sort_by_position(atoms)
        return atoms

    def create_atoms(self, max_cluster_dia):
        """
        Create atoms object that can handle the maximum diameter specified by
        the user. If maximum diameter is not specified, the user-specified cell
        size will be used, and the maximum diameter is set accordingly.
        """
        # get the cell with the given dimensions
        atoms = self.atoms_with_given_dim()

        if max_cluster_dia is None:
            lengths = atoms.get_cell_lengths_and_angles()[:3]/self.alat
            self.max_cluster_dia = int(min(lengths)/2*1000)
        else:
            self.max_cluster_dia = int(max_cluster_dia/self.alat*1000)
            cell_lengths = atoms.get_cell_lengths_and_angles()[:3]\
                           /(2.*self.alat)*1000
            min_length = int(min(cell_lengths))
            if min_length < self.max_cluster_dia:
                scale_factor = []
                for x in range(3):
                    factor = np.round(self.max_cluster_dia/cell_lengths[x],
                                      decimals=1)
                    factor = max([factor, 1])
                    scale_factor.append(factor)
                scale_factor = np.ceil(scale_factor).astype(int)
                atoms = atoms*scale_factor
        atoms = wrap_and_sort_by_position(atoms)
        return atoms

    def store_data(self):
        print('Generating cluster data. It may take several minutes depending '
              'on the values of max_cluster_size and max_cluster_dia...')
        self.cluster_names,
        self.cluster_dist,
        self.cluster_indx = self.get_cluster_information()
        self.trans_matrix = self.create_translation_matrix()
        self.conc_matrix = self.create_concentration_matrix()
        self.db.write(self.atoms, name='information',
                      data={'cluster_names': self.cluster_names,
                            'cluster_dist': self.cluster_dist,
                            'cluster_indx': self.cluster_indx,
                            'trans_matrix': self.trans_matrix,
                            'site_elements': self.site_elements,
                            'conc_matrix': self.conc_matrix})
        return True

    def get_cluster_information(self):
        """
        Create a list of parameters used to describe the structure.
        """
        cluster_names = [['c0'], ['c1']]
        cluster_dist = [[None], [None]]
        cluster_indx = [[None], [None]]
        indices = [a.index for a in self.atoms]
        del indices[indices.index(0)]

        for n in range(2, self.max_cluster_size+1):
            indx_set = []
            dist_set = []
            # if the min_cluster_size is specified, the size up to the
            # min_cluster_size has None for distance and index.
            if n < self.min_cluster_size:
                cluster_dist.append([None])
                cluster_indx.append([None])
                cluster_names.append(['c{}'.format(n)])
                continue

            for i in combinations(indices, n-1):
                d = self.get_min_distance((0,)+i)
                if max(d) >= self.max_cluster_dia:
                    continue
                dist_set.append(d.tolist())
                indx_set.append(i)

            if np.version.version > '1.13':
                dist_types = np.unique(dist_set, axis=0).tolist()
            else:
                raise ValueError('Please use numpy version of 1.13 or higher')
            dist_types = sorted(dist_types, reverse=True)

            # categorieze the indices to the distance types it belongs
            indx_types = [[] for _ in range(len(dist_types))]
            for x in range(len(indx_set)):
                category = dist_types.index(dist_set[x])
                indx_types[category].append(indx_set[x])

            dia_set = [row[0] for row in dist_types]
            name_types = []
            counter = 1
            for i in range(len(dia_set)):
                if i == 0:
                    name_types.append('c{}_{}_{}'.format(n, dia_set[i], counter))
                    counter += 1
                    continue
                if dia_set[i] != dia_set[i-1]:
                    counter = 1
                name_types.append('c{}_{}_{}'.format(n, dia_set[i], counter))
                counter += 1
            cluster_dist.append(dist_types)
            cluster_indx.append(indx_types)
            cluster_names.append(name_types)

        return cluster_names, cluster_dist, cluster_indx

    def get_spin_values(self):
        # Find odd/even
        spin_values = []
        element_types = len(self.all_elements)
        if element_types % 2 == 1:
            highest = (element_types - 1)/2
        else:
            highest = element_types/2
        # Assign spin value for each element
        while highest > 0:
            spin_values.append(highest)
            spin_values.append(-highest)
            highest -= 1
        if element_types % 2 == 1:
            spin_values.append(0)

        spin_dict = {}
        for x in range(element_types):
            spin_dict[self.all_elements[x]] = spin_values[x]
        return spin_dict

    def get_basis_functions(self):
        """
        Create basis functions to guarantee the orthonormality.
        """
        if self.num_elements == 2:
            d0_0 = 1.
        elif self.num_elements == 3:
            d0_0 = np.sqrt(3./2)
            c0_1 = np.sqrt(2)
            c1_1 = -3/np.sqrt(2)
        elif self.num_elements == 4:
            d0_0 = np.sqrt(2./5)
            c0_1 = -5./3
            c1_1 = 2./3
            d0_1 = -17./(3*np.sqrt(10))
            d1_1 = np.sqrt(5./2)/3
        elif self.num_elements == 5:
            d0_0 = 1./np.sqrt(2)
            c0_1 = -1*np.sqrt(10./7)
            c1_1 = np.sqrt(5./14)
            d0_1 = -17./(6*np.sqrt(2))
            d1_1 = 5./(6*np.sqrt(2))
            c0_2 = 3*np.sqrt(2./7)
            c1_2 = -155./(12*np.sqrt(14))
            c2_2 = 15*np.sqrt(7./2)/12
        elif self.num_elements == 6:
            d0_0 = np.sqrt(3./14)
            c0_1 = -np.sqrt(2)
            c1_1 = 3./(7*np.sqrt(2))
            d0_1 = -7./6
            d1_1 = 1./6
            c0_2 = 9*np.sqrt(3./2)/5
            c1_2 = -101./(28*np.sqrt(6))
            c2_2 = 7./(20*np.sqrt(6))
            d0_2 = 131./(15*np.sqrt(4))
            d1_2 = -7*np.sqrt(7./2)/12
            d2_2 = np.sqrt(7./2)/20
        else:
            raise ValueError("only compounds consisting of 2 to 6 types of"
                             " elements are supported")

        bf_list = []

        bf = {}
        for key, value in self.spin_dict.items():
            bf[key] = d0_0 * value
        bf_list.append(bf)

        if self.num_elements > 2:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = c0_1 + (c1_1*value*value)
            bf_list.append(bf)

        if self.num_elements > 3:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = d0_1*value + (d1_1*(value**3))
            bf_list.append(bf)

        if self.num_elements > 4:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = c0_2 + (c1_2*(value**2)) + (c2_2*(value**4))
            bf_list.append(bf)

        if self.num_elements > 5:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = d0_2 + (d1_2*(value**3)) + (d2_2*(value**5))
            bf_list.append(bf)

        return bf_list

    def create_distance_matrix(self):
        num_atoms = len(self.atoms)
        dist = np.zeros((num_atoms, num_atoms, 8), dtype=int)
        indices = [a.index for a in self.atoms]
        vec = self.atoms.get_cell()
        trans = [[0., 0., 0.],
                 vec[0]/2,
                 vec[1]/2,
                 vec[2]/2,
                 (vec[0]+vec[1])/2,
                 (vec[0]+vec[2])/2,
                 (vec[1]+vec[2])/2,
                 (vec[0]+vec[1]+vec[2])/2]
        for t in range(8):
            shifted = self.atoms.copy()
            shifted.translate(trans[t])
            shifted.wrap()
            for x in range(num_atoms):
                temp = shifted.get_distances(x, indices)/self.alat*1000
                temp = np.round(temp)
                dist[x, :, t] = temp.astype(int)
        return dist

    def create_concentration_matrix(self):
        """
        Creates and returns a concentration matrix based on the value of
        conc_args.
        """
        min_1 = np.array([i for row in self.conc_ratio_min_1 for i in row])
        max_1 = np.array([i for row in self.conc_ratio_max_1 for i in row])
        if sum(min_1) != sum(max_1):
            raise ValueError('conc_ratio values must be on the same scale')
        natoms_cell = len(self.atoms_with_given_dim())
        natoms_ratio = sum(min_1)
        scale = natoms_cell/natoms_ratio
        min_1 *= scale
        max_1 *= scale
        diff_1 = [i - j for i, j in zip(max_1, min_1)]
        nsteps_1 = max(diff_1)
        increment_1 = diff_1/nsteps_1

        if self.num_conc_var == 1:
            conc = np.zeros((nsteps_1 + 1, len(min_1)), dtype=int)
            for n in range(nsteps_1 + 1):
                if n == 0:
                    conc[0] = min_1
                    continue
                conc[n] = conc[n-1] + increment_1

        if self.num_conc_var == 2:
            min_2 = np.array([i for row in self.conc_ratio_min_2 for i in row])
            max_2 = np.array([i for row in self.conc_ratio_max_2 for i in row])
            if sum(min_2) != sum(max_2):
                raise ValueError('conc_ratio values must be on the same scale')
            scale = natoms_cell/natoms_ratio
            min_2 *= scale
            max_2 *= scale
            diff_2 = [i - j for i, j in zip(max_2, min_2)]
            nsteps_2 = max(diff_2)
            increment_2 = diff_2/nsteps_2
            conc = np.zeros((nsteps_1+1, nsteps_2+1, len(min_1)), dtype=int)
            for _1 in range(nsteps_1 + 1):
                if _1 == 0:
                    conc[_1][0] = min_1
                else:
                    conc[_1][0] = conc[_1-1][0] + increment_1
                for _2 in range(1, nsteps_2 + 1):
                    conc[_1][_2] = conc[_1][_2-1] + increment_2
        return conc

    def create_translation_matrix(self):
        """
        Translation matrix to accelerate the computation of the correlation
        function.
        """
        num_atoms = len(self.atoms)
        tm = np.zeros((num_atoms, num_atoms), dtype=int)
        tm[0, :] = index_by_position(self.atoms)
        for x in range(1, num_atoms):
            shifted = self.atoms.copy()
            vec = self.atoms.get_distance(x, 0, vector=True)
            shifted.translate(vec)
            shifted.wrap()
            tm[x, :] = index_by_position(shifted)
        return tm

    def get_min_distance(self, cluster):
        d = []
        for t in range(8):
            row = []
            for x in combinations(cluster, 2):
                row.append(self.dist_matrix[x[0], x[1], t])
            d.append(sorted(row, reverse=True))
        return np.array(min(d))
