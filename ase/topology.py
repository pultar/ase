import numpy as np
from copy import deepcopy
from ase.utils import basestring
from ase.neighborlist import NeighborList, natural_cutoffs


# ! TODO update 'names' array to the future ASE labels
def string2index(string):
    """converts sring to index
    cannot be imported from ase.io.formats."""
    if ':' not in string:
        return int(string)
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)


class _TopoAttribute(object):
    """
    Class to handle individual topology attribute like bonds, angles,
    impropers, dihedrals etc. It includes methods that makes handling the
    attribute easy to work with. Since, all these attributes have mostly the
    same data-structure, it is easily handled in this one class.

    It is attached to topology class as individual property, eg.
    topology.bonds, or topology.angles etc
    """

    def __init__(self, topo_attr_prop):
        """
        A simple initialisation that stores the name of the attribute
        this instance of class will represent, and the atoms object that
        holds the data.
        Parameters
            topo_attr_prop: _TopoAttributeProperty object
                contains pointer to the atoms object and name of property
        """
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = topo_attr_prop._ins
        self.prop = topo_attr_prop.prop

    def _check_exists(func):
        """Decorator to check if the property exists
        Since topology defines all properties already, like topology.bonds
        it is necessary to add this check to most get functions."""
        # Only added to functions which need already existing values in
        # atoms._topology, otherwise the counting methods return 0
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.prop not in self._ins._topology:
                raise RuntimeError('{0} object has no '
                                   '{1}'.format(self._ins.__class__.__name__,
                                                self.prop))
            return func(*args, **kwargs)
        return wrapper

    def __repr__(self):
        return str(self.get_types())

    @_check_exists
    def __getitem__(self, item):
        """Returns property by name, shortens the regular call
        >>> water = Atoms('H2O', positions=[[0, 0, 0],
        ...                                 [1, 1, 0],
        ...                                 [-1, -1, 0]]
        ...              )
        >>> water.set_topology()
        >>> water.topology.generate(bonds = [['H', 'O']],
         ...                        angles = [['H', 'O', 'H']]
         ...                       )
        >>> water.topology.angles(with_names=True)['H-O-H']
        >>> water.topology.angles['H-O-H']
        """
        props = self.get(with_names=True)
        if item not in props:
            raise RuntimeError('{} not in {}'.format(item, self.prop))
        return props[item]

    @_check_exists
    def __delitem__(self, items):
        """
        deletes attribute by their bond, angle, improper or dihedral name
        """
        if not isinstance(items, list):
            items = [items]
        types = self.get_types(verbose=True)
        array = self._ins._topology[self.prop]
        # a mask to get only needed properties
        mask = np.ones(len(array), dtype=int)
        for item in items:
            # if item does not exists then nothing is changed
            mask[types == item] = 0
        array = array[mask]
        self._ins._topology[self.prop] = array
        self.update()

    def get_num_types(self):
        """ returns number of types of prop: bonds, etc"""
        return len(self.get_types())

    def get_types(self, index=':', verbose=False):
        """returns types of prop: bonds, etc
        :param index: the index of atoms to affect
        :param verbose: if true, returns types of individual bond/angles/etc"""
        if self.prop not in self._ins._topology:
            return np.array([])

        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass

        items = self._ins._topology[self.prop]
        items_indx = []
        _ = np.arange(len(self._ins))[index]
        for connectivity in items:
            # connectivity is a list of indices
            if np.all([x in _ for x in connectivity]):
                # the connectivity is acceptible
                items_indx.append(connectivity)

        # giving connectivity names
        names = self._ins.arrays['names']
        type_list = [None for _ in range(len(items_indx))]
        for i, connectivity in enumerate(items_indx):
            name_list = names[connectivity].tolist()
            if self.prop == 'bonds':
                name_list.sort()
            elif self.prop == 'angles':
                # vertex atom is at second index
                vertex = name_list.pop(1)
                name_list.sort()
                name_list.insert(1, vertex)
            type_list[i] = '-'.join(name_list)

        if verbose:
            return np.array(type_list)
        else:
            return np.unique(type_list)

    @_check_exists
    def get(self, with_names=False):
        """
        Parameters
            with_names: bool
                returns bonds/angles/dihedrals/impropers as dict of
                connectivity name as key and list of indices of
                connectivities"""
        if not with_names:
            return self._ins._topology[self.prop]
        array = self._ins._topology[self.prop]
        types = self.get_types(verbose=True)
        dict_ = {}
        for i, connectivity in enumerate(array):
            dict_[types[i]] = (dict_.get(types[i], [])
                               + [connectivity.tolist()])
        return dict_

    __call__ = get

    def set(self, value=None):
        """
        value to set. None value deletes the property
        """
        if self.prop in self._ins._topology:
            # delete key in _topology
            del self._ins._topology[self.prop]
        if value is None:
            return None

        self.add(value)
        self.update()

    def get_count(self):
        """ returns number of prop: bonds, etc."""
        if self.prop not in self._ins._topology:
            return 0
        return len(self._ins._topology[self.prop])

    def add(self, items, _offset=None):
        """ adds to prop
        Parameters:
            items: array
                Array of shape (num_prop, x), where num prop is number of
                of bonds/angles/dihedrals/impropers, and x is size of
                connectivity.
            _offset: int
                Added to items to offset indices. Useful when adding atoms
                objects
        """
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        # if items is empty return
        if len(items) == 0:
            return

        # make sure items is int and of correct size
        items = np.array(items, dtype=int)
        if (np.any([len(np.unique(item)) != length[self.prop]
                   for item in items]) or len(items.shape) != 2):
            raise RuntimeError('{0} should be unique indices of size '
                               '{1}'.format(self.prop, length[self.prop]))

        if _offset is not None:
            items += _offset

        # make sure indices are within length of atoms object
        if np.any([x >= len(self._ins) for x in items.reshape(-1)]):
            raise RuntimeError('indices should be lower than length of '
                               '{}'.format(self._ins.__class__.__name__))

        if self.prop not in self._ins._topology:
            array = []
        else:
            array = self._ins._topology[self.prop].tolist()

        # check if connectivity already exists
        for item in items:
            item = item.tolist()
            if self.prop == 'bonds':
                # makes bonds unique
                item.sort()
            elif self.prop == 'angles':
                # makes angles unique
                indx = item.pop(1)
                item.sort()
                item.insert(1, indx)
            elif self.prop == 'dihedrals':
                # makes dihedrals unique
                if item[0] > item[3]:
                    item.reverse()
            if item not in array:
                array.append(item)
        self._ins._topology[self.prop] = np.array(array, dtype=int)
        self.update()

    # no _check_* since its called by functions
    # non error raising methods of checking implemented
    def update(self):
        """
        Updates the _topology by removing empty properties
        """
        # Correct for empty connectivity
        if self.prop in self._ins._topology:
            if len(self._ins._topology[self.prop]) == 0:
                # prop is empty
                del self._ins._topology[self.prop]

    def _set_indices_to(self, indx_of):
        """
        Sets indices according to indx_of. It doesn't call update, the function
        calling it is responsible for calling update.
        Parameters
            ind_of: dictionary
                Keys are indices to mapped to the value. If value is None
                then delete the index
        """
        # selecting set of ids to remove
        if self.prop in self._ins._topology:
            array = self._ins._topology[self.prop]
            # mask to choose only needed connectivity
            mask = np.ones(len(array), dtype=bool)
            for i in range(len(array)):
                # if indx_of any index in connectivity is None,
                # then remove the connectivity
                if np.any([indx_of[x] is None for x in array[i]]):
                    mask[i] = 0
                else:
                    # else change indices as indicated in indx_of
                    for j in range(array.shape[1]):
                        array[i, j] = indx_of[array[i, j]]
            array = array[mask]
            self._ins._topology[self.prop] = array

    @_check_exists
    def get_statistics(self, index=':'):
        """
        Returns statistics for specific attribute
        Parameters
            index: indices to return statistics for
        Returns
            Dictionary with bond/angles/dihedral names with a
            list of values
        """
        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass
        index = np.arange(len(self._ins))[index]

        if self.prop == 'bonds':
            func = self._ins.get_distance
        elif self.prop == 'angles':
            func = self._ins.get_angle
        elif self.prop == 'dihedrals':
            func = self._ins.get_dihedral
        elif self.prop == 'impropers':
            NotImplementedError('Statistics not implemented for {}'
                                '',format(self.prop))
        else:
            RuntimeError('{} has no statistics',format(self.prop))

        #! TODO: return named list instead of dictionary
        stats = {}
        for key, values in self.get(with_names=True).items():
            for value in values:
                if np.all([x in index for x in value]):
                    stats[key] = (stats.get(key, [])
                                  + [func(*value, mic=True)])
        return stats


class _TopoAttributeProperty(object):
    """Modified python property to handle passing of atoms object and
    attribute name"""

    def __init__(self, prop):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self.prop = prop

    def __get__(self, topo_base, owner):
        """
        Parameters
            topo_base: the object that this property is attached to
        Returns
            An instance of _TopoAttribute that handles the attribute
        """
        if topo_base is None:
            return self
        self._ins = topo_base._ins
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        return _TopoAttribute(self)

    def __set__(self, topo_base, value):
        self._ins = topo_base._ins
        _TopoAttribute(self).set(value)

    def __delete__(self, topo_base):
        self.__set__(topo_base, None)



class Topology(object):
    """
    Adds topology to atoms object. The topology saves bonds, angles,
    dihedrals and impropers. These are saved as numpy arrays in
    atoms._topology dictionary.
    On initialisation through atoms object, names are auto initiated
    and the connectivity is initiated by the user.
    """
    resnames = _TopoAttributeProperty('resnames')
    bonds = _TopoAttributeProperty('bonds')
    angles = _TopoAttributeProperty('angles')
    dihedrals = _TopoAttributeProperty('dihedrals')
    impropers = _TopoAttributeProperty('impropers')

    def __init__(self, instance, _persistent=False):
        """
        To initialise the topology.
        Parameters
            instance: Atoms object
            _persistent: bool, default False
                To indicate if the topology is persistent. If True, then the
                internal atoms object points to the passed atoms object, and
                updates are reflected in the passed atoms object. If False,
                then the topology class keeps its own copy of atoms object,
                and the passed atoms object is not affected.
        """
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive

        # a bool to see if the topology is persistent
        self._persistent = _persistent
        if self._persistent:
            # the instance is edited along with topology
            self._ins = instance
            if self._ins._topology is None:
                # no topology but atoms.topology called
                _ = self._ins.__class__.__name__
                raise RuntimeError('Topology not initialised;'
                                   ' use {atoms}.set_topology() to set '
                                   'a persistent topology, else use '
                                   '{atoms}.get_topology()'
                                   ''.format(atoms=_))
        else:
            # topology object gets its own atoms object
            self._ins = instance.copy()

        # a dict to hold all attributes
        # for easy access by loop functions
        self._dict = {'bonds': self.bonds,
                      'angles': self.angles,
                      'dihedrals': self.dihedrals,
                      'impropers': self.impropers}

        if not self._persistent:
            # persistent topology is only updated when
            # a function called by user needs it
            # this avoids multiple updating, as the topology object
            # is thrown away when not needed
            # non-persistent topology has a new atoms object
            # that should be updated here
            if self._ins._topology is None:
                # add topology for its own atoms object
                self._ins._topology = {}
            # requires _dict to be defined
            self.update()

    def __repr__(self):
        tokens = []
        for key, values in self._dict.items():
            if self.has(key):
                tokens.append("{}= {}".format(key, values))
        if self._persistent:
            return "{}.Topology({})".format(self._ins.__class__.__name__,
                                            ", ".join(tokens))
        else:
            return "Topology({})".format(", ".join(tokens))

    def has(self, prop):
        """
        Checks if prop is in _topology
        """
        return prop in self._ins._topology

    def __getitem__(self, item):
        return self._dict[item]

    def get_perstisting_atoms_object(self):
        '''Returns the atoms object that is being edited using topology
        if the topology is non-persistent, then this object has the topology
        generated with this topology object'''
        return self._ins

    def get_topology_dict(self):
        """Gives topology dict that can be inherited by other topology
        classes"""
        return deepcopy(self._ins._topology)

    __call__ = get_topology_dict

    def update(self, topo_dict=None):
        """
        Updates all topology attributes, reforms tags, and add names if
        they don't exists
        Parameters
            topo_dict: dictionary
                a dictionary of topology, with bonds, angles etc as keys, and
                connectivities as values.
        """
        if topo_dict is None:
            topo_dict = {}

        # add names if does not exist
        if not self._ins.has('names'):
            self._ins.set_array('names',
                                self._ins.get_chemical_symbols(),
                                object)

        for prop in self._dict.keys():
            if prop in topo_dict:
                self._dict[prop].set(topo_dict[prop])
            elif prop in self._ins._topology:
                self._dict[prop].update()

    def generate(self,
                 bonds=None,
                 angles=None,
                 dihedrals=None,
                 impropers=None,
                 cutoffs=None):
        """
        Generates bonds, angles, dihedrals, and impropers based on names
        of atoms. If None are passed, then all are generated with
        neighbour data

        Parameters:
            bonds, angles, dihedrals, impropers: list
                list of connectivities. The connectivities can be all int
                indicating connection between indices or all str, indicating
                connection between the atom names.
            cutoffs: float array with cuttoffs of each atom in atoms object

        Example:

        >>> water = Atoms('H2O', positions=[[0, 0, 0],
        ...                                 [1, 1, 0],
        ...                                 [-1, -1, 0]]
        ...              )
        >>> water.set_topology()
        >>> water.topology.generate(bonds = [['H', 'O']],
         ...                        angles = [['H', 'O', 'H']]
         ...                       )
        """
        def update_topo_dict_indx(prop, list_):
            '''updates topo_dict_indx for list of connectivity
            Parameters
                prop: str
                    name of connectivity
                list_: list of int
                    indices to be added to topo_dict_indx
            '''
            name_list = [symbols[x] for x in list_]
            if prop == 'bonds':
                name_list.sort()
            elif prop == 'angles':
                vertex = name_list.pop(1)
                name_list.sort()
                name_list.insert(1, vertex)
            name_list = '-'.join(name_list)
            if (name_list in topo_names.get(prop, [])
                    or generate_all):
                if prop not in topo_dict_indx:
                    # initiate topo_dict_indx
                    topo_dict_indx[prop] = np.array([list_],
                                                       dtype=int)
                    return
                _ = np.vstack([topo_dict_indx[prop], list_])
                topo_dict_indx[prop] = _

        # dict to hold str connectivities as lists
        topo_dict = {}
        # dict to hold int connectivities as lists
        topo_dict_indx = {}
        # dict to hold str connectivities joined by -
        topo_names = {}
        topo_prop = {'bonds': bonds,
                     'angles': angles,
                     'dihedrals': dihedrals,
                     'impropers': impropers}
        for name_, prop in topo_prop.items():
            if prop is not None:
                if len(prop) != 0:
                    if len(np.array(prop).shape) != 2:
                        raise RuntimeError('a list of connectivities is expected')
                    topo_dict[name_] = prop

        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        if (bonds is None
                and angles is None
                and dihedrals is None
                and impropers is None):
            # empty generate passed, all connectivities to be added
            generate_all = True
        else:
            # move connectivities based on str or indices
            generate_all = False
            for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
                if prop not in topo_dict:
                    continue
                if np.all([isinstance(y, (int, np.integer))
                           for x in topo_dict[prop] for y in x]):
                    # if indices are given
                    array = np.array(topo_dict[prop], dtype=int)
                    # check for length
                    if array.shape[1] != length[prop]:
                        raise RuntimeError('{} should be of length {}'
                                           ''.format(prop, length[prop]))
                    # and move them to topo_dict_indx
                    topo_dict_indx[prop] = array
                    topo_dict.pop(prop)
                elif np.all([isinstance(y, (str, np.str_))
                           for x in topo_dict[prop] for y in x]):
                    # else str given
                    topo_dict[prop] = np.array(topo_dict[prop])
                    # check for length
                    if topo_dict[prop].shape[1] != length[prop]:
                        raise RuntimeError('{} should be of length {}'
                                           ''.format(prop, length[prop]))
                    # and add topo_dict connectivity as single string
                    # in topo_names
                    for key, values in topo_dict.items():
                        for i, value in enumerate(values):
                            if key == 'bonds':
                                value = np.sort(value)
                            if key == 'angles':
                                edges = np.sort([value[0], value[2]])
                                value[0], value[2] = edges
                            topo_names[key] = (topo_names.get(key, [])
                                                    + ['-'.join(value)])
                else:
                    raise RuntimeError('indices should be either all int or str')

        # now convert all str connectivities to int connectivities
        if topo_names or generate_all:

            # making symmetric neighbor matrix
            if cutoffs is None:
                cutoffs = natural_cutoffs(self._ins)
            elif len(cutoffs) != len(self._ins):
                raise RuntimeError("length of cutoffs doesn't match "
                                   "the number of atoms")
            nl = NeighborList(cutoffs,
                              self_interaction=False,
                              bothways=True)
            nl.update(self._ins)
            neighbor_matrix = nl.get_connectivity_matrix().toarray()

            symbols = self._ins.get_array('names')
            d = []
            for i, array in enumerate(neighbor_matrix):
                neighbor_list = np.where(array)[0]
                d.append(neighbor_list)

            if 'bonds' in topo_names or generate_all:
                for i, neighbor in enumerate(d):
                    for j in neighbor:
                        if i <= j:
                            # removes double counting
                            continue
                        update_topo_dict_indx('bonds', [i, j])

            if 'angles' in topo_names or generate_all:
                for i, neighbor in enumerate(d):
                    for indx, j in enumerate(neighbor):
                        for k in neighbor[indx + 1:]:
                            update_topo_dict_indx('angles', [j, i, k])

            if 'dihedrals' in topo_names or generate_all:
                # sends i-j-k-l, and l-k-j-i. only the order given in generate
                # is accepted
                for i, neighbor_i in enumerate(d):
                    for j in neighbor_i:
                        for k in set(d[j]) - {i, j}:
                            for l in set(d[k]) - {i, j, k}:
                                update_topo_dict_indx('dihedrals',
                                                      [i, j, k, l])

            if 'impropers' in topo_names or generate_all:
                for i, neighbor in enumerate(d):
                    for indx_j, j in enumerate(neighbor):
                        for indx_k, k in enumerate(neighbor[indx_j + 1:],
                                                   start=indx_j + 1):
                            for l in neighbor[indx_k + 1:]:
                                update_topo_dict_indx('impropers',
                                                      [i, j, k, l])

        # finally extend all int connectivities
        if topo_dict_indx:
            self._extend(topo_dict_indx)

    def _set_indices_to(self, indx_of):
        """sets indices in bonds, etc as specified in indx_of"""
        # removing bonds etc containing non-existing ids
        for prop in self._dict.keys():
            if prop in self._ins._topology:
                self._dict[prop]._set_indices_to(indx_of)

    def _get_item(self, item, len_self):
        """used when __getitem__ is called in atoms object
        parameters
            item: slice
                Used for selecting atoms
            len_self: int
                length of original Atoms object"""
        indx_of = {i: None for i in range(len_self)}
        count = 0
        for i in np.array(range(len_self))[item]:
            indx_of[i] = count
            count += 1
        self._set_indices_to(indx_of)
        self.update()

    def _del_item(self, i):
        """used when __delitem__ is called in atoms object
        """
        if not isinstance(i, list):
            i = [i]
        # making index map dictionary, with index to delete as None
        # and setting indices to the map
        indx_of = {}
        count = 0
        for j in range(len(self._ins)):
            if j in i:
                indx_of[j] = None
                count += 1
            else:
                indx_of[j] = j - count

        self._set_indices_to(indx_of=indx_of)
        self.update()

    def _imul(self, m):
        """used when __imul__ is called in atoms object
        """
        size_m = np.product(m)
        # n contains the original length of atoms
        n = int(len(self._ins) / size_m)

        array = deepcopy(self._ins._topology)
        i0 = 0
        for m0 in range(m[0]):
            for m1 in range(m[1]):
                for m2 in range(m[2]):
                    i1 = i0 + n
                    if i0 != 0:
                        # extend connectivities with an offset
                        self._extend(array, _offset=i0)
                    i0 = i1

        self.update()

    def _extend(self, topo_dict, _offset=None):
        """
        Generates bonds, angles, dihedrals, and impropers based on indices
        of atoms, given as topo_dict

        Parameters
            topo_dict: dictionary
                Keys as bonds, angles, dihedrals, or impropers and values as
                numpy array of connectivity to be added
            _offset: int
                To offset the indices by _offset
        """
        for key, values in topo_dict.items():
            if key in self._dict:
                self._dict[key].add(values, _offset)
        self.update()

    def get_statistics(self, index=':'):
        """
        Returns statistics for bonds, angles, and dihedrals
        :param index: indices to return statistics for
        :return: a dictionary with names of attribute as keys, and
            the statistics dictionary as value
        """
        stats = {}
        for prop in ['bonds', 'angles', 'dihedrals']:
            if self.has(prop):
                stats[prop] = self._dict[prop].get_statistics(index)

        return stats
