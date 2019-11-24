import numpy as np
from copy import deepcopy
from ase.utils import basestring
from ase.neighborlist import NeighborList, natural_cutoffs

import operator


# ! TODO update 'names' array to the future ASE labels
def string2index(string):
    '''converts sring to index
    cannot be imported from ase.io.formats.'''
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

    def __init__(self, topo_attr_prop):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = topo_attr_prop._ins
        self.persistent = topo_attr_prop.persistent
        self.prop = topo_attr_prop.prop

    def _check_exists(func):
        '''Decorator to check if the property exists'''
        # Only added to functions which need already existing values in
        # atoms.arrays, otherwise the counting methods return 0
        def wrapper(*args, **kwargs):
            self = args[0]
            exist = True
            if self.prop in ['bonds',
                             'angles',
                             'dihedrals',
                             'impropers']:
                if self.prop not in self._ins._topology:
                    exist = False
            elif self._ins.has(self.prop):
                exist = False
            if not exist:
                raise RuntimeError('{0} object has no '
                                   '{1}'.format(self._ins.__class__.__name__,
                                                self.prop))
            return func(*args, **kwargs)
        return wrapper

    def _check_persistence(func):
        '''Decorator to check if the topology is persistent'''
        # Only added to functions which may edit/set topology data
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self.persistent:
                raise RuntimeError('Topology is not persistent')
            return func(*args, **kwargs)
        return wrapper

    def __repr__(self):
        if self.prop in ['ids', 'tags']:
            return '1 ... {}'.format(self.get_num_types())
        else:
            return str(self.get_types())

    @_check_exists
    def __getitem__(self, item):
        props = self.get(with_names=True)
        if item not in props:
            raise RuntimeError('{} not in {}'.format(item, self.prop))
        return props[item]

    @_check_persistence
    @_check_exists
    def __delitem__(self, items):
        if not isinstance(items, list):
            items = [items]
        if self.prop in ['resnames']:
            props = self.get()
            for item in items:
                props.pop(item)
            self.set(props)
        elif self.prop in ['bonds',
                           'angles',
                           'dihedrals',
                           'impropers']:
            types = self.get_types(verbose=True)
            array = self._ins._topology[self.prop]
            mask = np.ones(len(array))
            for item in items:
                # check if item exists
                if item not in types:
                    raise RuntimeWarning('{} {} does not exist'
                                         ''.format(self.prop, item))
                mask[types == item] = 0
            array = array[mask]
            self._ins._topology[self.prop] = array
            self.update()
        elif self.prop in ['tags',
                           'names',
                           'ids']:
            del_ind = []
            for item in items:
                del_ind += np.where(self.get() == item)[0].tolist()
            del self._ins[del_ind]

    def get_num_types(self):
        ''' returns number of types of prop: bonds, etc'''
        return len(self.get_types())

    def get_types(self, index=':', verbose=False):
        '''returns types of prop: bonds, etc
        :param prop: name of property
        :param verbose: if true, returns types of individual bond/angles/etc'''
        if self.prop not in self._ins._topology:
            return np.array([])

        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
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
            type_list = [None for _ in len(items_indx)]
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

        elif self.prop == 'resnames':
            return np.array(list(self._ins._topology[self.prop].keys()))
        else:
            # tags and names
            return np.unique(self._ins.get_array(self.prop)[index])

    @_check_exists
    def get(self, with_names=False):
        """
        Parameters
            with_names: bool
                returns bonds/angles/dihedrals/impropers as dict of
                connectivity name as key and list of indices of
                connectivities"""
        if self.prop in ['bonds',
                         'angles',
                         'dihedrals',
                         'impropers']:
            if not with_names:
                return self._ins._topology[self.prop]
            array = self._ins._topology[self.prop]
            types = self.get_types(verbose=True)
            dict_ = {}
            for i, connectivity in array:
                dict_[types[i]] = (dict_.get(types[i], [])
                                   + [connectivity])
            return dict_
        elif self.prop == 'resnames':
            return  self._ins._topology[self.prop]
        else:
            # tags and names
            return self._ins.arrays[self.prop]

    __call__ = get

    @_check_persistence
    def set(self, value=None):
        '''
        value to set. None value deletes the property
        '''
        if self.prop in self._ins._topology:
            # delete key in _topology
            del self._ins._topology[self.prop]
        if value is None:
            return None
        if self.prop == 'names':
            self._ins.set_array(self.prop, value, object)
            self.update()
        elif self.prop == 'tags':
            self._ins.set_array(self.prop, value, int)
            self.update()
        elif self.prop in ['resnames',
                           'bonds',
                           'angles',
                           'dihedrals',
                           'impropers']:
            self.add(value)
            self.update()

    def get_count(self):
        ''' returns number of prop: bonds, etc.'''
        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.prop not in self._ins._topology:
                return 0
            return len(self._ins._topology[self.prop])
        else:
            raise NotImplementedError('get_count only implemented for bonds,'
                                      ' angles, dihedrals, impropers')

    @_check_persistence
    def add(self, items, _offset=None):
        ''' adds to prop
        Parameters:
            items: array or dict
                Array of shape (num_prop, x), where num prop is number of
                of bonds/angles/dihedrals/impropers, and x is size of
                connectivity.
                or dict of keys as resname, and keys as list of indices
                pertaining to the key for resname
            _offset: int
                Added to items to offset indices. Useful when adding atoms
                objects
        '''
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        if isinstance(items, dict):
            # Deletes empty items
            _del_key = []
            for key, values in items.items():
                if len(values) == 0:
                    _del_key.append(key)
            for key in _del_key:
                items.pop(key)
        # if items is empty return
        if len(items) == 0:
            return

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            # make sure items is int and of correct size
            items = np.array(items, dtype=int)
            if (np.any([len(np.unique(item)) != length[self.prop]
                       for item in items]) or len(items.shape) != 2):
                raise RuntimeError('{0} should be unique indices of size '
                                   '{1}'.format(self.prop, length[self.prop]))
            # make sure indices are within length of atoms object
            if np.any([x >= len(self._ins) for x in items.reshape(-1)]):
                raise RuntimeError('indices should be lower than length of '
                                   '{}'.format(self._ins.__class__.__name__))

            if _offset is not None:
                items += _offset

            if self.prop not in self._ins._topology:
                array = []
            else:
                array = self._ins._topology[self.prop].tolist()

            # check if connectivity already exists
            for item in items:
                item = item.tolist()
                if self.prop == 'bonds':
                    item.sort()
                elif self.prop == 'angles':
                    indx = item.pop(1)
                    item.sort()
                    item.insert(1, indx)
                elif self.prop == 'dihedrals':
                    if item[0] > item[3]:
                        item.reverse()
                if item not in array:
                    array.append(item)
            self._ins._topology[self.prop] = np.array(array, dtype=int)
            self.update()

        elif self.prop == 'resnames':
            # make sure indices are within length of atoms object
            if np.any([x >= len(self._ins)
                       for values in items.values() for x in values]):
                raise RuntimeError('indices should be lower than length of '
                                   '{}'.format(self._ins.__class__.__name__))
            # to make sure each atom has a single resname we make an array
            resnames = ['' for _ in range(len(self._ins))]

            if self.prop in self._ins._topology:
                # add resnames to the array
                for key, values in self._ins._topology[self.prop]:
                    for value in values:
                        resnames[value] = key

            # add new resnames
            for key, vals in items.items():
                for val in vals:
                    resnames[val] = key

            # convert it back to dictionary
            d = {}
            for i, resname in enumerate(resnames):
                d[resname] = d.get(resname, []) + [i]
            self._ins._topology[self.prop] = resnames
        else:
            # tags and names
            raise NotImplementedError('{} does not support '
                                      'add'.format(self.prop))

    # no _check_* since its called by functions
    # non error raising methods of checking implemented
    def update(self):
        '''
        '''
        # if topology not persistent, then don't update
        # the _check_persistent wrapper is not used
        if not self.persistent:
            return

        if self.prop in ['bonds',
                         'angles',
                         'dihedrals',
                         'impropers']:
            # Correct for empty connectivity
            if self.prop in self._ins._topology:
                if len(self._ins._topology[self.prop]) == 0:
                    # prop is empty
                    del self._ins._topology[self.prop]

        elif self.prop == 'resnames':
            if self.prop in self._ins._topology:
                if len(self.get_types()) == 0:
                    del self._ins.array[self.prop]

        elif self.prop == 'tags':
            if not self._ins.has(self.prop):
                self._ins.set_array('tags',
                                    np.ones(len(self._ins)),
                                    int)
            # change sparse tags to uniform tags
            id_ = np.unique(self._ins.arrays[self.prop])
            ind_of = {}
            for i, val in enumerate(id_, start=1):
                if val != i:
                    ind_of[val] = i
            self._ins.set_array(self.prop,
                                [(ind_of[x] if x in ind_of else x)
                                 for x in self._ins.get_array(self.prop)],
                                int)

        elif self.prop == 'names':
            if not self._ins.has('names'):
                self._ins.set_array('names',
                                    self._ins.get_chemical_symbols(),
                                    object)

    @_check_exists
    @_check_persistence
    def set_types_to(self, indx_of, index=":"):
        '''
        :param indx_of: dictionary, changes type from keys -> values
        :param index: list of start and stop index to affect the change
        '''
        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass

        if len(indx_of) == 0:
            return

        if self.prop in ['tags', 'names']:
            for indx in np.arange(len(self._ins))[index]:
                try:
                    _ = indx_of[self._ins.arrays[self.prop][indx]]
                except IndexError:
                    continue
                self._ins.arrays[self.prop][indx] = _
            self.update()
        elif self.prop == 'resnames':
            prop = self._ins._topology[self.prop]
            for key in list(prop.keys()):
                if key in indx_of:
                    prop[indx_of[key]] = prop[key]
                    prop.pop(key)
            self._ins._topology[self.prop] = prop

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
        if (self.prop not in ['bonds', 'angles', 'dihedrals', 'impropers']
                or self.prop not in self._ins._topology):
            return None

        array = self._ins._topology[self.prop]
        mask = np.ones(len(array))
        for i in range(len(array)):
            if np.any([indx_of[x] is None for x in array[i]]):
                mask[i] = 0
                continue
            for j in range(array.shape[1]):
                array[i, j] = indx_of[array[i, j]]
        array = array[mask]
        self._ins._topology[self.prep] = array

    @_check_exists
    def get_statistics(self, index=':'):
        '''
        Returns statistics for specific attribute
        Parameters
            index: indices to return statistics for
        Returns
            Dictionary with bond/angles/dihedral names with a
            list of values
        '''
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

    def __init__(self, prop):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self.prop = prop

    def __get__(self, topo_base, owner):
        if topo_base is None:
            return self
        self._ins = topo_base._ins
        self.persistent = topo_base.persistent
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
    Adds topology to atoms object. The topology saves molecule tags, names,
    bonds, angles, dihedrals and impropers. The molecule tags and names are
    save as arrays in atoms object, and the rest are saved as numpy array in
    atoms._topology dictionary.
    On initialisation through atoms object, names and tags are auto initiated
    and the connectivity is initiated by the user.
    """
    names = _TopoAttributeProperty('names')
    tags = _TopoAttributeProperty('tags')
    resnames = _TopoAttributeProperty('resnames')
    bonds = _TopoAttributeProperty('bonds')
    angles = _TopoAttributeProperty('angles')
    dihedrals = _TopoAttributeProperty('dihedrals')
    impropers = _TopoAttributeProperty('impropers')

    def __init__(self, instance, persistent=True):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = instance
        # a bool to see if the topology is persistent
        # i.e. the set/edit methods can be used or not
        # changed to the passed persistent when __init__ ends
        self.persistent = True
        # a dict to hold all attributes
        self._dict = {'names': self.names,
                      'tags': self.tags,
                      'resnames': self.resnames,
                      'bonds': self.bonds,
                      'angles': self.angles,
                      'dihedrals': self.dihedrals,
                      'impropers': self.impropers}

        if not persistent:
            if self._ins._topology is None:
                # if no topology already,
                # update all attributes needed
                # add bonds/angles/dihedrals/impropers based on
                # connectivity, in ase.neighborlists
                self.update()
                self.generate_with_names()
            self.persistent = False

    def __repr__(self):
        tokens = []
        for key, values in self._dict.items():
            tokens.append("{}= {}".format(key, values))
        return "{}.Topology({})".format(self._ins.__class__.__name__, ", ".join(tokens))

    def __getitem__(self, item):
        return self._dict[item]

    def get_topology_object(self):
        '''Gives topology dict that can be inherited by other topology
        classes'''
        topo_dict = {'names': self._ins.arrays['names'],
                     'tags': self._ins.arrays['tags']}
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if prop in self._ins._topology:
                topo_dict[prop] = self._ins._topology[prop]

        return TopologyObject(topo_dict)

    __call__ = get_topology_object

    def update(self, topo_object=None):
        if not self.persistent:
            raise RuntimeError('Topology is not persistent')
        # sanity check for topo_object
        if topo_object is None:
            topo_dict = {}
        elif type(topo_object) is TopologyObject:
            topo_dict = topo_object._dict
        else:
            topo_dict = TopologyObject(topo_object)._dict

        # initialise topology
        if self._ins._topology is None:
            self._ins._topology = {}

        for prop in ['names',
                     'tags',
                     'resnames',
                     'bonds',
                     'angles',
                     'dihedrals',
                     'impropers']:
            if prop in topo_dict:
                self._dict[prop].set(topo_dict[prop])
            elif prop in self._dict:
                self._dict[prop].update()

    def generate_with_names(self, topo_dict=None, cutoffs=None):
        '''
        Generates bonds, angles, dihedrals, and impropers based on names
        of atoms, given as topo_dict

        Parameters:
        topo_dict: dictionary with 'bonds', 'angles', 'dihedrals', and
            'impropers' as keys, and a list of list of names involved in bonds/
            angles/impropers/dihedrals
        cutoffs: float array with cuttoffs of each atom in atoms object

        Example:

        >>> water = Atoms('H2O', positions=[[0, 0, 0],
        ...                                 [1, 1, 0],
        ...                                 [-1, -1, 0]]
        ...              )
        >>> water.set_topology()
        >>> water.topology.generate_with_names({'bonds': [['H', 'O']],
         ...                                    'angles': [['H', 'O', 'H']]}
         ...                                  )
        '''
        if not self.persistent:
            raise RuntimeError('Topology is not persistent')

        # check and reformat topo_dict
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        if topo_dict is None:
            topo_dict = {}

        # dumb method of adding all connectivities
        # if topo_dict is empty, then make a topo_dict that has all possible
        # combinations of connectivity. This way all connectivities will be
        # added with same code
        if len(topo_dict) == 0:
            # add all possible name interactions
            names = self.names.get_types()
            # bonds
            topo_dict['bonds'] = []
            for i, name_i in enumerate(names):
                for name_j in names[i:]:
                    topo_dict['bonds'].append([name_i, name_j])
            # angles
            topo_dict['angles'] = []
            for name_i in names:
                for j, name_j in enumerate(names):
                    for name_k in names[j:]:
                        topo_dict['angles'].append([name_j,
                                                    name_i,
                                                    name_k])
            # dihedrals and impropers
            topo_dict['dihedrals'] = []
            topo_dict['impropers'] = []
            for i, name_i in enumerate(names):
                for j, name_j in enumerate(names[i:],
                                           start=i):
                    for k, name_k in enumerate(names[j:],
                                               start=j):
                        for name_l in names[k:]:
                            topo_dict['dihedrals'].append([name_i,
                                                           name_j,
                                                           name_k,
                                                           name_l])
                            topo_dict['impropers'].append([name_i,
                                                           name_j,
                                                           name_k,
                                                           name_l])

        for key, values in topo_dict.items():
            if key not in ['bonds', 'angles', 'dihedrals', 'impropers']:
                raise ValueError('{} not supported. Supported properties: '
                                 '{}'.format(key, ['bonds',
                                                   'angles',
                                                   'dihedrals',
                                                   'impropers']))
            if type(values) == np.ndarray:
                values = values.tolist()
                topo_dict[key] = topo_dict[key].tolist()

            if not np.all([len(x) == length[key] for x in values]):
                raise ValueError('{} are lists of length '
                                 '{}'.format(key, length[key]))
            for i, value in enumerate(values):
                if key == 'bonds':
                    value.sort()
                if key == 'angles':
                    vertex = value.pop(1)
                    value.sort()
                    value.insert(1, vertex)
                topo_dict[key][i] = '-'.join(value)

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

        if 'bonds' in topo_dict:
            if 'bonds' in self._dict:
                n_types = self._dict['bonds'].get_num_types()
            else:
                n_types = 0
            seen_bonds = []
            bonds = {}
            for i, neighbor in enumerate(d):
                for j in neighbor:
                    name_list = [symbols[x] for x in [i, j]]
                    name_list.sort()
                    name_list = '-'.join(name_list)
                    if name_list in topo_dict['bonds'] and not [i, j] in seen_bonds:
                        indx = topo_dict['bonds'].index(name_list) + 1 + n_types
                        bonds[indx] = bonds.get(indx, []) + [[i, j]]
                        seen_bonds.append([i, j])
                        seen_bonds.append([j, i])
            self.bonds.add(bonds)

        if 'angles' in topo_dict:
            if 'angles' in self._dict:
                n_types = self._dict['angles'].get_num_types()
            else:
                n_types = 0
            angles = {}
            for i, neighbor in enumerate(d):
                for indx, j in enumerate(neighbor):
                    for k in neighbor[indx + 1:]:
                        name_list = [symbols[x] for x in [k, i, j]]
                        vertex = name_list.pop(1)
                        name_list.sort()
                        name_list.insert(1, vertex)
                        name_list = '-'.join(name_list)
                        if name_list in topo_dict['angles']:
                            indx = topo_dict['angles'].index(name_list) + 1 + n_types
                            angles[indx] = angles.get(indx, []) + [[k, i, j]]
            self.angles.add(angles)

        if 'dihedrals' in topo_dict:
            if 'dihedrals' in self._dict:
                n_types = self._dict['dihedrals'].get_num_types()
            else:
                n_types = 0
            dihedrals = {}
            for i, neighbor_i in enumerate(d):
                for j in neighbor_i:
                    for k in set(d[j]) - set([i, j]):
                        for l in set(d[k]) - set([i, j, k]):
                            name_list = [symbols[x] for x in [i, j, k, l]]
                            name_list.sort()
                            name_list = '-'.join(name_list)
                            if name_list in topo_dict['dihedrals']:
                                indx = topo_dict['dihedrals'].index(name_list) + 1 + n_types
                                dihedrals[indx] = dihedrals.get(indx, []) + [[i, j, k, l]]
            self.dihedrals.add(dihedrals)

        if 'impropers' in topo_dict:
            if 'impropers' in self._dict:
                n_types = self._dict['impropers'].get_num_types()
            else:
                n_types = 0
            impropers = {}
            for i, neighbor in enumerate(d):
                for indx_j, j in enumerate(neighbor):
                    for indx_k, k in enumerate(neighbor[indx_j + 1:], start=indx_j + 1):
                        for l in neighbor[indx_k + 1:]:
                            name_list = [symbols[x] for x in [i, j, k, l]]
                            name_list.sort()
                            name_list = '-'.join(name_list)
                            if name_list in topo_dict['impropers']:
                                indx = topo_dict['impropers'].index(name_list) + 1 + n_types
                                impropers[indx] = impropers.get(indx, []) + [[i, j, k, l]]
            self.impropers.add(impropers)

    def generate_with_indices(self, topo_dict):
        '''
        Generates bonds, angles, dihedrals, and impropers based on indices
        of atoms, given as topo_dict
        The property is automatically assigned type, base of names of property
        involved

        Parameters:
        topo_dict: dictionary with 'bonds', 'angles', 'dihedrals', and
            'impropers' as keys, and a list of list of indices involved in bonds/
            angles/impropers/dihedrals

        Example:

        >>> water = Atoms('H2O', positions=[[0, 0, 0],
        ...                                 [1, 1, 0],
        ...                                 [-1, -1, 0]]
        ...              )
        >>> water.set_topology()
        >>> water.topology.generate_with_indices({'bonds': [[0, 2], [1, 2]],
         ...                                      'angles': [[0, 2, 1]]}
         ...                                    )
        '''
        if not self.persistent:
            raise RuntimeError('Topology is not persistent')

        # check and reformat topo_dict
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        for key, values in topo_dict.items():
            if key not in ['bonds', 'angles', 'dihedrals', 'impropers']:
                raise ValueError('{} not supported. Supported properties: '
                                 '{}'.format(key, ['bonds',
                                                   'angles',
                                                   'dihedrals',
                                                   'impropers']))
            if type(values) == np.ndarray:
                values = values.tolist()
                topo_dict[key] = topo_dict[key].tolist()

            if not np.all([len(x) == length[key] for x in values]):
                raise ValueError('{} are lists of length '
                                 '{}'.format(key, length[key]))

        for key, values in topo_dict.items():
            self._dict[key].add({1: values})

        # update as if names has to be set
        # this updates properties with names
        self.names.update()

    def _set_indices_to(self, indx_of):
        '''sets indices in bonds, etc as specified in indx_of'''
        # removing bonds etc containing non-existing ids
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if prop in self._dict:
                self._dict[prop]._set_indices_to(indx_of)

    def _get_item(self, item, len_self):
        '''used when _get_item is called in atoms object
        Method corrects bonds, angles, dihedrals, and impropers that
        point to wrong indices due to array slicing
        parameters
            item: slice
                Used for selecting atoms
            len_self: int
                length of original Atoms object'''
        indx_of = {i: None for i in range(len_self)}
        count = 0
        for i in np.array(range(len_self))[item]:
            indx_of[i] = count
            count += 1
        self._set_indices_to(indx_of)
        self.update()

    def _del_item(self, i):
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
        size_m = np.product(m)
        # n contains the original length of atoms
        n = int(len(self._ins) / size_m)

        i0 = 0
        max_tags = 0
        n_tags = np.max(self._ins.get_tags())
        for m0 in range(m[0]):
            for m1 in range(m[1]):
                for m2 in range(m[2]):
                    i1 = i0 + n
                    if i0 != 0:
                        for prop in ['bonds',
                                     'angles',
                                     'dihedrals',
                                     'impropers']:
                            if prop in self._ins._topology:
                                # add prop with an offset
                                array = self._ins._topology[prop]
                                self._dict[prop].add(array, _offset=i0)
                    if self._ins.has('tags'):
                        _ = self._ins.arrays['tags'][i0:i1] + max_tags
                        self._ins.arrays['tags'][i0:i1] = _
                        max_tags += n_tags
                    i0 = i1

        self.update()

    def _extend(self, topo_dict, _offset=None):
        """
        Parameters
            topo_dict: dictionary
                Keys as bonds, angles, dihedrals, or impropers and values as
                numpy array of connectivity to be added
            _offset: int
                To offset the indices by _offset
        """
        if not self.persistent:
            raise RuntimeError('Topology is not persistent')
        for key, values in topo_dict.items():
            self._dict[key].add(values, _offset)
        self.update()

    def get_statistics(self, index=':'):
        '''
        Returns statistics for bonds, angles, and dihedrals
        :param index: indices to return statistics for
        :return: a dictionary with names of attribute as keys, and
            the statistics dictionary as value
        '''
        stats = {}
        for prop in ['bonds', 'angles', 'dihedrals']:
            if prop in self._ins._topology:
                stats[prop] = self._dict[prop].get_statistics(index)

        return stats


class TopologyObject(object):
    '''Topology object that can be used to transfer topologies
    attached to Atoms object'''

    topo_props = ['ids',
                  'names',
                  'tags',
                  'resnames',
                  'bonds',
                  'angles',
                  'dihedrals',
                  'impropers']

    lengths = {'bonds': 2,
               'angles': 3,
               'dihedrals': 4,
               'impropers': 4}

    def __init__(self, topology_dict={}):
        self._dict = {}
        for i in topology_dict:
            if i in self.topo_props:
                if i == 'ids':
                    self.ids = topology_dict[i]
                if i == 'names':
                    self.names = topology_dict[i]
                if i == 'tags':
                    self.tags = topology_dict[i]
                if i == 'resnames':
                    self.resnames = topology_dict[i]
                if i == 'bonds':
                    self.bonds = topology_dict[i]
                if i == 'angles':
                    self.angles = topology_dict[i]
                if i == 'dihedrals':
                    self.dihedrals = topology_dict[i]
                if i == 'impropers':
                    self.impropers = topology_dict [i]
            else:
                raise RuntimeError('{} not a topology property'.format(i))

    @property
    def ids(self):
       return self._dict['ids']

    @ids.setter
    def ids(self, values):
        try:
            ids = np.array(values, dtype=int)
        except ValueError as e:
            raise ValueError(e, 'ids should be integer')
        self._dict['ids'] = ids

    @property
    def names(self):
        return self._dict['names']

    @names.setter
    def names(self, values):
        for i in values:
            if not type(i) is str:
                raise ValueError('names should be str')
        self._dict['names'] = values

    @property
    def tags(self):
        return self._dict['tags']

    @tags.setter
    def tags(self, values):
        try:
            tags = np.array(values, dtype=int)
        except ValueError as e:
            raise ValueError(e, 'tags should be integer')
        self._dict['tags'] = tags

    @property
    def resnames(self):
        return self._dict['resnames']

    @resnames.setter
    def resnames(self, values):
        for i in values.keys():
            if not type(i) is str:
                raise ValueError('resnames should be str')
        for i in values.values():
            try:
                _ = np.array(list(i), dtype=int)
            except ValueError as e:
                raise ValueError(e, 'resnames indices should be int')
        self._dict['resnames'] = values

    @property
    def bonds(self):
        return self._dict['bonds']

    @bonds.setter
    def bonds(self, values):
        if not type(values) is dict:
            raise TypeError('bonds should be dict')
        try:
            _ = np.array(list(values.keys()), dtype=int)
        except ValueError as e:
            raise ValueError(e.args[0]
                            + ' bonds keys should be int')
        for bonds in values.values():
            for bond in bonds:
                if len(bond) != self.lengths['bonds']:
                    raise ValueError('bonds should be of size '
                                     '{}'.format(self.lengths['bonds']))
                try:
                    _ = np.array(bond, dtype=int)
                except ValueError as e:
                    raise ValueError(e.args[0]
                                    + ' bond indices should be int')
        self._dict['bonds'] = values

    @property
    def angles(self):
        return self._dict['angles']

    @angles.setter
    def angles(self, values):
        if not type(values) is dict:
            raise TypeError('angles should be dict')
        for key in values.keys():
            try:
                _ = operator.index(key)
            except TypeError as e:
                raise TypeError(e.args[0]
                                + ' angles keys should be int')
        for angles in values.values():
            for angle in angles:
                if len(angle) != self.lengths['angles']:
                    raise ValueError('angles should be of size '
                                     '{}'.format(self.lengths['angles']))
                try:
                    _ = np.array(angle, dtype=int)
                except ValueError as e:
                    raise ValueError(e.args[0]
                                    + ' angle indices should be int')
        self._dict['angles'] = values

    @property
    def dihedrals(self):
        return self._dict['dihedrals']

    @dihedrals.setter
    def dihedrals(self, values):
        if not type(values) is dict:
            raise TypeError('dihedrals should be dict')
        for key in values.keys():
            try:
                _ = operator.index(key)
            except TypeError as e:
                raise TypeError(e.args[0]
                                + ' dihedrals keys should be int')
        for dihedrals in values.values():
            for dihedral in dihedrals:
                if len(dihedral) != self.lengths['dihedrals']:
                    raise ValueError('dihedrals should be of size '
                                     '{}'.format(self.lengths['dihedrals']))
                try:
                    _ = np.array(dihedral, dtype=int)
                except ValueError as e:
                    raise ValueError(e.args[0]
                                    + ' dihedral indices should be int')
        self._dict['dihedrals'] = values

    @property
    def impropers(self):
        return self._dict['impropers']

    @impropers.setter
    def impropers(self, values):
        if not type(values) is dict:
            raise TypeError('impropers should be dict')
        for key in values.keys():
            try:
                _ = operator.index(key)
            except TypeError as e:
                raise TypeError(e.args[0]
                                + ' impropers keys should be int')
        for impropers in values.values():
            for improper in impropers:
                if len(improper) != self.lengths['impropers']:
                    raise ValueError('impropers should be of size '
                                     '{}'.format(self.lengths['impropers']))
                try:
                    _ = np.array(improper, dtype=int)
                except ValueError as e:
                    raise ValueError(e.args[0]
                                     + ' improper indices should be int')
        self._dict['impropers'] = values
