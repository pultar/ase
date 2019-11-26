import numpy as np
from copy import deepcopy
from ase.utils import basestring
from ase.neighborlist import NeighborList, natural_cutoffs

import operator


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

    def __init__(self, topo_attr_prop):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = topo_attr_prop._ins
        self.persistent = topo_attr_prop.persistent
        self.prop = topo_attr_prop.prop

    def _check_exists(func):
        """Decorator to check if the property exists"""
        # Only added to functions which need already existing values in
        # atoms.arrays, otherwise the counting methods return 0
        def wrapper(*args, **kwargs):
            self = args[0]
            if self.prop not in self._ins._topology and not self._ins.has(self.prop):
                raise RuntimeError('{0} object has no '
                                   '{1}'.format(self._ins.__class__.__name__,
                                                self.prop))
            return func(*args, **kwargs)
        return wrapper

    def _check_persistence(func):
        """Decorator to check if the topology is persistent"""
        # Only added to functions which may edit/set topology data
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self.persistent:
                raise RuntimeError('Topology is not persistent')
            return func(*args, **kwargs)
        return wrapper

    def __repr__(self):
        if self.prop == 'tags':
            return '0-{}'.format(np.max(self.get_types()))
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
        """ returns number of types of prop: bonds, etc"""
        return len(self.get_types())

    def get_types(self, index=':', verbose=False):
        """returns types of prop: bonds, etc
        :param prop: name of property
        :param verbose: if true, returns types of individual bond/angles/etc"""
        if (self.prop not in self._ins._topology and self.prop not in ['names', 'tags']):
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
            for i, connectivity in enumerate(array):
                dict_[types[i]] = (dict_.get(types[i], [])
                                   + [connectivity.tolist()])
            return dict_
        elif self.prop == 'resnames':
            if with_names:
                return  self._ins._topology[self.prop]
            else:
                resnames = ['' for _ in range(len(self._ins))]
                for key, values in self._ins._topology[self.prop].items():
                    for value in values:
                        resnames[value] = key
                return resnames
        else:
            # tags and names
            if with_names:
                prop = {}
                for i, name in enumerate(self._ins.arrays[self.prop]):
                    prop[name] = prop.get(name, []) + [i]
                return prop
            else:
                return self._ins.arrays[self.prop]

    __call__ = get

    @_check_persistence
    def set(self, value=None):
        """
        value to set. None value deletes the property
        """
        if self.prop in self._ins._topology:
            # delete key in _topology
            del self._ins._topology[self.prop]
        if value is None:
            return None
        if self.prop == 'names':
            if np.any([type(x) is not str for x in value]):
                raise RuntimeError('names should be str')
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
        """ returns number of prop: bonds, etc."""
        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.prop not in self._ins._topology:
                return 0
            return len(self._ins._topology[self.prop])
        else:
            raise NotImplementedError('get_count only implemented for bonds,'
                                      ' angles, dihedrals, impropers')

    @_check_persistence
    def add(self, items, _offset=None):
        """ adds to prop
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
        """
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
            # make sure resnames are valid
            if np.any([type(i) is not str for i in items.keys()]):
                raise ValueError('resnames should be str')
            try:
                for i in items.keys():
                    items[i] = np.array(items[i], dtype=int)
            except ValueError as e:
                raise ValueError('resnames indices should be int')

            if _offset is not None:
                for i in items.keys():
                    items[i] += _offset

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
                if resname != '':
                    d[resname] = d.get(resname, []) + [i]
            self._ins._topology[self.prop] = d
        else:
            # tags and names
            raise NotImplementedError('{} does not support '
                                      'add'.format(self.prop))

    # no _check_* since its called by functions
    # non error raising methods of checking implemented
    def update(self):
        """
        """
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
            for i, val in enumerate(id_):
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
        """
        :param indx_of: dictionary, changes type from keys -> values
        :param index: list of start and stop index to affect the change
        """
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
                and self.prop not in self._ins._topology):
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
                self.generate()
            self.persistent = False

    def __repr__(self):
        tokens = []
        for key, values in self._dict.items():
            if key in self._ins._topology or key in ['names', 'tags']:
                tokens.append("{}= {}".format(key, values))
        return "{}.Topology({})".format(self._ins.__class__.__name__, ", ".join(tokens))

    def __getitem__(self, item):
        return self._dict[item]

    def get_topology_dict(self):
        """Gives topology dict that can be inherited by other topology
        classes"""
        topo_dict = {'names': self._ins.arrays['names'],
                     'tags': self._ins.arrays['tags']}
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if prop in self._ins._topology:
                topo_dict[prop] = self._ins._topology[prop]

        return topo_dict

    __call__ = get_topology_dict

    def update(self, topo_dict=None):
        if not self.persistent:
            raise RuntimeError('Topology is not persistent')
        # sanity check for topo_object
        if topo_dict is None:
            topo_dict = {}

        # initialise topology
        if not hasattr(self._ins, '_topology'):
            # setting new atoms object from prev one with topology
            # leads to this case
            self._ins._topology = {}
        elif self._ins._topology is None:
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
        class check_exist():
            """
            Class to check if index is present in topo_dict.
            """
            def __init__(self, topo_dict):
                # dict to hold connectivity names
                self.topo_ = {}
                if topo_dict is True:
                    self.gen_all = True
                else:
                    self.gen_all = False
                    for key, values in topo_dict.items():
                        for i, value in enumerate(values):
                            if key == 'bonds':
                                value = np.sort(value)
                            if key == 'angles':
                                edges = np.sort([value[0], value[2]])
                                value[0], value[2] = edges
                            self.topo_[key] = (self.topo_.get(key, [])
                                                + ['-'.join(value)])

            def __call__(self, index, prop):
                if self.gen_all:
                    return True
                else:
                    return index in self.topo_[prop]

            def prop_exists(self, prop):
                if self.gen_all:
                    return True
                else:
                    return prop in self.topo_


        if not self.persistent:
            raise RuntimeError('Topology is not persistent')

        # dict to hold str connectivities
        topo_dict = {}
        # dict to hold int connectivities
        topo_dict_indx = {}
        if bonds is not None:
            if len(bonds) != 0:
                if len(np.array(bonds).shape) != 2:
                    raise RuntimeError('a list of connectivities is expected')
                topo_dict['bonds'] = bonds
        if angles is not None:
            if len(angles) != 0:
                if len(np.array(angles).shape) != 2:
                    raise RuntimeError('a list of connectivities is expected')
                topo_dict['angles'] = angles
        if dihedrals is not None:
            if len(dihedrals) != 0:
                if len(np.array(dihedrals).shape) != 2:
                    raise RuntimeError('a list of connectivities is expected')
                topo_dict['dihedrals'] = dihedrals
        if impropers is not None:
            if len(impropers) != 0:
                if len(np.array(impropers).shape) != 2:
                    raise RuntimeError('a list of connectivities is expected')
                topo_dict['impropers'] = impropers

        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        if (bonds is None
                and angles is None
                and dihedrals is None
                and impropers is None):
            # empty generate passed, all connectivities to be added
            topo_dict = True
        else:
            # move connectivities based on str or indices
            for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
                if prop not in topo_dict:
                    continue
                if np.all([isinstance(y, (int, np.integer))
                           for x in topo_dict[prop] for y in x]):
                    # if indices are given
                    array = np.array(topo_dict[prop], dtype=int)
                    if array.shape[1] != length[prop]:
                        raise RuntimeError('{} should be of length {}'
                                           ''.format(prop, length[prop]))
                    # move them to topo_dict_indx
                    topo_dict_indx[prop] = array
                    topo_dict.pop(prop)
                elif np.all([type(y) is str
                           for x in topo_dict[prop] for y in x]):
                    # else str given
                    topo_dict[prop] = np.array(topo_dict[prop])
                    if topo_dict[prop].shape[1] != length[prop]:
                        raise RuntimeError('{} should be of length {}'
                                           ''.format(prop, length[prop]))
                else:
                    raise RuntimeError('indices should be either all int, or str')

        # convert str connectivities to int connectivities
        if topo_dict:
            check_ = check_exist(topo_dict)

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

            if check_.prop_exists('bonds'):
                for i, neighbor in enumerate(d):
                    for j in neighbor:
                        if i <= j:
                            # removes double counting
                            continue
                        name_list = [symbols[x] for x in [i, j]]
                        name_list.sort()
                        name_list = '-'.join(name_list)
                        if check_(name_list, 'bonds'):
                            if 'bonds' not in topo_dict_indx:
                                # initiate topo_dict_indx
                                topo_dict_indx['bonds'] = np.array([[i, j]],
                                                                   dtype=int)
                                continue
                            _ = np.vstack([topo_dict_indx['bonds'], [i, j]])
                            topo_dict_indx['bonds'] = _


            if check_.prop_exists('angles'):
                for i, neighbor in enumerate(d):
                    for indx, j in enumerate(neighbor):
                        for k in neighbor[indx + 1:]:
                            name_list = [symbols[x] for x in [k, i, j]]
                            vertex = name_list.pop(1)
                            name_list.sort()
                            name_list.insert(1, vertex)
                            name_list = '-'.join(name_list)
                            if check_(name_list, 'angles'):
                                if 'angles' not in topo_dict_indx:
                                    # initiate topo_dict_indx
                                    _ = np.array([[j, i, k]], dtype=int)
                                    topo_dict_indx['angles'] = _
                                    continue
                                _ = np.vstack([topo_dict_indx['angles'],
                                               [j, i, k]])
                                topo_dict_indx['angles'] = _

            if check_.prop_exists('dihedrals'):
                for i, neighbor_i in enumerate(d):
                    for j in neighbor_i:
                        for k in set(d[j]) - set([i, j]):
                            for l in set(d[k]) - set([i, j, k]):
                                name_list = [symbols[x] for x in [i, j, k, l]]
                                name_list = '-'.join(name_list)
                                if check_(name_list, 'dihedrals'):
                                    if 'dihedrals' not in topo_dict_indx:
                                        # initiate topo_dict_indx
                                        _ = np.array([[i, j, k, l]], dtype=int)
                                        topo_dict_indx['dihedrals'] = _
                                        continue
                                    _ = np.vstack([topo_dict_indx['dihedrals'],
                                                   [i, j, k, l]])
                                    topo_dict_indx['dihedrals'] = _

            if check_.prop_exists('impropers'):
                for i, neighbor in enumerate(d):
                    for indx_j, j in enumerate(neighbor):
                        for indx_k, k in enumerate(neighbor[indx_j + 1:],
                                                   start=indx_j + 1):
                            for l in neighbor[indx_k + 1:]:
                                name_list = [symbols[x] for x in [i, j, k, l]]
                                name_list = '-'.join(name_list)
                                if check_(name_list, 'impropers'):
                                    if 'impropers' not in topo_dict_indx:
                                        # initiate topo_dict_indx
                                        _ = np.array([[i, j, k, l]], dtype=int)
                                        topo_dict_indx['impropers'] = _
                                        continue
                                    _ = np.vstack([topo_dict_indx['impropers'],
                                                   [i, j, k, l]])
                                    topo_dict_indx['impropers'] = _

        # extend all connectivities
        if topo_dict_indx:
            self._extend(topo_dict_indx)

    def _set_indices_to(self, indx_of):
        """sets indices in bonds, etc as specified in indx_of"""
        # removing bonds etc containing non-existing ids
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if prop in self._dict:
                self._dict[prop]._set_indices_to(indx_of)

    def _get_item(self, item, len_self):
        """used when _get_item is called in atoms object
        Method corrects bonds, angles, dihedrals, and impropers that
        point to wrong indices due to array slicing
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

        array = deepcopy(self._ins._topology)
        max_tags = 0
        n_tags = np.max(self._ins.get_tags()) + 1
        i0 = 0
        for m0 in range(m[0]):
            for m1 in range(m[1]):
                for m2 in range(m[2]):
                    i1 = i0 + n
                    if i0 != 0:
                        # extend connectivities with an offset
                        self._extend(self._ins._topology, _offset=i0)
                    if self._ins.has('tags'):
                        _ = self._ins.arrays['tags'][i0:i1] + max_tags
                        self._ins.arrays['tags'][i0:i1] = _
                        max_tags += n_tags
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
        if not self.persistent:
            raise RuntimeError('Topology is not persistent')
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
            if prop in self._ins._topology:
                stats[prop] = self._dict[prop].get_statistics(index)

        return stats
