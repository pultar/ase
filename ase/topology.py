import numpy as np
from copy import deepcopy
from ase.utils import basestring
from ase.neighborlist import NeighborList, natural_cutoffs

import operator


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
        self.prop = topo_attr_prop.prop

    def _check_exists(func):
        '''Decorator to check if the property exists'''
        # Only added to functions which need already existing values in
        # atoms.arrays, otherwise the counting methods return 0
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self._ins.has(self.prop):
                raise RuntimeError('{0} object has no '
                                   '{1}'.format(self._ins.__class__.__name__,
                                                self.prop))
            return func(*args, **kwargs)
        return wrapper

    def __repr__(self):
        if self.prop in ['ids', 'mol-ids']:
            return '1 ... {}'.format(self.get_num_types())
        else:
            return str(self.get_types())

    @_check_exists
    def __getitem__(self, item):
        if type(item) is str and self.prop in ['bonds',
                                               'angles',
                                               'dihedrals',
                                               'impropers']:
            reverse_type = {i:j for j, i in self.get_types().items()}
            item = reverse_type[item]
        return self.get()[item]

    @_check_exists
    def __delitem__(self, items):
        if not isinstance(items, list):
            items = [items]
        if self.prop in ['resnames',
                         'bonds',
                         'angles',
                         'dihedrals',
                         'impropers']:
            props = self.get()
            for item in items:
                if type(item) is str and self.prop in ['bonds',
                                                       'angles',
                                                       'dihedrals',
                                                       'impropers']:
                    reverse_type = {i: j for j, i in self.get_types().items()}
                    item = reverse_type[item]
                props.pop(item)
            self.set(props)
        elif self.prop in ['mol-ids',
                           'names',
                           'types',
                           'ids']:
            del_ind = []
            for item in items:
                del_ind += np.where(self.get() == item)[0].tolist()
            del self._ins[del_ind]

    def get_num_types(self):
        ''' returns number of types of prop: bonds, etc'''
        return len(self.get_types(verbose=False))

    def get_types(self, index=':', verbose=True):
        '''returns types of prop: bonds, etc
        :param prop: name of property
        :param verbose: gives name abbreviation for the types'''
        if not self._ins.has(self.prop):
            return np.array([])

        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            items = deepcopy(self._ins.arrays[self.prop][index])
            # remove indices not present in index
            indx_of = {}
            _ = np.arange(len(self._ins))[index]
            count = 0
            # ids to delete
            ids = []
            for i in range(len(self._ins)):
                if i in _:
                    indx_of[i] = count
                    count += 1
                else:
                    ids.append(i)
            for indx in range(len(items)):
                # holds empty keys for deletion later
                del_key = []
                for key, value in items[indx].items():
                    # holds index of list that contains removed bonds etc
                    del_id = []
                    for i, j in enumerate(value):
                        if np.any([x in ids for x in j]):
                            del_id.append(i)
                        else:
                            for k, l in enumerate(j):
                                value[i][k] = indx_of[l]
                    del_id.sort(reverse=True)
                    for i in del_id:
                        value.pop(i)
                    # If value is empty then mark key for delition
                    # note that if it is empty, the arrays is not changed
                    # hence, delition is necessary
                    if len(value) == 0:
                        del_key.append(key)
                    else:
                        items[indx][key] = value

                for i in del_key:
                    items[indx].pop(i)

            type_list = np.unique([i for x in items for i in x.keys()])

            if not verbose:
                return type_list

            names = self._ins.arrays['names'][index]
            types = {}
            for key in type_list:
                for i, item in enumerate(items):
                    if key in item.keys():
                        eg_list = [i] + item[key][0]
                        break
                name_list = names[eg_list].tolist()
                if self.prop == 'bonds':
                    name_list.sort()
                if self.prop == 'angles':
                    # vertex atom is at first index
                    vertex = name_list.pop(0)
                    name_list.sort()
                    name_list.insert(1, vertex)
                types[key] = '-'.join(name_list)
            return types
        elif self.prop == 'resnames':
            types = np.unique(self._ins.get_array(self.prop)[index])
            return np.array(list(set(types) - set([''])))
        else:
            return np.unique(self._ins.get_array(self.prop)[index])

    @_check_exists
    def get(self):
        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            d = {}
            for i, item in enumerate(self._ins.arrays[self.prop]):
                for key, values in item.items():
                    for value in values:
                        if self.prop == 'angles':
                            value = [value[0], i, value[1]]
                        else:
                            value = [i] + value
                        d[key] = d.get(key, []) + [value]
            return d
        elif self.prop == 'resnames':
            d = {}
            for i, resname in enumerate(self._ins.get_array(self.prop)):
                if not resname == '':
                    d[resname] = d.get(resname, []) + [i]
            return d
        else:
            return self._ins.arrays[self.prop]

    __call__ = get

    def set(self, value=None):
        '''
        value to set. None value deletes the property
        '''
        if self._ins.has(self.prop):
            # delete array
            del self._ins.arrays[self.prop]
        if value is None:
            return None
        if self.prop == 'ids':
            self.update()
            #raise NotImplementedError('changing ids shuffles the atoms,'
            #                          'which is not implemented yet')
        elif self.prop == 'names':
            self._ins.set_array(self.prop, value, object)
            self.update(_name_set=True)
        elif self.prop in ['types', 'mol-ids']:
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
            if not self._ins.has(self.prop):
                return 0
            items = self._ins.arrays[self.prop]
            values = [j for x in items for i in x.values() for j in i]
            return len(values)
        else:
            raise NotImplementedError('get_count only implemented for bonds,'
                                      ' angles, dihedrals, impropers')

    def add(self, items):
        ''' adds to prop
        Parameters:
            items: dict of keys as type/resname, and keys as list of indices
                   pertaining to the key for resname, bonds, angles, dihedrals,
                   impropers;
        '''
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        # Deletes empty items
        _del_key = []
        for key, values in items.items():
            if len(values) == 0:
                _del_key.append(key)
        for key in _del_key:
            items.pop(key)
        if len(items) == 0:
            return

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if not self._ins.has(self.prop):
                array = [{} for i in range(len(self._ins))]
            else:
                array = self._ins.get_array(self.prop)

            for key, values in items.items():
                # make sure values are correct
                try:
                    _ = np.array(key, int)
                    _ = np.array(values, int)
                except ValueError as e:
                    raise ValueError('key should be int and '
                                     'values should be a list of '
                                     '{0}: '.format(self.prop) + e.args[0])
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                if len(_.shape) == 1:
                    values = [values]
                    _ = np.array(values, int)
                if _.shape[1] != length[self.prop]:
                    raise RuntimeError('{0} should be set of '
                                       '{1}'.format(self.prop, length[self.prop]))
                values_copy = deepcopy(values)

                if self.prop == 'angles':
                    indx = [i.pop(1) for i in values]
                else:
                    indx = [i.pop(0) for i in values]

                for i, j in enumerate(indx):
                    # check if the prop already exists
                    # its done here as array gets updated
                    exists = False
                    if self.prop in ['dihedrals', 'impropers']:
                        list_ = [y for x in array[j].values() for y in x]
                        exists = values[i] in list_
                    elif self.prop == 'bonds':
                        list_ = [y for x in array[j].values() for y in x]
                        list_1 = [y for x in array[values[i][0]].values()
                                  for y in x]
                        exists = [j] in list_1 or values[i] in list_
                    elif self.prop == 'angles':
                        list_ = [y for x in array[j].values() for y in x]
                        list_ += [list(reversed(y))
                                  for x in array[j].values() for y in x]
                        exists = values[i] in list_
                    if not exists:
                        array[j][key] = array[j].get(key, []) + [values[i]]
                    else:
                        print('{0} already exists'.format(values_copy[i]))

            self._ins.set_array(self.prop, array, object)
            self.update()
        elif self.prop == 'resnames':
            if self._ins.has('resnames'):
                resnames = self._ins.get_array(self.prop)
            else:
                resnames = ['' for _ in range(len(self._ins))]

            for key, vals in items.items():
                for val in vals:
                    resnames[val] = key
            self._ins.set_array(self.prop, resnames, object)
        else:
            raise NotImplementedError('{} does not support '
                                      'add'.format(self.prop))

    def update(self, _name_set=False):
        '''
        Parameters
            _name_set: bool
                when names are set, types, bonds etc needs updating
        '''
        def unique_ind(a):
            ''' Returns dict with new indices as values of old indices as keys
            if old index needs a change'''
            id_ = np.unique(a)
            ind_of = {}
            for i, val in enumerate(id_, start=1):
                if val != i:
                    ind_of[val] = i
            return ind_of

        if self.prop in ['bonds',
                         'angles',
                         'dihedrals',
                         'impropers']:
            # Correct if same prop has two types
            if self._ins.has(self.prop):
                types = self.get_types()
                if len(types) == 0:
                    # prop is empty
                    del self._ins.arrays[self.prop]
                    return
                if len(set(types.keys())) != len(set(types.values())):
                    # same prop has two types
                    rev_types = {}
                    for i in reversed(list(types.keys())):
                        rev_types[types[i]] = i
                    ind_of = {}
                    for i, j in types.items():
                        if i != rev_types[j]:
                            ind_of[i] = rev_types[j]
                    self.set_types_to(ind_of)

                ind_of = unique_ind(self.get_types(verbose=False))
                self.set_types_to(ind_of)

        elif self.prop == 'resnames':
            if self._ins.has(self.prop):
                if len(self.get_types()) == 0:
                    del self._ins.arrays[self.prop]

        elif self.prop == 'ids':
            self._ins.arrays[self.prop] = np.arange(len(self._ins)) + 1

        elif self.prop == 'mol-ids':
            if not self._ins.has(self.prop):
                self._ins.set_array('mol-ids',
                                    np.ones(len(self._ins)),
                                    int)
            ind_of = unique_ind(self._ins.arrays[self.prop])
            self._ins.set_array(self.prop,
                                [(ind_of[x] if x in ind_of else x)
                                 for x in self._ins.get_array(self.prop)],
                                int)

        elif self.prop == 'names':
            if not self._ins.has('names'):
                # types should be updated first
                # if types are defined
                # but not names, eg when reading lammps dump
                # names have to be different
                if (len(np.unique(self._ins.arrays['types']))
                        == len(np.unique(self._ins.numbers))):
                    self._ins.set_array('names',
                                        self._ins.get_chemical_symbols(),
                                        object)
                else:
                    names = self._ins.get_chemical_symbols()
                    types = self._ins.get_array('types')
                    type_of_name = {}
                    names_counter = {}
                    for i, j in enumerate(names):
                        if j in type_of_name:
                            if type_of_name[j] != types[i]:
                                if types[i] in type_of_name.values():
                                    _ = {k: l for l, k in type_of_name.items()}
                                    name_of_types = _
                                    names[i] = name_of_types[types[i]]
                                else:
                                    names_counter[j] = names_counter.get(j, 0) + 1
                                    names[i] += str(names_counter[j])
                                    type_of_name[names[i]] = types[i]
                        else:
                            type_of_name[j] = types[i]
                    self._ins.set_array('names',
                                        names,
                                        object)
            # update type when
            # same name has two types
            # happens during self.extend
            # or same type has two names
            types = self._ins.get_array('types')
            names = self._ins.get_array('names')
            types_dict = {}
            # ASE atoms with no atoms can be encountered
            # during ASE.__getitem__
            if len(types) != 0:
                n_max = np.max(types)
                for i in set(types):
                    name_ = sorted(np.unique(names[types == i]))
                    types_dict[i] = name_[0]
                    for j in name_[1:]:
                        # same type has two names
                        # then make it into case:
                        # same name has two types
                        # since j might already exist in other types
                        n_max += 1
                        types_dict[n_max] = j
                        types[np.logical_and(types == i, names == j)] = n_max
                # same name has two types
                rev_types = {} # names to types
                for i in reversed(list(types_dict.keys())):
                    rev_types[types_dict[i]] = i
                ind_of = {}
                for i, j in types_dict.items():
                    if i != rev_types[j]:
                        ind_of[i] = rev_types[j]
                self._ins.set_array('types',
                                    [(ind_of[x] if x in ind_of else x)
                                     for x in types],
                                    int)
                self._ins.topology['types'].update()

        if _name_set:
            # update bonds etc
            for prop in ['bonds',
                         'angles',
                         'dihedrals',
                         'impropers']:
                if not self._ins.has(prop):
                    continue
                items = self._ins.get_array(prop)
                types = self._ins.topology[prop].get_types()
                names = self._ins.arrays['names']
                rev_types = {j: i for i, j in types.items()}
                for i, item in enumerate(items):
                    item_ = deepcopy(item)
                    for key, values in item.items():
                        pop_ind = []
                        for j, value in enumerate(values):
                            eg_list = [i] + value
                            name_list = names[eg_list].tolist()
                            if self.prop == 'bonds':
                                name_list.sort()
                            if self.prop == 'angles':
                                # vertex atom is at first index
                                vertex = name_list.pop(0)
                                name_list.sort()
                                name_list.insert(1, vertex)
                            prop_name = '-'.join(name_list)
                            if prop_name in rev_types:
                                if rev_types[prop_name] != key:
                                    pop_ind.append(j)
                                    hold = item_.get(rev_types[prop_name], [])
                                    hold.append(value)
                                    item_[rev_types[prop_name]] = hold
                            else:
                                max_ = max(rev_types.values())
                                rev_types[prop_name] = max_ + 1
                                pop_ind.append(j)
                                hold = item_.get(rev_types[prop_name], [])
                                hold.append(value)
                                item_[rev_types[prop_name]] = hold
                        pop_ind = reversed(pop_ind)
                        for k in pop_ind:
                            item_[key].pop(k)
                        if item_[key] == []:
                            del item_[key]
                    items[i] = deepcopy(item_)
                self._ins.set_array(prop,
                                    items,
                                    object)
                self._ins.topology[prop].update()

        if self.prop == 'types':
            # if types get changed,
            # names can set types uniquely
            # to change types, change names first
            if not self._ins.has('types'):
                self._ins.set_array('types',
                                    self._ins.get_atomic_numbers(),
                                    int)
            ind_of = unique_ind(self._ins.arrays[self.prop])
            self._ins.set_array(self.prop,
                                [(ind_of[x] if x in ind_of else x)
                                 for x in self._ins.get_array(self.prop)],
                                int)

    @_check_exists
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

        if indx_of == {}:
            return

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            for indx in np.arange(len(self._ins))[index]:
                keys = self._ins.arrays[self.prop][indx].keys()
                new_keys = []
                for i in keys:
                    if i in indx_of.keys():
                        new_keys.append(indx_of[i])
                    else:
                        new_keys.append(i)
                values = [self._ins.arrays[self.prop][indx][x] for x in keys]
                self._ins.arrays[self.prop][indx] = dict(zip(new_keys, values))

        elif self.prop in ['mol-ids', 'types', 'names', 'resnames']:
            for indx in np.arange(len(self._ins))[index]:
                try:
                    _ = indx_of[self._ins.arrays[self.prop][indx]]
                except IndexError:
                    continue
                self._ins.arrays[self.prop][indx] = _
            self.update()

    @_check_exists
    def _set_indices_to(self, indx_of, index):
        # selecting set of ids to remove
        if self.prop not in ['bonds', 'angles', 'dihedrals', 'impropers']:
            return None
        ids = []
        for key, value in indx_of.items():
            if value is None:
                ids.append(key)
        for indx in np.arange(len(self._ins))[index]:
            item =  self._ins.arrays[self.prop][indx]
            # holds empty keys for deletion later
            del_key = []
            # extend() can bring int item
            # which gives AttributeError at item.items()
            try:
                for key, value in item.items():
                    # holds index of list that contains removed bonds etc
                    del_id = []
                    for i, j in enumerate(value):
                        if np.any([x in ids for x in j]):
                            del_id.append(i)
                        else:
                            for k, l in enumerate(j):
                                value[i][k] = indx_of[l]
                    del_id.sort(reverse=True)
                    for i in del_id:
                        value.pop(i)
                    # If value is empty then mark key for delition
                    # note that if it is empty, the arrays is not changed
                    # hence, delition is necessary
                    if len(value) == 0:
                        del_key.append(key)
                    else:
                        self._ins.arrays[self.prop][indx][key] = value
            except AttributeError:
                self._ins.arrays[self.prop][indx] = {}
            for i in del_key:
                self._ins.arrays[self.prop][indx].pop(i)


class _TopoAttributeProperty(object):

    def __init__(self, prop):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self.prop = prop

    def __get__(self, topo_base, owner):
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
    ids = _TopoAttributeProperty('ids')
    names = _TopoAttributeProperty('names')
    mol_ids = _TopoAttributeProperty('mol-ids')
    types = _TopoAttributeProperty('types')
    resnames = _TopoAttributeProperty('resnames')
    bonds = _TopoAttributeProperty('bonds')
    angles = _TopoAttributeProperty('angles')
    dihedrals = _TopoAttributeProperty('dihedrals')
    impropers = _TopoAttributeProperty('impropers')

    def __init__(self, instance):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = instance
        # a dict to hold all attributes
        self._prop_dict = {'names': self.names,
                           'ids': self.ids,
                           'mol-ids': self.mol_ids,
                           'types': self.types,
                           'resnames': self.resnames,
                           'bonds': self.bonds,
                           'angles': self.angles,
                           'dihedrals': self.dihedrals,
                           'impropers': self.impropers}
        # a dict to store only available properties
        # ids, mol-ids, names, and types should always be present
        self._dict = {}
        for key, val in self._prop_dict.items():
            if self._ins.has(key) or key in ['ids',
                                             'mol-ids',
                                             'names',
                                             'types']:
                self._dict[key] = val

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
        topo_dict = {}
        for key, values in self._dict.items():
            topo_dict[key] = values.get()

        return TopologyObject(topo_dict)

    __call__ = get_topology_object

    def update(self, topo_object=None, specorder=None):
        if topo_object is None:
            topo_dict = {}
        elif type(topo_object) is TopologyObject:
            topo_dict = topo_object._dict
        else:
            topo_dict = TopologyObject(topo_object)._dict

        if specorder is not None:
            order = np.unique(self._ins.get_atomic_numbers())
            if np.any(order != np.unique(specorder)):
                raise RuntimeError('Atomic numbers found in specorder'
                                   ' mismatch those found in system:'
                                   ' {}'.format(order))
            order_dict = dict(zip(order, specorder))
            self._ins.set_array('types',
                                [order_dict[i]
                                 for i in self._ins.get_atomic_numbers()],
                                int)

        # types should be updated before names
        # if types are defined
        # but not names, eg when reading lammps dump
        # names have to be different
        for prop in ['ids',
                     'types',
                     'names',
                     'mol-ids',
                     'resnames',
                     'bonds',
                     'angles',
                     'dihedrals',
                     'impropers']:
            if prop in topo_dict:
                self._dict[prop] = self._prop_dict[prop]
                self._dict[prop].set(topo_dict[prop])
            elif prop in self._dict:
                self._dict[prop].update()

    def generate_with_names(self, topo_dict, cutoffs=None):
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
            self._prop_dict[key].add({1: values})

        # update as if names has to be set
        # this updates properties with names
        self.names.update(_name_set=True)

    def _set_indices_to(self, indx_of, index=":"):
        '''sets indices in bonds, etc as specified in indx_of'''
        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass

        # removing bonds etc containing non-existing ids
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if prop in self._dict:
                self._dict[prop]._set_indices_to(indx_of, index)

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
        if self._ins.has('mol-ids'):
            n_molid = np.max(self._ins.arrays['mol-ids'])
            nmolids = 0

        topo_props = ['bonds', 'angles', 'dihedrals', 'impropers']

        for name in topo_props:
            if self._ins.has(name):
                a = self._ins.arrays[name][:n]
                self._ins.arrays[name] = np.empty(size_m * n, dtype='object')
                for i in range(size_m):
                    self._ins.arrays[name][i * n:(i + 1) * n] = deepcopy(a)

        i0 = 0
        natoms = 0
        indx_of = {}
        for m0 in range(m[0]):
            for m1 in range(m[1]):
                for m2 in range(m[2]):
                    i1 = i0 + n
                    for i in range(n):
                        indx_of[i] = i + natoms
                    self._set_indices_to(indx_of, "{}:{}".format(i0, i1))
                    if self._ins.has('mol-ids'):
                        _ = self._ins.arrays['mol-ids'][i0:i1] + nmolids
                        self._ins.arrays['mol-ids'][i0:i1] = _
                        nmolids += n_molid
                    i0 = i1
                    natoms += n

        self.update()

    def _extend(self, n1, n2):
        # raise types and mo-ids of other with the highest in self
        for prop in ['mol-ids', 'types']:
            self._ins.arrays[prop][n1:] += np.max(self._ins.arrays[prop][:n1])

        indx_of = {}
        for i in range(n2):
            indx_of[i] = i + n1
        self._set_indices_to(indx_of, "{}:".format(n1))

        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if prop in self._dict:
                try:
                    max_ = np.max(self._dict[prop].get_types(index=':{}'.format(n1),
                                                             verbose=False))
                except ValueError:
                    # get_types is empty
                    max_ = 0
                indx_of = {}
                for i in self._dict[prop].get_types(index='{}:'.format(n1),
                                                    verbose=False):
                    indx_of[i] = i + max_
                self._dict[prop].set_types_to(indx_of, index='{}:'.format(n1))

        # updating ids
        self.update()


class TopologyObject(object):
    '''Topology object that can be used to transfer topologies
    attached to Atoms object'''

    topo_props = ['ids',
                  'names',
                  'types',
                  'mol-ids',
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
                if i == 'types':
                    self.types = topology_dict[i]
                if i == 'mol-ids':
                    self.mol_ids = topology_dict[i]
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
    def types(self):
        return self._dict['types']

    @types.setter
    def types(self, values):
        try:
            types = np.array(values, dtype=int)
        except ValueError as e:
            raise ValueError(e, 'types should be integer')
        self._dict['types'] = types

    @property
    def mol_ids(self):
        return self._dict['mol-ids']

    @mol_ids.setter
    def mol_ids(self, values):
        try:
            mol_ids = np.array(values, dtype=int)
        except ValueError as e:
            raise ValueError(e, 'mol-ids should be integer')
        self._dict['mol-ids'] = mol_ids

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
