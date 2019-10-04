import numpy as np
from ase.atoms import Atoms
from ase.atom import Atom
from copy import deepcopy
import numbers
from ase.utils import basestring
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.io.formats import string2index


def unique_ind(a):
    ''' Returns dict with new indices as values of old indices as keys
    if old index needs a change'''
    id_ = np.unique(a)
    ind_of = {}
    for i, val in enumerate(id_, start=1):
        if val != i:
            ind_of[val] = i
    return ind_of


class _TopoAttribute(object):

    def __init__(self, topo_attr_prop):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = topo_attr_prop._ins
        self.prop = topo_attr_prop.prop

    def _check_exists(func):
        '''Decorator to check if the property exists'''
        # Only added to functions on which other functions already depend on
        # eg get_types, or to functions that don't depend on any other func
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self._ins.has(self.prop):
                raise KeyError('{0} object has no '
                                   '{1}'.format(self._ins.__class__.__name__,
                                                self.prop))
            return func(*args, **kwargs)
        return wrapper

    def __repr__(self):
        if self.prop == 'ids':
            return '...'
        else:
            return str(self.get_types())

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
        if self.prop in ['resname',
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

    def get_types(self, verbose=True):
        '''returns types of prop: bonds, etc
        :param prop: name of property
        :param verbose: gives name abbreviation for the types'''
        if not self._ins.has(self.prop):
            return np.array([])

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            items = self._ins.arrays[self.prop]
            type_list = np.unique([i for x in items for i in x.keys()])

            if not verbose:
                return type_list

            names = self._ins.arrays['names']
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
            types = set()
            for i in self._ins.arrays[self.prop]:
                types |= i
            return np.asarray(list(types))
        else:
            return np.unique(self._ins.get_array(self.prop))

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
            for i, resnames in enumerate(self._ins.arrays['resnames']):
                for resname in resnames:
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
        if self.prop == 'ids':
            self.update()
            raise NotImplementedError('changing ids shuffles the atoms,'
                                      'which is not implemented yet')
        elif self.prop == 'names':
            self._ins.set_array(self.prop, value, object)
        elif self.prop in ['types', 'mol-ids']:
            self._ins.set_array(self.prop, value, int)
        elif self.prop in ['resnames',
                           'bonds',
                           'angles',
                           'dihedrals',
                           'impropers']:
            self.add(value)
        self.update()

    @_check_exists
    def get_count(self):
        ''' returns number of prop: bonds, etc.'''
        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
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
            if not self._ins.has('resnames'):
                array = [set() for _ in range(len(self._ins))]
            else:
                array = self._ins.get_array('resnames')

            for key, values in items.items():
                for i in values:
                    array[i] |= set([key])

            self._ins.set_array('resnames', array, object)
        else:
            raise NotImplementedError('{} does not support '
                                      'add'.format(self.prop))

    def update(self):

        if self.prop in ['bonds',
                         'angles',
                         'dihedrals',
                         'impropers']:
            if not self._ins.has(self.prop):
                self._ins.set_array(self.prop,
                                    [{} for _ in range(len(self._ins))],
                                    object)
            # Correct if same prop has two types
            types = self.get_types()
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

        elif self.prop == 'ids':
            self._ins.arrays[self.prop] = np.arange(len(self._ins)) + 1

        elif self.prop == 'mol-ids':
            if not self._ins.has(self.prop):
                self._ins.set_array('mol-ids',
                                    np.ones(len(self)),
                                    int)

        elif self.prop == 'resnames':
            if not self._ins.has(self.prop):
                self._ins.set_array(self.prop,
                                    [set() for _ in range(len(self._ins))],
                                    object)

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

        if self.prop == 'types':
            # if types get changed,
            # names can set types uniquely
            # to change types, change names first
            if not self._ins.has('types'):
                self._ins.set_array('types',
                                    self._ins.get_atomic_numbers(),
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

        if self.prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            # keys are reversed so that double change is stopped
            # eg {1:2, 2:3} would change 1 to 3, which is not intended
            for key in reversed(list(indx_of.keys())):
                for indx in np.arange(len(self._ins))[index]:
                    if key in self._ins.arrays[self.prop][indx].keys():
                        _ = self._ins.arrays[self.prop][indx].pop(key)
                        old = self._ins.arrays[self.prop][indx].get(indx_of[key], [])
                        self._ins.arrays[self.prop][indx][indx_of[key]] = old + _
        elif self.prop in ['mol-ids', 'types', 'names']:
            for indx in np.arange(len(self._ins))[index]:
                try:
                    _ = indx_of[self._ins.arrays[self.prop][indx]]
                except IndexError as e:
                    continue
                self._ins.arrays[self.prop][indx] = _
                self.update()
        elif self.prop == 'resname':
            for indx in np.arange(len(self._ins))[index]:
                for resname in self._ins.arrays[self.prop][indx]:
                    if resname in indx_of.keys():
                        self._ins.arrays[self.prop][indx] -= set([resname])
                        self._ins.arrays[self.prop][indx] |= set([indx_of[resname]])

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
    Ids = _TopoAttributeProperty('ids')
    Names = _TopoAttributeProperty('names')
    Mol_ids = _TopoAttributeProperty('mol-ids')
    Types = _TopoAttributeProperty('types')
    Resnames = _TopoAttributeProperty('resnames')
    Bonds = _TopoAttributeProperty('bonds')
    Angles = _TopoAttributeProperty('angles')
    Dihedrals = _TopoAttributeProperty('dihedrals')
    Impropers = _TopoAttributeProperty('impropers')

    def __init__(self, instance):
        # every trivial call goes through this step
        # Thus, init should not be computationally intensive
        self._ins = instance
        self._dict = {'names': self.Names,
                      'ids': self.Ids,
                      'mol-ids': self.Mol_ids,
                      'types': self.Types,
                      'resnames': self.Resnames,
                      'bonds': self.Bonds,
                      'angles': self.Angles,
                      'dihedrals': self.Dihedrals,
                      'impropers': self.Impropers}

    def __repr__(self):
        tokens = []
        for key, values in self._dict.items():
            tokens.append("{}= {}".format(key, values))
        return "{}.Topology({})".format(self._ins.__class__.__name__, ", ".join(tokens))

    def __getitem__(self, item):
        return self._dict[item]

    def get_topology_dict(self):
        '''Gives topology dict that can be inherited by other topology
        classes'''
        topo_dict = {}
        for key, values in self._dict.items():
            topo_dict[key] = values.get()

        return topo_dict

    __call__ = get_topology_dict

    def update(self, topo_dict={}, specorder=None):
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
                self._dict[prop].set(topo_dict[prop])
            else:
                self._dict[prop].update()

    def generate(self, topo_dict, cutoffs=None):
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
            self.Bonds.add(bonds)

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
            self.Angles.add(angles)

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
            self.Dihedrals.add(dihedrals)

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
            self.Impropers.add(impropers)

    def _set_indices_to(self, indx_of, index=":"):
        '''sets indices in bonds, etc as specified in indx_of'''
        if isinstance(index, basestring):
            try:
                index = string2index(index)
            except ValueError:
                pass

        # removing bonds etc containing non-existing ids
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            self._dict[prop]._set_indices_to(indx_of, index)


class TopoAtoms(Atoms):
    '''
    Atoms class with methods that support Topology-property methods
    '''

    def __init__(self,
                 symbols=None,
                 topo_dict={},
                 specorder=None,
                 *args, **kwargs):
        if isinstance(symbols, Atoms) and symbols.has('id'):
            # topology exists
            topo_dict.update(symbols.topology())

        Atoms.__init__(self, symbols, *args, **kwargs)

        if topo_dict:
            self.topology = topo_dict
            self.update(specorder)

    def update(self, specorder=None):
       self.topology.update(specorder)

    @property
    def types(self):
        return self.arrays['types']

    @types.setter
    def types(self, other):
        self.set_array('types', other, int)

    @property
    def mol_ids(self):
        return self.arrays['mol-ids']

    @mol_ids.setter
    def mol_ids(self, other):
        self.set_array('mol-ids', other, int)

    @property
    def names(self):
        return self.arrays['names']

    @names.setter
    def names(self, other):
        self.set_array('names', other, int)

    def __delitem__(self, i=-1):
        if not isinstance(i, list):
            i = [i]
        # making index map dictionary, with index to delete as None
        # and setting indices to the map
        indx_of = {}
        count = 0
        for j in range(len(self)):
            if j in i:
                indx_of[j] = None
                count += 1
            else:
                indx_of[j] = j - count

        self.topology._set_indices_to(indx_of=indx_of)

        Atoms.__delitem__(self, i)

        # updating ids
        self.update()

    def __imul__(self, m):
        """In-place repeat of atoms."""
        if isinstance(m, int):
            m = (m, m, m)

        size_m = np.product(m)
        n = len(self)
        if self.has('mol-ids'):
            n_molid = np.max(self.arrays['mol-ids'])
            nmolids = 0

        Atoms.__imul__(self, m)

        topo_props = ['bonds', 'angles', 'dihedrals', 'impropers']

        for name in topo_props:
            if self.has(name):
                a = self.arrays[name][:n]
                self.arrays[name] = np.empty(size_m * n, dtype='object')
                for i in range(size_m):
                    self.arrays[name][i * n:(i + 1) * n] = deepcopy(a)

        i0 = 0
        natoms = 0
        indx_of = {}
        for m0 in range(m[0]):
            for m1 in range(m[1]):
                for m2 in range(m[2]):
                    i1 = i0 + n
                    for i in range(n):
                        indx_of[i] = i + natoms
                    self.topology._set_indices_to(indx_of, "{}:{}".format(i0, i1))
                    if self.has('mol-ids'):
                        _ = self.arrays['mol-ids'][i0:i1] + nmolids
                        self.arrays['mol-ids'][i0:i1] = _
                        nmolids += n_molid
                    i0 = i1
                    natoms += n

        # updating ids
        self.update()

        return self

    def extend(self,
               other):
        """Extend atoms object by appending atoms from *other*.
        Parameters:
            other: the other object
            extend_prop: list of properties to extend"""
        if isinstance(other, Atom):
            other = self.__class__([other])

        if type(other) != self.__class__:
            other = self.__class__(other)
        other = other.copy()
        other.update()
        self.update()

        n1 = len(self)
        n2 = len(other)
        for prop in ['mol-ids', 'types']:
            other.arrays[prop] += np.max(self.arrays[prop])
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop) and other.has(prop):
                indx_of = {}
                max_ = np.max(self.topology[prop].get_types(verbose=False))
                for i in other.topology[prop].get_types(verbose=False):
                    indx_of[i] = i + max_
                other.topology[prop].set_types_to(indx_of)

        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if not self.has(prop) and other.has(prop):
                self.set_array(prop, [{} for _ in range(len(self))], object)

        if not self.has('resnames') and other.has('resnames'):
            self.set_array('resnames', [set([]) for _ in range(len(self))], object)

        Atoms.extend(self, other)

        indx_of = {}
        for i in range(n2):
            indx_of[i] = i + n1
        self.topology._set_indices_to(indx_of, "{}:".format(n1))

        # updating ids
        self.update()

        return self

    __iadd__ = extend

    def copy(self):
        """Return a copy."""
        atoms = Atoms.copy(self)

        topo_props = ['bonds', 'angles', 'dihedrals', 'impropers']
        for name in topo_props:
            if self.has(name):
                atoms.arrays[name] = deepcopy(self.arrays[name])
        return atoms

    def get_array(self, name, copy=True):
        """Get an array.

        Returns a copy unless the optional argument copy is false.
        """
        if copy:
            return deepcopy(self.arrays[name])
        else:
            return self.arrays[name]

    def __getitem__(self, item):
        atoms = Atoms.__getitem__(self, item)

        if isinstance(item, numbers.Integral):
            return atoms
        elif not isinstance(item, slice):
            item = np.array(item)
            if item.dtype == bool:
                item = np.arange(len(self))[item]

        topo_props = ['names',
                      'resnames',
                      'bonds',
                      'angles',
                      'dihedrals',
                      'impropers']
        for name in topo_props:
            if self.has(name):
                atoms.arrays[name] = deepcopy(self.arrays[name][item])

        indx_of = {i: None for i in range(len(self))}
        count = 0
        for i in np.array(range(len(self)))[item]:
            indx_of[i] = count
            count += 1
        atoms.topology._set_indices_to(indx_of)
        atoms.update()

        return atoms

    def get_topology(self):
        return Topology(self)

    def set_topology(self, value={}):
        top = Topology(self)
        top.update(value)

    def _del_topology(self):
        top = Topology(self)
        for i in top._dict.keys():
            del self.arrays[i]

    topology = property(get_topology, set_topology,
                        _del_topology, doc='handles topology information of atoms')
