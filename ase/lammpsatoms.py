import numpy as np
from ase.atoms import Atoms
from ase.atom import Atom
from copy import deepcopy
import numbers
from ase.utils import natural_cutoffs, basestring
from ase.neighborlist import NeighborList
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
        return str(self.get_types())

    def __getitem__(self, item):
        if type(item) is str:
            reverse_type = {i:j for j, i in self.get_types().items()}
            item = reverse_type[item]
        return self.get()[item]

    @_check_exists
    def __delitem__(self, item):
        if type(item) is str:
            reverse_type = {i: j for j, i in self.get_types().items()}
            item = reverse_type[item]
        props = self.get()
        props.pop(item)
        self.set(props)

    def get_num_types(self):
        ''' returns number of types of prop: bonds, etc'''
        return len(self.get_types(verbose=False))

    @_check_exists
    def get_types(self, verbose=True):
        '''returns types of prop: bonds, etc
        :param prop: name of property
        :param verbose: gives name abbreviation for the types'''

        items = self._ins.arrays[self.prop]
        type_list = np.unique([i for x in items for i in x.keys()])

        if not verbose:
            return type_list

        names = self._ins.arrays['name']
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

    def get(self):

        d = {}
        for key in self.get_types(verbose=False):
            d[key] = []
        for i, item in enumerate(self._ins.arrays[self.prop]):
            for key, values in item.items():
                for value in values:
                    if self.prop == 'angles':
                        value = [value[0], i, value[1]]
                    else:
                        value = [i] + value
                    d[key].append(value)
        return d

    __call__ = get

    def set(self, value):
        '''
        '''
        if self._ins.has(self.prop):
            # delete array
            del self._ins.arrays[self.prop]
        self.add(value)

    @_check_exists
    def get_count(self):
        ''' returns number of prop: bonds, etc.'''
        items = self._ins.arrays[self.prop]
        values = [j for x in items for i in x.values() for j in i]
        return len(values)

    def add(self, items):
        ''' adds to prop
        Parameters:
            items: dict of keys as type/resname, and keys as list of indices
                   pertaining to the key
        '''
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        if not self._ins.has(self.prop):
            if items == {}:
                # instance does not have the prop, and
                # nothing will be added
                return
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

    def update(self):

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

        for key in indx_of.keys():
            for indx in np.arange(len(self._ins))[index]:
                if key in self._ins.arrays[self.prop][indx].keys():
                    _ = self._ins.arrays[self.prop][indx].pop(key)
                    old = self._ins.arrays[self.prop][indx].get(indx_of[key], [])
                    self._ins.arrays[self.prop][indx][indx_of[key]] = old + _

    @_check_exists
    def _set_indices_to(self, indx_of, index):
        # selecting set of ids to remove
        ids = []
        for key, value in indx_of.items():
            if value is None:
                ids.append(key)
        for indx, item in zip(index, self._ins.arrays[self.prop][index]):
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
        self.prop = prop

    def __get__(self, topo_base, owner):
        if topo_base is None:
            return self
        self._ins = topo_base._ins
        return _TopoAttribute(self)

    def __set__(self, topo_base, value):
        self._ins = topo_base
        _TopoAttribute(self).set(value)

    def __delete__(self, topo_base):
        del topo_base._ins.arrays[self.prop]


class _TopoBase(object):

    Bonds = _TopoAttributeProperty('bonds')
    Angles = _TopoAttributeProperty('angles')
    Dihedrals = _TopoAttributeProperty('dihedrals')
    Impropers = _TopoAttributeProperty('impropers')

    topo_props = ['bonds', 'angles', 'dihedrals', 'impropers']

    def __init__(self, topo_base_prop):
        self._ins = topo_base_prop._ins
        self._prop_dict = {'bonds': self.Bonds,
                           'angles': self.Angles,
                           'dihedrals': self.Dihedrals,
                           'impropers': self.Impropers}
        self.update()

    def __repr__(self):
        tokens = []
        for key, values in self._dict.items():
            tokens.append("{}= {}".format(key, values))
        return "{}.Topology({})".format(self._ins.__class__.__name__, ", ".join(tokens))

    def __getitem__(self, item):
        return self._prop_dict[item]

    def update(self):

        self._dict = {}
        for prop in self.topo_props:
            if self._ins.has(prop):
                self._dict[prop] = self._prop_dict[prop]
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

        symbols = self._ins.get_array('name')
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
        for prop in self._dict.values():
            prop._set_indices_to(indx_of, index)


class _TopoBaseProperty(object):

    def __get__(self, instance, owner):
        if instance is None:
            return self
        self._ins = instance
        return _TopoBase(self)

    def __delete__(self, instance):
        self._ins = instance
        topo = _TopoBase(self)
        for i in topo._dict.values():
            del i


class _Resname(object):

    def __init__(self, _resmame_prop):
        self._ins = _resmame_prop._ins

    def _check_exists(func):
        '''Decorator to check if the property exists'''
        # Only added to functions on which other functions already depend on
        # eg get_types, or to functions that don't depend on any other func
        def wrapper(*args, **kwargs):
            self = args[0]
            if not self._ins.has('resname'):
                raise KeyError('{0} object has no '
                                   'resname'.format(self._ins.__class__.__name__))
            return func(*args, **kwargs)
        return wrapper

    def __repr__(self):
        tokens = []
        for token in self.get_types():
            tokens.append("{}=...".format(token))
        return "{}.Resname({})".format(self._ins.__class__.__name__, ", ".join(tokens))

    def __getitem__(self, item):
        return self.get()[item]

    @_check_exists
    def __delitem__(self, i):
        if not isinstance(i, list):
            i = [i]
        resnames = self.get()
        for ii in i:
            resnames.pop(ii)
        self.set(resnames)

    def get(self):
        d = {}
        for resname in self.get_types():
            d[resname] = []
        for i, resnames in enumerate(self._ins.arrays['resname']):
            for resname in resnames:
                d[resname].append(i)
        return d

    __call__ = get

    def set(self, value):
        if self._ins.has('resname'):
            # delete array
            del self._ins.arrays['resname']
        self.add(value)

    @_check_exists
    def get_types(self):
        types = set()
        for i in self._ins.arrays['resname']:
            types |= i
        return np.asarray(list(types))

    def add(self, items):
        if not self._ins.has('resname'):
            array = [set() for _ in range(len(self._ins))]
        else:
            array = self._ins.get_array('resname')

        for key, values in items.items():
            for i in values:
                array[i] |= set([key])

        self._ins.set_array('resname', array, object)

class _ResnameProperty(object):

    def __get__(self, instance, owner):
        if instance is None:
            return self
        self._ins = instance
        return _Resname(self)

    def __set__(self, instance, value):
        self._ins = instance
        _Resname(self).set(value)

    def __delete__(self, instance):
        del self._ins.arrays['resname']


class LammpsAtoms(Atoms):
    '''
    Atoms class with methods that support lammps-property methods
    '''

    def __init__(self,
                 symbols=None,
                 id=None,
                 type=None,
                 name=None,
                 resname=None,
                 mol_id=None,
                 mmcharge=None,
                 bonds=None,
                 angles=None,
                 dihedrals=None,
                 impropers=None,
                 specorder=None,
                 *args, **kwargs):
        if isinstance(symbols, Atoms):
            if symbols.has('id') and id is None:
                id = symbols.get_array('id')
            if symbols.has('type') and type is None:
                type = symbols.get_array('type')
            if symbols.has('name') and name is None:
                name = symbols.get_array('name')
            if symbols.has('resname') and resname is None:
                resname = symbols.get_array('resname')
            if symbols.has('mol-id') and mol_id is None:
                mol_id = symbols.get_array('mol-id')
            if symbols.has('mmcharge') and mmcharge is None:
                mmcharge = symbols.get_array('mmcharge')
            if symbols.has('bonds') and bonds is None:
                bonds = symbols.get_array('bonds')
            if symbols.has('angles') and angles is None:
                angles = symbols.get_array('angles')
            if symbols.has('dihedrals') and dihedrals is None:
                dihedrals = symbols.get_array('dihedrals')
            if symbols.has('impropers') and impropers is None:
                impropers = symbols.get_array('impropers')

        Atoms.__init__(self, symbols, *args, **kwargs)

        if id is not None:
            self.set_array('id', id, int)
        if type is not None:
            self.set_array('type', type, int)
        if name is not None:
            # numpy string dtypes are truncated
            # numpy strings should be stored as objects
            self.set_array('name', name, object)
        if resname is not None:
            self.set_array('resname', resname, object)
        if mol_id is not None:
            self.set_array('mol-id', mol_id, int)
        if mmcharge is not None:
            self.set_array('mmcharge', mmcharge, float)
        if bonds is not None:
            self.set_array('bonds', bonds, object)
        if angles is not None:
            self.set_array('angles', angles, object)
        if dihedrals is not None:
            self.set_array('dihedrals', dihedrals, object)
        if impropers is not None:
            self.set_array('impropers', impropers, object)
        self.update(specorder)

    def update(self, specorder=None):
        '''updates id, mol-id and type to 1-Ntype'''

        self.arrays['id'] = np.arange(len(self)) + 1

        if not self.has('type'):
            self.set_array('type',
                           self.get_atomic_numbers(),
                           int)

        if not self.has('mol-id'):
            self.set_array('mol-id',
                           np.ones(len(self)),
                           int)

        if not self.has('name'):
            self.set_array('name',
                           self.get_chemical_symbols(),
                           object)

        for prop in ['mol-id', 'type']:
            ind_of = unique_ind(self.get_array(prop))
            self.set_array(prop,
                          [(ind_of[x] if x in ind_of else x) for x in self.get_array(prop)],
                          int)

        self.Topology.update()

        if specorder is not None:
            order = np.unique(self.get_atomic_numbers())
            if np.any(order != np.unique(specorder)):
                raise RuntimeError('Atomic numbers found in specorder'
                                   ' mismatch those found in system:'
                                   ' {}'.format(order))
            order_dict = dict(zip(order, specorder))
            self.set_array('type',
                           [order_dict[i] for i in self.get_atomic_numbers()],
                           int)

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

        self.Topology._set_indices_to(indx_of=indx_of)

        Atoms.__delitem__(self, i)

        # updating ids
        self.update()

    def __imul__(self, m):
        """In-place repeat of atoms."""
        if isinstance(m, int):
            m = (m, m, m)

        size_m = np.product(m)
        n = len(self)
        if self.has('mol-id'):
            n_molid = np.max(self.arrays['mol-id'])
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
                    self.Topology._set_indices_to(indx_of, "{}:{}".format(i0, i1))
                    if self.has('mol-id'):
                        _ = self.arrays['mol-id'][i0:i1] + nmolids
                        self.arrays['mol-id'][i0:i1] = _
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
        for prop in ['mol-id', 'type']:
            other.arrays[prop] += np.max(self.arrays[prop])
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop) and other.has(prop):
                indx_of = {}
                max_ = np.max(self.Topology[prop].get_types(verbose=False))
                for i in other.Topology[prop].get_types(verbose=False):
                    indx_of[i] = i + max_
                other.Topology[prop].set_types_to(indx_of)

        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if not self.has(prop) and other.has(prop):
                self.set_array(prop, [{} for _ in range(len(self))], object)

        if not self.has('resname') and other.has('resname'):
            self.set_array('resname', [set([]) for _ in range(len(self))], object)

        Atoms.extend(self, other)

        indx_of = {}
        for i in range(n2):
            indx_of[i] = i + n1
        self.Topology._set_indices_to(indx_of, "{}:".format(n1))

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

        topo_props = ['name',
                      'resname',
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
        atoms.Topology._set_indices_to(indx_of)
        atoms.update()

        return atoms

    Topology = _TopoBaseProperty()
    Resname = _ResnameProperty()
