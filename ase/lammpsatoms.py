import numpy as np
from ase.atoms import Atoms
from ase.atom import Atom
from copy import deepcopy
import numbers
from ase.utils import natural_cutoffs
from ase.neighborlist import NeighborList


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
                id = symbols.get_prop('id')
            if symbols.has('type') and type is None:
                type = symbols.get_prop('type')
            if symbols.has('name') and name is None:
                name = symbols.get_prop('name')
            if symbols.has('resname') and resname is None:
                resname = symbols.get_prop('resname')
            if symbols.has('mol-id') and mol_id is None:
                mol_id = symbols.get_prop('mol-id')
            if symbols.has('mmcharge') and mmcharge is None:
                mmcharge = symbols.get_prop('mmcharge')
            if symbols.has('bonds') and bonds is None:
                bonds = symbols.get_prop('bonds')
            if symbols.has('angles') and angles is None:
                angles = symbols.get_prop('angles')
            if symbols.has('dihedrals') and dihedrals is None:
                dihedrals = symbols.get_prop('dihedrals')
            if symbols.has('impropers') and impropers is None:
                impropers = symbols.get_prop('impropers')

        Atoms.__init__(self, symbols, *args, **kwargs)

        if id is not None:
            self.set_prop('id', id, int)
        if type is not None:
            self.set_prop('type', type, int)
        if name is not None:
            # numpy string dtypes are truncated
            # numpy strings should be stored as objects
            self.set_prop('name', name, object)
        if resname is not None:
            self.set_prop('resname', resname, object)
        if mol_id is not None:
            self.set_prop('mol-id', mol_id, int)
        if mmcharge is not None:
            self.set_prop('mmcharge', mmcharge, float)
        if bonds is not None:
            self.set_prop('bonds', bonds)
        if angles is not None:
            self.set_prop('angles', angles)
        if dihedrals is not None:
            self.set_prop('dihedrals', dihedrals)
        if impropers is not None:
            self.set_prop('impropers', impropers)
        self.update(specorder)

    def get_num_types(self, prop):
        ''' returns number of types of prop: bonds, etc'''
        return len(self.get_types(prop, verbose=False))

    def get_types(self, prop, verbose=True):
        '''returns types of prop: bonds, etc
        :param prop: name of property
        :param verbose: gives name abbreviation for the types'''
        if not self.has(prop):
            return []

        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            items = self.arrays[prop]
            type_list = np.unique([i for x in items for i in x.keys()])

            if not verbose:
                return type_list

            names = self.get_prop('name')
            types = {}
            for key in type_list:
                for i, item in enumerate(items):
                    if key in item.keys():
                        eg_list = [i] + item[key][0]
                        break
                name_list = names[eg_list].tolist()
                if prop == 'bonds':
                    name_list.sort()
                if prop == 'angles':
                    # vertex atom is at first index
                    vertex = name_list.pop(0)
                    name_list.sort()
                    name_list.insert(1, vertex)
                types[key] = '-'.join(name_list)
            return types

        elif prop == 'resname':
            types = set()
            for i in self.arrays[prop]:
                types |= i
            return np.asarray(list(types))
        else:
            return np.unique(self.get_array(prop))

    def get_prop(self, prop):
        if not self.has(prop):
            raise RuntimeError('{0} object has no '
                               '{1}'.format(self.__class__.__name__, prop))

        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            d = {}
            for key in self.get_types(prop, verbose=False):
                d[key] = []
            for i, item in enumerate(self.get_array(prop)):
                for key, values in item.items():
                    for value in values:
                        if prop == 'angles':
                            value = [value[0], i, value[1]]
                        else:
                            value = [i] + value
                        d[key].append(value)
            return d
        elif prop == 'resname':
            d = {}
            for resname in self.get_types(prop, verbose=False):
                d[resname] = []
            for i, resnames in enumerate(self.get_array(prop)):
                for resname in resnames:
                    d[resname].append(i)
            return d
        else:
            return self.get_array(prop)

    def get_num_prop(self, prop):
        ''' returns number of prop: bonds, etc.'''
        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if not self.has(prop):
                return 0
            items = self.arrays[prop]
            values = [j for x in items for i in x.values() for j in i]
            return len(values)
        else:
            raise NotImplementedError('get_num_prop not implemented for '
                                      '{0}'.format(prop))

    def add_prop(self, prop, items):
        ''' adds to prop
        Parameters:
            prop: name of property
            items: dict of keys as type/resname, and keys as list of indices
                   pertaining to the key
        '''
        length = {'bonds': 2,
                  'angles': 3,
                  'dihedrals': 4,
                  'impropers': 4}

        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if not self.has(prop):
                array = [{} for i in range(len(self))]
            else:
                array = self.get_array(prop)

            for key, values in items.items():
                # make sure values are correct
                try:
                    _ = np.array(key, int)
                    _ = np.array(values, int)
                except ValueError as e:
                    raise ValueError('key should be int and '
                                     'values should be a list of '
                                     '{0}: '.format(prop) + e.args[0])
                if isinstance(values, np.ndarray):
                    values = values.tolist()
                if len(_.shape) == 1:
                    values = [values]
                    _ = np.array(values, int)
                if _.shape[1] != length[prop]:
                    raise RuntimeError('{0} should be set of '
                                       '{1}'.format(prop, length[prop]))
                values_copy = deepcopy(values)

                if prop == 'angles':
                    indx = [i.pop(1) for i in values]
                else:
                    indx = [i.pop(0) for i in values]

                for i, j in enumerate(indx):
                    # check if the prop already exists
                    # its done here as array gets updated
                    exists = False
                    if prop in ['dihedrals', 'impropers']:
                        list_ = [y for x in array[j].values() for y in x]
                        exists = values[i] in list_
                    elif prop == 'bonds':
                        list_ = [y for x in array[j].values() for y in x]
                        list_1 = [y for x in array[values[i][0]].values()
                                  for y in x]
                        exists = [j] in list_1 or values[i] in list_
                    elif prop == 'angles':
                        list_ = [y for x in array[j].values() for y in x]
                        list_ += [list(reversed(y))
                                  for x in array[j].values() for y in x]
                        exists = values[i] in list_
                    if not exists:
                        array[j][key] = array[j].get(key, []) + [values[i]]
                    else:
                        print('{0} already exists'.format(values_copy[i]))

            self.set_array(prop, array, object)
        elif prop == 'resname':
            if not self.has(prop):
                array = [set() for _ in range(len(self))]
            else:
                array = self.get_array(prop)

            for key, values in items.items():
                for i in values:
                    array[i] |= set([key])

            self.set_array(prop, array, object)
        else:
            raise NotImplementedError('add_prop not implemented for '
                                      '{0}'.format(prop))

    def generate_topology(self, topo_dict):
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
        nl = NeighborList(natural_cutoffs(self),
                          self_interaction=False,
                          bothways=True)
        nl.update(self)
        neighbor_matrix = nl.get_connectivity_matrix().toarray()

        symbols = self.get_prop('name')
        d = []
        for i, array in enumerate(neighbor_matrix):
            neighbor_list = np.where(array)[0]
            d.append(neighbor_list)

        if 'bonds' in topo_dict:
            seen_bonds = []
            bonds = {}
            for i, neighbor in enumerate(d):
                for j in neighbor:
                    name_list = [symbols[x] for x in [i, j]]
                    name_list.sort()
                    name_list = '-'.join(name_list)
                    if name_list in topo_dict['bonds'] and not [i, j] in seen_bonds:
                        indx = topo_dict['bonds'].index(name_list) + 1
                        bonds[indx] = bonds.get(indx, []) + [[i, j]]
                        seen_bonds.append([i, j])
                        seen_bonds.append([j, i])
            self.add_prop('bonds', bonds)

        if 'angles' in topo_dict:
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
                            indx = topo_dict['angles'].index(name_list) + 1
                            angles[indx] = angles.get(indx, []) + [[k, i, j]]
            self.add_prop('angles', angles)

        if 'dihedrals' in topo_dict:
            dihedrals = {}
            for i, neighbor_i in enumerate(d):
                for j in neighbor_i:
                    for k in set(d[j]) - set([i, j]):
                        for l in set(d[k]) - set([i, j, k]):
                            name_list = [symbols[x] for x in [i, j, k, l]]
                            name_list.sort()
                            name_list = '-'.join(name_list)
                            if name_list in topo_dict['dihedrals']:
                                indx = topo_dict['dihedrals'].index(name_list) + 1
                                dihedrals[indx] = dihedrals.get(indx, []) + [[i, j, k, l]]
            self.add_prop('dihedrals', dihedrals)

        if 'impropers' in topo_dict:
            impropers = {}
            for i, neighbor in enumerate(d):
                for indx_j, j in enumerate(neighbor):
                    for indx_k, k in enumerate(neighbor[indx_j + 1:], start=indx_j + 1):
                        for l in neighbor[indx_k + 1:]:
                            name_list = [symbols[x] for x in [i, j, k, l]]
                            name_list.sort()
                            name_list = '-'.join(name_list)
                            if name_list in topo_dict['impropers']:
                                indx = topo_dict['impropers'].index(name_list) + 1
                                impropers[indx] = impropers.get(indx, []) + [[i, j, k, l]]
            self.add_prop('impropers', impropers)

    def set_prop(self, prop, value, dtype=None):
        '''
        :param prop: Name of property
        :param value: dict of keys as type/resname, and keys as list of
                      indices pertaining to the key
        :param dtype: (int/float/'object') type of values;
                      'object' in case of bonds, angles, impropers, and
                      dihedrals
        :return: None
        '''
        if prop in ['resname', 'bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop):
                # delete array
                del self.arrays[prop]
            self.add_prop(prop, value)
        else:
            self.set_array(prop, value, dtype)

    def _set_indices_to(self, indx_of, index=[0, None]):
        '''sets indices in bonds, etc as specified in indx_of'''
        # selecting set of ids to remove
        ids = []
        for key, value in indx_of.items():
            if value is None:
                ids.append(key)

        # removing bonds etc containing non-existing ids
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop):
                for indx, item in enumerate(self.arrays[prop][slice(*index)],
                                            start=index[0]):
                    # holds empty keys for delition later
                    del_key = []
                    # extend can bring int item
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
                                self.arrays[prop][indx][key] = value
                    except AttributeError:
                        self.arrays[prop][indx] = {}
                    for i in del_key:
                        self.arrays[prop][indx].pop(i)

    def set_types_to(self, prop, indx_of, index=[0, None]):
        '''
        :param prop: property name
        :param indx_of: dictionary, changes type from keys -> values
        :param index: list of start and stop index to affect the change
        '''
        if not self.has(prop):
            raise RuntimeError('{0} object has no '
                               '{1}'.format(self.__class__.__name__, prop))

        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            for indx, item in enumerate(self.arrays[prop][slice(*index)],
                                                start=index[0]):
                for key in item.keys():
                    if key in indx_of.keys():
                        _ = self.arrays[prop][indx].pop(key)
                        old = self.arrays[prop][indx].get(indx_of[key], [])
                        self.arrays[prop][indx][indx_of[key]] = old + _
        elif prop in ['mol-id', 'type', 'name']:
            for indx, item in enumerate(self.arrays[prop][slice(*index)],
                                        start=index[0]):
                if item in indx_of.keys():
                    self.arrays[prop][indx] = indx_of[item]
        elif prop == 'resname':
            for indx, resnames in enumerate(self.arrays[prop][slice(*index)],
                                        start=index[0]):
                for resname in resnames:
                    if resname in indx_of.keys():
                        self.arrays[prop][indx] -= set([resname])
                        self.arrays[prop][indx] |= set([indx_of[resname]])
        else:
            raise NotImplementedError('set_types_to not implemented for '
                                      '{0}'.format(prop))

    def update(self, specorder=None):
        '''updates id, mol-id and type to 1-Ntype'''

        def unique_ind(a):
            id_ = np.unique(a)
            if np.all(id_ == np.arange(len(id_)) + 1):
                return None
            d = {}
            for i, val in enumerate(id_, start=1):
                d[val] = i
            return d

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
            d = unique_ind(self.get_array(prop))
            if d:
                self.set_prop(prop,
                              [d[x] for x in self.get_array(prop)],
                              int)

        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop):
                d = unique_ind(self.get_types(prop, verbose=False))
                if d:
                    for index, item in enumerate(self.get_array(prop)):
                        for key in sorted(item.keys()):
                            _ = self.arrays[prop][index].pop(key)
                            self.arrays[prop][index][d[key]] = _

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

        self._set_indices_to(indx_of=indx_of)

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

        lammps_props = ['bonds', 'angles', 'dihedrals', 'impropers']

        for name in lammps_props:
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
                    self._set_indices_to(indx_of, [i0, i1])
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
               other,
               extend_prop=('mol-id',
                            'type',
                            'bonds',
                            'angles',
                            'dihedrals',
                            'impropers')):
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
            if prop in extend_prop:
                other.arrays[prop] += np.max(self.arrays[prop])
        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop) and other.has(prop) and prop in extend_prop:
                indx_of = {}
                max_ = np.max(self.get_types(prop, verbose=False))
                for i in other.get_types(prop, verbose=False):
                    indx_of[i] = i + max_
                other.set_types_to(prop, indx_of)

        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if not self.has(prop) and other.has(prop):
                self.set_array(prop, [{} for _ in range(len(self))], object)

        if not self.has('resname') and other.has('resname'):
            self.set_array('resname', [set([]) for _ in range(len(self))], object)

        Atoms.extend(self, other)

        indx_of = {}
        for i in range(n2):
            indx_of[i] = i + n1
        self._set_indices_to(indx_of, [n1, None])

        # updating ids
        self.update()

        return self

    __iadd__ = extend

    def copy(self):
        """Return a copy."""
        atoms = Atoms.copy(self)

        lammps_props = ['bonds', 'angles', 'dihedrals', 'impropers']
        for name in lammps_props:
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

        lammps_props = ['name', 'bonds', 'angles', 'dihedrals', 'impropers']
        for name in lammps_props:
            if self.has(name):
                atoms.arrays[name] = deepcopy(self.arrays[name][item])

        indx_of = {i: None for i in range(len(self))}
        count = 0
        for i in np.array(range(len(self)))[item]:
            indx_of[i] = count
            count += 1
        atoms._set_indices_to(indx_of)
        atoms.update()

        return atoms
