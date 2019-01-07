import numpy as np
from ase.atoms import Atoms
from ase.atom import Atom
from copy import deepcopy
import numbers


class LammpsAtoms(Atoms):
    '''
    Atoms class with methods that support lammps-property methods
    '''

    def __init__(self,
                 symbols=None,
                 *args,
                 id=None,
                 type=None,
                 mol_id=None,
                 mmcharge=None,
                 bonds=None,
                 angles=None,
                 dihedrals=None,
                 impropers=None,
                 specorder=None, **kwargs):
        if isinstance(symbols, Atoms):
            if symbols.has('id') and id is None:
                id = symbols.get_prop('id')
            if symbols.has('type') and type is None:
                type = symbols.get_prop('type')
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
        return len(self.get_types(prop))

    def get_types(self, prop):
        '''returns types of prop: bonds, etc'''
        if not self.has(prop):
            return []
        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            items = self.arrays[prop]
            return np.unique([i for x in items for i in x.keys()])
        else:
            return np.unique(self.get_array(prop))

    def get_prop(self, prop):
        if not self.has(prop):
            return {}

        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            d = {}
            for key in self.get_types(prop):
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
        else:
            return self.get_array(prop)

    def get_num_prop(self, prop):
        ''' returns number of prop: bonds, etc.'''
        if not self.has(prop):
            return 0
        items = self.arrays[prop]
        values = [j for x in items for i in x.values() for j in i]
        return len(values)

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

                if prop == 'angles':
                    indx = [i.pop(1) for i in values]
                else:
                    indx = [i.pop(0) for i in values]

                for i, j in enumerate(indx):
                    array[j][key] = array[j].get(key, []) + [values[i]]

            self.set_array(prop, array, 'object')
            self.update()
        else:
            raise NotImplementedError('add_prop not implemented for '
                                      '{0}'.format(prop))

    def set_prop(self, prop, value, dtype=None):
        if prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
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
        for indx, item in enumerate(self.arrays[prop][slice(*index)],
                                            start=index[0]):
            for key in item.keys():
                if key in indx_of.keys():
                    _ = self.arrays[prop][indx].pop(key)
                    self.arrays[prop][indx][indx_of[key]] = _

    def update(self, specorder=None):
        '''updates id, mol-id and type to 1-Ntype'''

        def unique_ind(a):
            id_ = np.unique(a)
            d = {}
            for i, val in enumerate(id_, start=1):
                d[val] = i
            return d

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

        self.arrays['id'] = np.arange(len(self)) + 1

        if not self.has('type'):
            self.set_array('type',
                           self.get_atomic_numbers(),
                           int)

        if not self.has('mol-id'):
            self.set_array('mol-id',
                           np.ones(len(self)),
                           int)

        for prop in ['mol-id', 'type']:
            d = unique_ind(self.get_array(prop))
            self.set_prop(prop,
                          [d[x] for x in self.get_array(prop)],
                          int)

        for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
            if self.has(prop):
                d = unique_ind(self.get_types(prop))
                for index, item in enumerate(self.get_array(prop)):
                    for key in sorted(item.keys()):
                        _ = self.arrays[prop][index].pop(key)
                        self.arrays[prop][index][d[key]] = _

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
                max_ = np.max(self.get_types(prop))
                for i in other.get_types(prop):
                    indx_of[i] = i + max_
                other.set_types_to(prop, indx_of)

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

        # Converting to LammpsAtoms now not earlier,
        # since if single item is required, then
        # converting Atom to LammpsAtoms causes errors
        atoms = LammpsAtoms(atoms)

        lammps_props = ['bonds', 'angles', 'dihedrals', 'impropers']
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
