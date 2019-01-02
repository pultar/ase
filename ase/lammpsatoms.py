import numpy as np
from ase.atoms import Atoms
from ase.atom import Atom
from copy import deepcopy


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
                 impropers=None, **kwargs):
        if isinstance(symbols, Atoms):
            if symbols.has('id') and id is None:
                id = symbols.get_array('id')
            if symbols.has('type') and type is None:
                type = symbols.get_array('type')
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
        if mol_id is not None:
            self.set_array('mol-id', mol_id, int)
        if mmcharge is not None:
            self.set_array('mmcharge', mmcharge, float)
        if bonds is not None:
            self.set_array('bonds', bonds, 'object')
        if angles is not None:
            self.set_array('angles', angles, 'object')
        if dihedrals is not None:
            self.set_array('dihedrals', dihedrals, 'object')
        if impropers is not None:
            self.set_array('impropers', impropers, 'object')
        self.update()

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

    def get_num_prop(self, prop):
        ''' returns number of prop: bonds, etc.'''
        if not self.has(prop):
            return 0
        items = self.arrays[prop]
        values = [j for x in items for i in x.values() for j in i]
        return len(values)

    def _set_indices_to(self, indx_of=None, index=[None, None]):
        '''sets indices in bonds, etc as specified in indx_of'''
        if indx_of is None:
            return

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

    def update(self):
        '''updates id, mol-id and type to 1-Ntype'''

        def unique(a):
            id_ = np.unique(a)
            if not np.all(id_ == np.arange(id_.size) + 1):
                a = np.array([np.where(id_ == i)[0][0] + 1 for i in a])
            return a

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
            self.arrays[prop] = unique(self.get_array(prop))

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

        M = np.product(m)
        n = len(self)
        if self.has('mol-id'):
            n_molid = np.max(self.arrays['mol-id'])
            nmolids = 0

        Atoms.__imul__(self, m)

        lammps_props = ['bonds', 'angles', 'dihedrals', 'impropers']

        for name in lammps_props:
            if self.has(name):
                a = self.arrays[name][:n]
                self.arrays[name] = np.empty(M * n, dtype='object')
                for i in range(M):
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

    def extend(self, other, extend_type=False):
        """Extend atoms object by appending atoms from *other*."""
        if isinstance(other, Atom):
            other = self.__class__([other])

        if type(other) != self.__class__:
            other = self.__class__(other)
        other = other.copy()

        n1 = len(self)
        n2 = len(other)
        molid1_max = None
        if self.has('mol-id'):
            molid1_max = np.max(self.arrays['mol-id'])
        if self.has('type'):
            type1_max = np.max(self.arrays['type'])

        Atoms.extend(self, other)

        indx_of = {}
        for i in range(n2):
            indx_of[i] = i + n1
        self._set_indices_to(indx_of, [n1, None])
        if molid1_max and other.has('mol-id'):
            self.arrays['mol-id'][n1:] += molid1_max
        if type1_max and other.has('type') and extend_type:
            self.arrays['type'][n1:] += type1_max

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
