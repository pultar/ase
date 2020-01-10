import numpy as np
from ase.utils import basestring

from ase.data import (chemical_symbols, atomic_numbers, atomic_masses,
                      atomic_masses_common, atomic_masses_iupac2016,
                      atomic_masses_legacy)


_masses = {'defaults': atomic_masses,
           'averaged': atomic_masses,
           'common': atomic_masses_common,
           '2016': atomic_masses_iupac2016,
           'legacy': atomic_masses_legacy}

adata = [('numbers', np.int32),
         ('tags', np.int32),
         ('masses', np.float64),
         ('initial_charges', np.float64),
         ('positions', np.float64, (3,)),
         ('momenta', np.float64, (3,)),
         ('initial_magmoms', np.float64, (3,)),
         ('symbols', 'U3'),
         ('labels', 'U5')]

adtype = np.dtype(adata)

def symbol_from_number(number):
    if number < 0 or number >= len(chemical_symbols):
        return 'X'
    return chemical_symbols[number]

class AtomsData:
    """The state of the atoms stored in an Atoms object.

    This object stores the state of each atom in an Atoms object as a
    single entry in an array with a custom dtype (a structured array).

    The entries of the array are as follows:

    numbers: signed 64-bit integer
        The atomic numbers of the elements, nominally between 1 and 118,
        inclusive, but we used a signed 64-bit int to add flexibility.

        Modifying 'numbers' resets the following properties:
            - masses
            - labels

    tags: signed 64-bit integer
        Optional integer tags. Defaults to 0.

    masses: 64-bit float
        Atomic masses. Defaults to the Earth-abundance-weighted
        isotope mass average.

    initial_charges: 64-bit float
        The initial atomic charges. Defaults to 0.

    positions: 3 x 64-bit float
        The absolute position of the atoms, in Angstrom.
        Defaults to (0, 0, 0).

    momenta: 3 x 64-bit float
        The momenta vectors of the atoms, in (FIXME: What units?).
        Defaults to (0, 0, 0).

    initial_magmoms: 3 x 64-bit float
        The magnetic moment vectors of the atoms.
        Scalar magnetic moments are stored along in the z-axis
        component. Defaults to (0, 0, 0).

    symbols: Up to 3 unicode characters
        The elemental symbols of the atoms. This value is determined from
        numbers, and cannot be set manually.

    labels: Up to 5 unicode characters
        Optional labels, allowing the possibility to distinguish
        between different kinds of the same element. Default to ''.
        If the label is '', then the symbol will be returned instead
        of the label.
"""
    def __init__(self, data, *,
                 default_masses='defaults',
                 noncollinear=False,
                 initialized=None):
        self.data = data
        self.default_masses = default_masses
        self.noncollinear = noncollinear
        if initialized is None:
            initialized = dict()
            for item in adata:
                prop = item[0]
                if prop in ['symbols']:
                    continue
                initialized[prop] = False
            initialized['numbers'] = True
        self.initialized = initialized

    def copy(self):
        new = self.__class__(self.data.copy(),
                             default_masses=self.default_masses,
                             noncollinear=self.noncollinear,
                             initialized=self.initialized)
        return new

    def todict(self):
        result = dict(numbers=self.data.numbers,
                      default_masses=self.default_masses,
                      noncollinear=self.noncollinear)
        for prop, init in self.initialized.items():
            if init:
                result[prop] = getattr(self.data, prop)
        return result

    def __len__(self):
        return len(self.data)

    def __getitem__(self, val):
        return self.__class__(np.atleast_1d(self.data[val]),
                              default_masses=self.default_masses,
                              noncollinear=self.noncollinear)

    @classmethod
    def new(cls, natoms, **kwargs):
        return cls(np.rec.array(np.zeros(natoms, dtype=adtype)), **kwargs)

    @classmethod
    def from_numbers(cls, numbers, **kwargs):
        new = cls.new(len(numbers), **kwargs)
        new.set_numbers(numbers)
        return new

    def _validate_size(self, prop, vals, vector):
        if vals is None:
            return 0.

        # If the user provides a scalar value (rather than a numpy array),
        # then that probably means they want to set the entire array to a
        # particular value, so bypass any size checking.
        if np.isscalar(vals):
            return vals

        if vector:
            x = np.atleast_2d(vals)
        else:
            x = np.atleast_1d(vals)

        curr = getattr(self, prop)
        if x.shape != curr.shape:
            raise ValueError("Property {} has wrong size! "
                             "(expected: {}, got: {})"
                             .format(prop, x.shape, curr.shape))
        return x

    # Atomic numbers
    def set_numbers(self, vals):
        numbers = self._validate_size('numbers', vals, False)
        for i, number in enumerate(numbers):
            # Only invalidate if the number is changing
            if self.data.numbers[i] == number:
                continue
            self.data.masses[i] = 0.
            self.data.labels[i] = ''
            self.data.numbers[i] = number
            self.data.symbols[i] = symbol_from_number(number)

    numbers = property(lambda self: self.data.numbers)

    def _set_prop(self, prop, val, vector=False):
        if val is None:
            self.initialized[prop] = False
            setattr(self.data, prop, 0.)
        else:
            x = self._validate_size(prop, val, vector=vector)
            setattr(self.data, prop, x)
            self.initialized[prop] = True

    # Tags
    def set_tags(self, val):
        self._set_prop('tags', val)

    tags = property(lambda self: self.data.tags)

    # Masses
    def get_masses(self):
        result = self.data.masses.copy()
        for i, mass in enumerate(result):
            if mass == 0.:
                result[i] = _masses[self.default_masses][self.data.numbers[i]]
        return result

    def set_masses(self, val):
        if isinstance(val, basestring):
            self.default_masses = val
        else:
            self._set_prop('masses', val)

    masses = property(get_masses)

    def set_initial_charges(self, val):
        self._set_prop('initial_charges', val)

    initial_charges = property(lambda self: self.data.initial_charges)

    # Positions
    def set_positions(self, val):
        self._set_prop('positions', val, vector=True)

    positions = property(lambda self: self.data.positions)

    # Momenta
    def set_momenta(self, val):
        self._set_prop('momenta', val, vector=True)

    momenta = property(lambda self: self.data.momenta)

    # Magnetic moments
    def get_initial_magmoms(self):
        if self.noncollinear:
            return self.data.initial_magmoms
        return self.data.initial_magmoms[:, 2]

    def set_initial_magmoms(self, val):
        if val is None:
            self.initialized['initial_magmoms'] = False
            self.data.magmom = 0.
            return

        self.initialized['initial_magmoms'] = True
        magmoms = np.asarray(val).reshape((len(self), -1))
        if magmoms.shape[1] == 3:
            self.noncollinear = True
            self.data.initial_magmoms = magmoms
        elif magmoms.shape[1] == 1:
            self.noncollinear = False
            self.data.initial_magmoms[:, :2] = 0.
            self.data.initial_magmoms[:, 2] = magmoms.ravel()
        else:
            raise ValueError('initial_magmoms has a weird shape!')

    initial_magmoms = property(get_initial_magmoms)

    # Atomic symbols (no setter, use Numbers instead)
    symbols = property(lambda self: list(self.data.symbols))
    symbol = symbols

    # Labels
    def get_labels(self):
        labels = []
        for i, label in enumerate(self.data.labels):
            if not label:
                labels.append(self.data.symbols[i])
            else:
                labels.append(label)
        return labels

    def set_labels(self, val):
        self._set_prop('labels', val)

    labels = property(get_labels)

    def __mul__(self, val):
        newdata = np.tile(self.data, val)
        return self.__class__(newdata,
                              default_masses=self.default_masses,
                              noncollinear=self.noncollinear,
                              initialized=self.initialized)

    def __add__(self, other):
        noncollinear = self.noncollinear or other.noncollinear
        initialized = self.initialized.copy()
        for name, init in other.initialized.items():
            if init:
                initialized[name] = True
        new = self.__class__.new(len(self) + len(other),
                                 default_masses=self.default_masses,
                                 noncollinear=noncollinear,
                                 initialized=initialized)
        new.data[:len(self)] = self.data
        new.data[len(self):] = other.data
        return new
