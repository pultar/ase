from collections.abc import MutableMapping

import numpy as np
from ase.utils import basestring

adata = [('numbers', np.int64),
         ('positions', np.float64, (3,))]


class AtomsData(MutableMapping):
    def __init__(self, data):
        self.data = data

    @classmethod
    def new(cls, natoms, dtype=adata):
        return cls(np.rec.array(np.zeros(natoms, dtype=dtype, order='C')))

    def copy(self):
        return self.__class__(self.data.copy())

    def todict(self):
        return {key: val.copy() for key, val in self.items()}

    def __contains__(self, key):
        return key in self.data.dtype.names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        if isinstance(key, basestring):
            if key not in self:
                raise KeyError(key)
            return self.data[key]

        # Otherwise, assume it's a slice
        return self.__class__(self.data[key].copy())

    def __setitem__(self, key, val):
        arr = np.asarray(val)
        # If we already have this key, make sure the provided value has
        # the same shape (unless the key is 'magmoms', which can be either
        # Nx1 or Nx3).
        if key in self and arr.shape[1:] != self.data[key].shape[1:]:
            if key == 'magmoms':
                del self[key]
            else:
                raise ValueError("Array {} has wrong shape! "
                                 "Expected: {}, got: {}."
                                 .format(key, self.data[key].shape,
                                         arr.shape))

        # Reallocate the data array if we don't have this key already
        if key not in self:
            descr = [key, arr.dtype]
            if len(arr.shape) > 1:
                descr.append(arr.shape[1:])
            self.new_array(key, descr)

        # If we're changing the numbers, invalidate masses and labels
        # if they are present
        if key == 'numbers':
            reset = []
            if 'masses' in self:
                reset.append(['masses', 0.])
            if 'symbols' in self:
                reset.apend(['labels', ''])
            for i, number in enumerate(arr):
                if self.data['numbers'][i] != number:
                    for badkey, defval in reset:
                        self.data[badkey][i] = defval

        self.data[key] = arr

    def new_array(self, key, descr):
        assert key not in self

        new_dtype = self.data.dtype.descr.copy()
        new_dtype.append(tuple(descr))

        newdata = np.zeros(len(self), dtype=new_dtype, order='C')
        for oldkey, data in self.items():
            newdata[oldkey] = data

        self.data = newdata

    def __delitem__(self, name):
        if name not in self:
            raise ValueError("No key '{}' to delete!".format(name))

        new_dtype = [d for d in self.data.dtype.descr if d[0] != name]
        newdata = np.zeros(len(self), dtype=new_dtype, order='C')

        for key, data in self.items():
            if key == name:
                continue
            newdata[key] = data

        self.data = newdata

    def __iter__(self):
        for key in self.data.dtype.names:
            yield key

    def __mul__(self, val):
        newdata = np.tile(self.data, val)
        return self.__class__(newdata)

    def __add__(self, other):
        for key in self:
            if key in other:
                descra = self.data.dtype.fields[key]
                descrb = other.data.dtype.fields[key]
                if descra != descrb:
                    raise ValueError("Same key '{}' but different descr!"
                                     "self: {}, other:{}."
                                     .format(key, descra, descrb))

        descr = self.data.dtype.descr.copy()
        for row in other.data.dtype.descr:
            if row not in descr:
                descr.append(row)

        new = self.__class__.new(len(self) + len(other), dtype=descr)

        for key in self:
            new.data[key][:len(self)] = self.data[key]
        for key in other:
            new.data[key][len(self):] = other.data[key]

        return new
