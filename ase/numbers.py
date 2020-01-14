import numpy as np
from ase.utils.arraywrapper import arraylike


__all__ = ['Numbers']

@arraylike
class Numbers:
    """
    A class that handles numbers in atoms object

    This class behaves as a numpy array of numbers, however
    on setting of the property the atom labels are updated
    """

    def __init__(self, atoms):
        """
        Create Numbers class
        Parameters
            atoms: Atoms object
        """
        self._atoms = atoms
        self.array = self._atoms.arrays['numbers']

    def __repr__(self):
        return str(self.array)

    def __array__(self, dtype=int):
        if dtype != int:
            raise ValueError('Cannot convert atomic numbers to array of type {}'
                             .format(dtype))
        return self.array

    def set(self, numbers, key=None):
        """
        Sets numbers according to key, and changes labels accordingly
        """
        array = self.array.copy()
        if key is None:
            key = np.arange(len(self._atoms), dtype=int)
        array[key] = np.array(numbers, dtype=int)
        self._atoms.set_array('numbers', array, int, ())

        # if labels in atoms, then rollback to default
        if 'labels' in self._atoms.arrays:
            # object ndarray ensures str is correctly handled
            labels = np.array(self._atoms.arrays['labels'],
                              dtype=object)
            symbols = np.array(self._atoms.get_chemical_symbols(),
                               dtype=object)
            labels[key] = symbols[key]
            # no set_array, since labels have to be objects
            # set_array doesn't change dtype of already existing array
            self._atoms.arrays['labels'] = labels

    def __setitem__(self, key, numbers):
        self.set(numbers, key)
