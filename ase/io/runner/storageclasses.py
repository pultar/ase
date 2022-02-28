"""Implementation of classes for storing RuNNer parameters and results.

This module provides custom classes for storing the different types of data
produced and/or read by RuNNer.

Provides
--------
ElementStorageMixin : object
    A mixin for all storage containers which store data in element-wise fashion.
RunnerScaling : ElementStorageMixin
    Storage container for RuNNer symmetry function scaling data.
RunnerWeights : ElementStorageMixin
    Storage container for weights and bias values of atomic neural networks.
RunnerStructureSymmetryFunctionValues: ElementStorageMixin
    Storage container for all symmetry function values of one single structure.
RunnerSymmetryFunctionValues : object
    Storage container for many `RunnerStructureSymmetryFunctionValues` objects,
    i.e. for a collection of symmetry function values in one training dataset.
RunnerFitResults : object
    Storage container for fit quality indicators in RuNNer Mode 2 stdout.
RunnerSplitTrainTest : object
    Storage container for the split between train and test set in RuNNer Mode 1.
RunnerResults : TypedDict
    Type specifications for all results that the RuNNer calculator can produce.
    Essentially a collection of all the previously mentioned storage container
    classes plus energies and forces.
SymmetryFunction : object
    Storage container for a single symmetry function.
SymmetryFunctionSet : object
    Storage container for a collection of symmetry functions.

Reference
---------
- [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
- Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Optional, Union, Iterator, Tuple, List, Dict

import os
import io
import re

import numpy as np
from ase.data import atomic_numbers, chemical_symbols

# Custom type specification for lists of symmetry function parameters. Can be
# two kinds of tuples, depending on whether it is a radial or an angular
# symmetry function.
SFListType = Union[Tuple[str, int, str, float, float, float],
                   Tuple[str, int, str, str, float, float, float, float]]


class ElementStorageMixin:
    """Abstract mixin for storing element-specific RuNNer parameters/results.

    When constructing a neural network potential with RuNNer, one atomic neural
    network is built for every element in the system. As a result, many RuNNer
    parameters are element-specific.
    This mixin transforms any class into a storage container for
    element-specific data by
        - defining an abstract `data` container, i.e. a dictionary with
          str-np.ndarray pairs holding one numpy ndarray for each element.
          The keys are chemical symbols of elements.
        - defining magic methods like __iter__ and __setitem__ for convenient
          access back to that data storage.

    The storage of data in elementwise format is more efficient, as data can
    often be compressed into non-ragged numpy arrays.
    """

    def __init__(self) -> None:
        """Initialize the object."""
        # Data container for element - data pairs.
        self.data: Dict[str, np.ndarray] = {}

    def __iter__(self) -> Iterator[Tuple[str, np.ndarray]]:
        """Iterate over all key-value pairs in the `self.data` container."""
        for key, value in self.data.items():
            yield key, value

    def __len__(self) -> int:
        """Show the combined length of all stored element data arrays."""
        length = 0
        for value in self.data.values():
            shape: Tuple[int, ...] = value.shape
            length += shape[0]

        return length

    def __setitem__(
        self,
        key: Union[str, int],
        value: np.ndarray
    ) -> None:
        """Set one key-value pair in the self.data dictionary.

        Each key in self.data is supposed to be the chemical symbol of an
        element. Therefore, when an integer key is provided it is translated
        into the corresponding chemical symbol.

        Parameters
        ----------
        key : str or int
            The dictionary key were `value` will be stored. Integer keys are
            translated into the corresponding chemical symbol.
        value : np.ndarray
            The element-specific data to be stored in the form of a numpy array.
        """
        if isinstance(key, int):
            key = chemical_symbols[key]

        self.data[key] = value

    def __getitem__(self, key: Union[str, int]) -> np.ndarray:
        """Get the data associated with `key` in the self.data dictionary.

        The data can either be accessed by the atomic number or the chemical
        symbol.

        Parameters
        ----------
        key : str or int
            Atomic number of chemical symbol of the desired `self.data`.
        """
        if isinstance(key, int):
            key = chemical_symbols[key]

        return self.data[key]

    @property
    def elements(self) -> List[str]:
        """Show a list of elements for which data is available in storage."""
        return list(self.data.keys())


class RunnerScaling(ElementStorageMixin):
    """Storage container for RuNNer symmetry function scaling data.

    Resources
    ---------
    For more information on the `scaling.data` file format in RuNNer please
    visit the
    [documentation](https://theochemgoettingen.gitlab.io/RuNNer/1.3/reference/files/#scalingdata).
    """

    def __init__(self, infile: Optional[io.TextIOWrapper] = None) -> None:
        """Initialize the object.

        Parameters
        ----------
        infile : TextIOWrapper, optional, _default_ `None`
            If given, data will be read from this fileobj upon initialization.
        """
        # Initialize the base class. This creates the main self.data storage.
        super().__init__()

        # Store additional non-element-specific properties.
        self.target_min: float = np.NaN
        self.target_max: float = np.NaN

        # Read data from fileobj, if given.
        if isinstance(infile, io.TextIOWrapper):
            self.read(infile)

    def __repr__(self) -> str:
        """Show a string representation of the object."""
        return f'{self.__class__.__name__}(elements={self.elements}, ' \
               + f'min={self.target_min}, max={self.target_max})'

    def read(self, infile: io.TextIOWrapper) -> None:
        """Read symmetry function scaling data.

        Parameters
        ----------
        infile : TextIOWrapper
            The fileobj containing the scaling data in RuNNer format.
        """
        scaling: Dict[str, List[List[float]]] = {}
        for line in infile:
            data = line.split()

            # Lines of length five hold scaling data for each symmetry function.
            if len(data) == 5:
                element_id = data[0]

                if element_id not in scaling:
                    scaling[element_id] = []

                scaling[element_id].append([float(i) for i in data[1:]])

            # The final line holds only the min. and max. of the target property.
            elif len(data) == 2:
                self.target_min = float(data[0])
                self.target_max = float(data[1])

        # Transform data into numpy arrays.
        for element_id, scalingdata in scaling.items():
            npdata: np.ndarray = np.array(scalingdata)
            self.data[element_id] = npdata

    def write(self, outfile: io.TextIOWrapper) -> None:
        """Write symmetry function scaling data.

        Parameters
        ----------
        outfile : TextIOWrapper
            The fileobj to which the scaling data will be written.
        """
        for element_id, data in self.data.items():
            # First, write the scaling data for each symmetry function.
            for line in data:
                outfile.write(f'{element_id:5s} {int(line[0]):5d}'
                              + f'{line[1]:18.9f} {line[2]:18.9f} '
                              + f'{line[3]:18.9f}\n')

        # The last line contains the minimum and maximum of the target property.
        outfile.write(f'{self.target_min:18.9f} {self.target_max:18.9f}\n')


class RunnerWeights(ElementStorageMixin):
    """Storage container for RuNNer neural network weights and bias values.

    Resources
    ---------
    For more information on the `weights.XXX.data` file format in RuNNer please
    visit the
    [docs](https://theochemgoettingen.gitlab.io/RuNNer/1.3/reference/files/#weightsxxxdata).
    """

    # Weights can be read either from a single file (`infile` argument) or from
    # a set of files under the given `path`. In the latter case, `elements`,
    # `prefix`, and `suffix` need to be exposed to the user.
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        infile: Optional[io.TextIOWrapper] = None,
        path: Optional[str] = None,
        elements: Optional[List[str]] = None,
        prefix: str = 'weights',
        suffix: str = 'data'
    ) -> None:
        """Initialize the object.

        Upon initialization, the user may specify to read weights either from
        a fileobj (`infile`) or from weight files residing under `path`.

        Parameters
        ----------
        infile : TextIOWrapper, optional, _default_ `None`
            If given, data will be read from this fileobj upon initialization.
        path : str, optional, _default_ `None`
            If given, data will be read from all weight files under the given
            directory.
        elements : List[str], optional, _default_ `None`
            A selection of chemical symbols for which the weights under `path`
            will be read. Only active when `path` is given.
        prefix : str, optional, _default_ `weights`
            The filename prefix of weight files under `path`. Only necessary
            when path is specified.
        suffix : str, optional, _default_ `data`
            The filename suffix of weight files under `path`. Only necessary
            when path is specified.
        """
        super().__init__()

        if infile is not None:
            self.read(infile)

        if path is not None:
            self.readall(path, elements, prefix, suffix)

    def __repr__(self) -> str:
        """Show a string representation of the object."""
        # Get the number of weights for each element in storage.
        num_weights = []
        for element, data in self.data.items():
            num_weights.append(f'{element}: {data.shape[0]}')
        return f"{self.__class__.__name__}({', '.join(num_weights)})"

    def read(self, infile: io.TextIOWrapper) -> None:
        """Read the atomic neural network weights and biases for one element.

        Parameters
        ----------
        infile : TextIOWrapper
            Data will be read from this fileobj.
        """
        # Obtain the chemical symbol of the element. RuNNer names weights files
        # like `prefix`.{atomic_number}.`suffix`.
        atomic_number = int(infile.name.split('.')[1])
        element = chemical_symbols[atomic_number]

        # Store the weights as a np.ndarray.
        self.data[element] = np.genfromtxt(infile, usecols=(0))

    def readall(
        self,
        path: str = '.',
        elements: Optional[List[str]] = None,
        prefix: str = 'weights',
        suffix: str = 'data'
    ) -> None:
        """Read atomic neural network weights and bias values of many elements.

        Read all atomic neural network weight and bias files found under the
        specified path. The selection may be constrained by additional keywords.

        Parameters
        ----------
        path : str, _default_ '.'
            Data will be read from all weight files found under the given
            path.
        elements : List[str], optional, _default_ `None`
            A selection of chemical symbols for which the weights under `path`
            will be read.
        prefix : str, optional, _default_ `weights`
            The filename prefix of weight files under `path`.
        suffix : str, optional, _default_ `data`
            The filename suffix of weight files under `path`.
        """
        # If no elements were supplied, find all the element weights files at
        # the given path.
        if elements is None:
            elements = []
            for file in os.listdir(path):
                if file.startswith(prefix):
                    # Transform the atomic numbers into the chemical symbol.
                    element = chemical_symbols[int(file.split('.')[1])]
                    elements.append(element)

        # Read in all desired element neural network weights.
        for element in elements:

            # Obtain the atomic number of the element and the path to the file.
            number = atomic_numbers[element]
            fullpath = os.path.join(path, f'{prefix}.{number:03d}.{suffix}')

            # Store the weights as a np.ndarray.
            self.data[element] = np.genfromtxt(fullpath, usecols=(0))

    def write(
        self,
        path: str = '.',
        elements: Optional[List[str]] = None,
        prefix: str = 'weights',
        suffix: str = 'data'
    ) -> None:
        """Write atomic neural network weights and biases for one element.

        Parameters
        ----------
        path : str, _default_ '.'
            Data will be read from all weight files found under the given
            path.
        elements : List[str], optional, _default_ `None`
            A selection of chemical symbols for which the weights under `path`
            will be read.
        prefix : str, optional, _default_ `weights`
            The filename prefix of weight files under `path`.
        suffix : str, optional, _default_ `data`
            The filename suffix of weight files under `path`.
        """
        for element, weights in self.data.items():
            # Skip over unspecified elements, if given.
            if elements is not None and element not in elements:
                continue

            # Write the data to file.
            number = atomic_numbers[element]
            element_path = os.path.join(path, f'{prefix}.{number:03d}.{suffix}')
            np.savetxt(element_path, weights, fmt='%.10f')


class RunnerStructureSymmetryFunctionValues(ElementStorageMixin):
    """Storage container for the symmetry function values of one structure."""

    def __init__(
        self,
        energy_total: float = np.NaN,
        energy_short: float = np.NaN,
        energy_elec: float = np.NaN,
        charge: float = np.NaN,
    ) -> None:
        """Initialize the object.

        Parameters
        ----------
        infile : TextIOWrapper, optional, _default_ `None`
            If given, data will be read from this fileobj upon initialization.
        """
        # Initialize the base class. This will create the main data storage.
        super().__init__()

        # Save additional non-element-specific parameters.
        # Each parameter is stored in one large array for all structures.
        self.energy_total = energy_total
        self.energy_short = energy_short
        self.energy_elec = energy_elec
        self.charge = charge

    def __repr__(self) -> str:
        """Show a string representation of the object."""
        return f'{self.__class__.__name__}(n_atoms={len(self)})'

    def by_atoms(self) -> List[Tuple[str, np.ndarray]]:
        """Expand dictionary of element symmetry functions into atom tuples."""
        data_tuples = []
        index = []

        for element, element_sfvalues in self.data.items():
            index += list(element_sfvalues[:, 0])
            sfvalues_list: List[np.ndarray] = list(element_sfvalues[:, 1:])

            for atom_sfvalues in sfvalues_list:
                data_tuples.append((element, atom_sfvalues))

        return [x for _, x in sorted(zip(index, data_tuples))]


class RunnerSymmetryFunctionValues:
    """Storage container for RuNNer symmetry function values.

    In RuNNer Mode 1, many-body symmetry functions (SFs) are calculated aiming
    to describe the chemical environment of every atom. As a result, every atom
    is characterized by a vector of SF values. These SF vectors always have the
    same size for each kind of element in the system.

    In the RuNNer Fortran code, this information is written to two files,
    'function.data' (for train set structures) and 'testing.data' (for test set
    structures). The files also contain additional information for each
    structure (all atomic units): the total energy, the short-range energy,
    the electrostatic energy, and the charge.
    """

    def __init__(self, infile: Optional[io.TextIOWrapper] = None) -> None:
        """Initialize the object.

        Parameters
        ----------
        infile : TextIOWrapper, optional, _default_ `None`
            If given, data will be read from this fileobj upon initialization.
        """
        # Initialize the base class. This will create the main data storage.
        super().__init__()

        self.data: List[RunnerStructureSymmetryFunctionValues] = []

        # If given, read data from `infile`.
        if isinstance(infile, io.TextIOWrapper):
            self.read(infile)

    def __len__(self) -> int:
        """Show the number of structures in storage."""
        return len(self.data)

    def __getitem__(self, index: int) -> RunnerStructureSymmetryFunctionValues:
        """Get the data for one structure at `index` in storage."""
        return self.data[index]

    def __repr__(self) -> str:
        """Show a string representation of the object."""
        return f'{self.__class__.__name__}(n_structures={len(self)})'

    def __add__(
        self,
        blob: 'RunnerSymmetryFunctionValues'
    ) -> 'RunnerSymmetryFunctionValues':
        """Add a blob of symmetry function values to storage."""
        self.append(blob)
        return self

    def sort(self, index: List[int]) -> None:
        """Sort the structures in storage by `index`."""
        self.data = [x for _, x in sorted(zip(index, self.data))]

    def append(self, blob: 'RunnerSymmetryFunctionValues') -> None:
        """Append another blob of symmetry function values to storage."""
        for structure in blob.data:
            self.data.append(structure)

    def read(self, infile: io.TextIOWrapper) -> None:
        """Read symmetry function values from `infile`."""
        allsfvalues: Dict[str, List[List[float]]] = {}
        idx_atom = 0
        for line in infile:
            spline = line.split()

            # Line of length 1 marks a new structure and holds the # of atoms.
            if len(spline) == 1:
                idx_atom = 0
                allsfvalues = {}
                structure = RunnerStructureSymmetryFunctionValues()

            # Line of length 4 marks the end of a structure.
            elif len(spline) == 4:
                structure.charge = float(spline[0])
                structure.energy_total = float(spline[1])
                structure.energy_short = float(spline[2])
                structure.energy_elec = float(spline[3])

                for element, data in allsfvalues.items():
                    structure.data[element] = np.array(data)

                self.data.append(structure)

            # All other lines hold symmetry function values.
            else:
                # Store the symmetry function values in the element dictionary.
                element = chemical_symbols[int(spline[0])]
                sfvalues = [float(i) for i in spline[1:]]

                if element not in allsfvalues:
                    allsfvalues[element] = []

                allsfvalues[element].append([float(idx_atom)] + sfvalues)
                idx_atom += 1

    def write(
        self,
        outfile: io.TextIOWrapper,
        index: Union[int, slice, List[int]] = slice(0, None),
        fmt: str = '16.10f'
    ) -> None:
        """Write symmetry function scaling data."""
        # Retrieve the data.
        images: List[RunnerStructureSymmetryFunctionValues] = self.data

        # Filter the images which should be printed according to `index`.
        if isinstance(index, slice):
            images = images[index]
        elif isinstance(index, int):
            images = [images[index]]
        else:
            images = [images[i] for i in index]

        for image in images:
            # Start a structure by writing the number of atoms.
            outfile.write(f'{len(image):6}\n')

            # Write one line for each atom containing the atomic number followed
            # by the symmetry function values.
            for element, sfvalues in image.by_atoms():
                number = atomic_numbers[element]

                outfile.write(f'{number:3}')
                outfile.write(''.join(f'{i:{fmt}}' for i in sfvalues))
                outfile.write('\n')

            # End a structure by writing charge and energy information.
            outfile.write(f'{image.charge:{fmt}} {image.energy_total:{fmt}} '
                          + f'{image.energy_short:{fmt}} '
                          + f'{image.energy_elec:{fmt}}\n')


class RunnerFitResults:
    """Storage container for RuNNer training results.

    RuNNer Mode 2 generates a neural network potential in an iterative training
    process typical when working with neural networks. The information generated
    in course of this procedure enables the evaluation of the potential quality.
    This class stores typical quality markers to facilitate training process
    analysis:
    epochs : int
        The number of epochs in the training process.
    rmse_energy : Dict[str, float]
        Root mean square error of the total energy. Possible keys are 'train',
        for the RMSE in the train set, and 'test', for the RMSE in the test set.
    rmse_force : Dict[str, float]
        Root mean square error of the atomic forces. See `rmse_energy`.
    rmse_charge : Dict[str, float]
        Root mean square error of the atomic charges. See `rmse_energy`.
    opt_rmse_epoch : int, optional, _default_ `None`
        The number of the epoch were the best fit was obtained.
    units : Dict[str, str]
        The units of the energy and force RMSE.
    """

    def __init__(self, infile: Optional[io.TextIOWrapper]) -> None:
        """Initialize the object.

        Parameters
        ----------
        infile : TextIOWrapper, optional, _default_ `None`
            If given, data will be read from this fileobj upon initialization.
        """
        # Helper type hint definition for code brevity.
        RMSEDict = Dict[str, List[Optional[float]]]

        self.epochs: List[Optional[int]] = []
        self.rmse_energy: RMSEDict = {'train': [], 'test': []}
        self.rmse_forces: RMSEDict = {'train': [], 'test': []}
        self.rmse_charge: RMSEDict = {'train': [], 'test': []}
        self.opt_rmse_epoch: Optional[int] = None
        self.units: Dict[str, str] = {'rmse_energy': '', 'rmse_force': ''}

        # If given, read data from `infile`.
        if isinstance(infile, io.TextIOWrapper):
            self.read(infile)

    def __repr__(self) -> str:
        """Show a string representation of the object."""
        num_epochs = len(self.epochs)
        return f'{self.__class__.__name__}(num_epochs={num_epochs}, ' \
             + f'best epoch={self.opt_rmse_epoch})'

    def read(self, infile: io.TextIOWrapper) -> None:
        """Read RuNNer Mode 2 results.

        Parameters
        ----------
        infile : TextIOWrapper
            Data will be read from this fileobj.
        """
        for line in infile:
            data = line.split()

            # Read the RMSEs of energies, forces and charges, and the
            # corresponding epochs.
            if line.strip().startswith('ENERGY'):
                if '*****' in line:
                    epoch, rmse_train, rmse_test = None, None, None
                else:
                    epoch = int(data[1])
                    rmse_train, rmse_test = float(data[2]), float(data[3])

                self.epochs.append(epoch)
                self.rmse_energy['train'].append(rmse_train)
                self.rmse_energy['test'].append(rmse_test)

            elif line.strip().startswith('FORCES'):
                if '*****' in line:
                    rmse_train, rmse_test = None, None
                else:
                    rmse_train, rmse_test = float(data[2]), float(data[3])

                self.rmse_forces['train'].append(rmse_train)
                self.rmse_forces['test'].append(rmse_test)

            elif line.strip().startswith('CHARGE'):
                rmse_train, rmse_test = float(data[2]), float(data[3])
                self.rmse_charge['train'].append(rmse_train)
                self.rmse_charge['test'].append(rmse_test)

            # Read the fitting units, indicated by the heading 'RMSEs'.
            if 'RMSEs' in line:
                # Use regular expressions to find the units. All units
                # conveniently start with two letters ('Ha', or 'eV'), followed
                # by a slash and some more letters (e.g. 'Bohr', or 'atom').
                units: List[str] = re.findall(r'\w{2}/\w+', line)
                self.units['rmse_energy'] = units[0]
                self.units['rmse_force'] = units[0]

            # Read in the epoch where the best fit was obtained.
            if 'Best short range fit has been obtained in epoch' in line:
                self.opt_rmse_epoch = int(data[-1])

            # Explicitely handle the special case that the fit did not yield any
            # improvement. This also means that no weights were written.
            if 'No improvement' in line:
                self.opt_rmse_epoch = None


class RunnerSplitTrainTest:
    """Storage container for the split between train and test set in RuNNer.

    In RuNNer Mode 1, the dataset presented to the program is separated into
    a training portion, presented to the neural networks for iteratively
    improving the weights, and a testing portion which is only used for
    evaluation.
    This class stores this data and enables to read it from Mode 1 output files.
    """

    def __init__(self, infile: Optional[io.TextIOWrapper]) -> None:
        """Initialize the object.

        Parameters
        ----------
        infile : TextIOWrapper, optional, _default_ `None`
            If provided, data will be read from this fileobj.
        """
        self.train: List[int] = []
        self.test: List[int] = []

        # If given, read data from `infile`.
        if isinstance(infile, io.TextIOWrapper):
            self.read(infile)

    def __repr__(self) -> str:
        """Show a string representation of the object."""
        return f'{self.__class__.__name__}(n_train={len(self.train)}, ' \
             + f'n_test={len(self.test)})'

    def read(self, infile: io.TextIOWrapper) -> None:
        """Read RuNNer splitting data.

        Parameters
        ----------
        infile : TextIOWrapper
            Data will be read from this fileobj. The file should contain the
            stdout from a RuNNer Mode 1 run.
        """
        for line in infile:
            if 'Point is used for' in line:
                # In Python, indices start at 0, therefore we subtract 1.
                point_idx = int(line.split()[0]) - 1
                split_type = line.split()[5]

                if split_type == 'training':
                    self.train.append(point_idx)
                elif split_type == 'testing':
                    self.test.append(point_idx)


# Originally inherited from TypedDict, but this was removed for now to retain
# backwards compatibility with Python 3.6 and 3.7.
# class RunnerResults(TypedDict, total=False)
#     """Type hints for RuNNer results dictionaries."""
# 
#     fitresults: RunnerFitResults
#     sfvalues: RunnerSymmetryFunctionValues
#     weights: RunnerWeights
#     scaling: RunnerScaling
#     splittraintest: RunnerSplitTrainTest
#     energy: Union[float, NDArray]
#     forces: NDArray


class SymmetryFunction:
    """Generic class for one single symmetry function."""

    # Symmetry functions have a few arguments which need to be given upon
    # class initialization.
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        sftype: Optional[int] = None,
        cutoff: Optional[float] = None,
        elements: Optional[List[str]] = None,
        coefficients: Optional[List[float]] = None,
        sflist: Optional[SFListType] = None
    ) -> None:
        """Initialize the class.

        Parameters
        ----------
        sftype : int
            The type of symmetry function.
        cutoff : float
            The symmetry function cutoff radius in units of Bohr.

        Optional Arguments
        ------------------
        coefficients : List[float]
            The coefficients of this symmetry function. The number of necessary
            coefficients depends on the `sftype` as documented in the RuNNer
            manual.
        elements : List[str]
            Symbols of the elements described by this symmetry function.
            The first entry of the list gives the central atom, while all
            following entries stand for neighbor atoms. Usually, the number of
            neighbors should be 1 (= radial symfun) or 2 (= angular symfun)
        """
        self.sftype = sftype
        self.cutoff = cutoff
        self.elements = elements
        self.coefficients = coefficients

        if sflist is not None:
            self.from_list(sflist)

    def __repr__(self) -> str:
        """Show a clean summary of the class object and its contents."""
        return f'{self.__class__.__name__}(type={self.sftype}, ' \
             + f'cutoff={self.cutoff}, ' \
             + f'elements={self.elements}, coefficients={self.coefficients})'

    def __add__(
        self,
        blob: Union['SymmetryFunctionSet', 'SymmetryFunction']
    ) -> Union['SymmetryFunctionSet', 'SymmetryFunction']:
        """Define addition behaviour."""
        if isinstance(blob, SymmetryFunctionSet):
            blob += self
            return blob

        if isinstance(blob, SymmetryFunction):
            sfset = SymmetryFunctionSet()
            sfset += self
            sfset += blob
            return sfset

        raise TypeError('unsupported operand type(s) for multiplication.')

    def __mul__(self, multiplier: int) -> 'SymmetryFunctionSet':
        """Define multiplication behaviour."""
        sfset = SymmetryFunctionSet()
        for _ in range(multiplier):
            sfset += self.copy()

        return sfset

    def __rmul__(self, multiplier: int) -> 'SymmetryFunctionSet':
        """Multiply the object from the left."""
        return self.__mul__(multiplier)

    def copy(self) -> 'SymmetryFunction':
        """Make the object copyable."""
        return SymmetryFunction(self.sftype, self.cutoff, self.elements,
                                self.coefficients)

    @property
    def tag(self) -> str:
        """Show a human-readable equivalent of the `sftype` parameter."""
        if self.sftype in [2]:
            tag = 'radial'
        elif self.sftype in [3, 4, 5]:
            tag = 'angular'
        else:
            tag = 'unknown'

        return tag

    def to_runner(self, fmt: str = '16.8f') -> str:
        """Convert the symmetry function into RuNNer input.nn format."""
        if self.coefficients is None or self.elements is None:
            raise Exception('Symmetryfunction is not fully defined yet.')

        centralatom = self.elements[0]
        neighbors = self.elements[1:]
        coefficients = [f'{c:{fmt}}' for c in self.coefficients]

        string = f'{centralatom} {self.sftype} ' \
                + ' '.join(neighbors) \
                + ' '.join(coefficients) \
                + f' {self.cutoff:{fmt}}'

        return string

    def to_list(self) -> SFListType:
        """Create a list representation of the symmetry function."""
        if (self.elements is None or self.coefficients is None
            or self.sftype is None or self.cutoff is None):
            raise AttributeError('Symmetry function not fully defined.')

        if self.tag == 'radial':
            return (self.elements[0], self.sftype, self.elements[1],
                      self.coefficients[0], self.coefficients[1], self.cutoff)

        if self.tag == 'angular':
            return (self.elements[0], self.sftype, self.elements[1],
                      self.elements[2], self.coefficients[0],
                      self.coefficients[1], self.coefficients[2], self.cutoff)

        raise NotImplementedError('Cannot convert symmetry functions of '
                                  + f'type {self.tag} to list.')

    def from_list(self, sflist: SFListType) -> None:
        """Fill storage from a list of symmetry function parameters."""
        self.sftype = sflist[1]
        self.cutoff = sflist[-1]

        # The type: ignore statements are justified because the len() checks in
        # the if-statements make sure that the number of parameters is
        # compatible with the sftype.
        if self.tag == 'radial' and len(sflist) == 6:
            self.elements = [sflist[0], sflist[2]]
            self.coefficients = [float(sflist[3]), float(sflist[4])]  # type: ignore

        elif self.tag == 'angular' and len(sflist) == 8:
            self.elements = [sflist[0], sflist[2], sflist[3]]  # type: ignore
            self.coefficients = [float(sflist[4]), float(sflist[5]), float(sflist[6])]  # type: ignore
        else:
            raise ValueError('sftype incompatible with number of parameters.')


class SymmetryFunctionSet:
    """Class for storing groups/sets of symmetry functions."""

    def __init__(
        self,
        sflist: Optional[List[SFListType]] = None,
        min_distances: Optional[Dict[str, float]] = None
    ) -> None:
        """Initialize the class.

        This class can be nested to group symmetry functions together.

        Optional Parameters
        -------------------
        elements : List[str], _default_ None
        min_distances : List[float], _default_ None
        """
        self._sets: List[SymmetryFunctionSet] = []
        self._symmetryfunctions: List[SymmetryFunction] = []
        self.min_distances = min_distances

        if sflist is not None:
            self.from_list(sflist)

    def __str__(self) -> str:
        """Show a clean summary of the class object and its contents."""
        n_sets = len(self._sets)
        n_symmetryfunctions = len(self._symmetryfunctions)

        return f'{self.__class__.__name__}(type={self.sftypes}, ' \
             + f'sets={n_sets}, symmetryfunctions={n_symmetryfunctions})'

    def __repr__(self) -> str:
        """Show a unique summary of the class object."""
        return f'{self.to_list()}'

    def __len__(self) -> int:
        """Return the number of symmetry functions as the object length."""
        return len(self._symmetryfunctions)

    def __add__(
        self,
        blob: Union['SymmetryFunctionSet', SymmetryFunction]
    ) -> 'SymmetryFunctionSet':
        """Overload magic routine to enable nesting of multiple sets."""
        # Add a new subset of symmetry functions.
        if isinstance(blob, SymmetryFunctionSet):
            self._sets += blob.sets
            self._symmetryfunctions += blob.symmetryfunctions

        # Add a single symmetry function to storage.
        elif isinstance(blob, SymmetryFunction):
            self._symmetryfunctions.append(blob)
        else:
            raise NotImplementedError

        return self

    def __iter__(self) -> Iterator[SymmetryFunction]:
        """Iterate over all stored symmetry functions."""
        for symmetryfunction in self.storage:
            yield symmetryfunction

    def to_list(self) -> List[SFListType]:
        """Create a list of all stored symmetryfunctions."""
        symmetryfunction_list = []
        for symmetryfunction in self.storage:
            symmetryfunction_list.append(symmetryfunction.to_list())
        return symmetryfunction_list

    def from_list(
        self,
        symmetryfunction_list: List[SFListType]
    ) -> None:
        """Fill storage from a list of symmetry functions."""
        for entry in symmetryfunction_list:
            self.append(SymmetryFunction(sflist=entry))

    @property
    def sets(self) -> List['SymmetryFunctionSet']:
        """Show a list of all stored `SymmetryFunctionSet` objects."""
        return self._sets

    @property
    def symmetryfunctions(self) -> List[SymmetryFunction]:
        """Show a list of all stored `SymmetryFunction` objects."""
        return self._symmetryfunctions

    @property
    def storage(self) -> List[SymmetryFunction]:
        """Show all stored symmetry functions recursively."""
        storage = self.symmetryfunctions.copy()
        for sfset in self.sets:
            storage += sfset.storage

        return storage

    @property
    def sftypes(self) -> Optional[str]:
        """Show a list of symmetry function types in self.symmetryfunctions."""
        sftypes = list(set(sf.tag for sf in self.symmetryfunctions))
        if len(sftypes) == 1:
            return sftypes[0]

        if len(sftypes) > 1:
            return 'mixed'

        return None

    @property
    def elements(self) -> Optional[List[str]]:
        """Show a list of all elements covered in self.symmetryfunctions."""
        # Store all elements of all symmetryfunctions.
        elements = []
        for symfun in self.symmetryfunctions:
            if symfun.elements is not None:
                elements += symfun.elements

        # Remove duplicates.
        elements = list(set(elements))

        # If the list is empty, return None instead.
        if len(elements) == 0:
            return None

        return elements

    @property
    def cutoffs(self) -> Optional[List[Optional[float]]]:
        """Show a list of all cutoffs in self.symmetryfunctions."""
        # Collect the cutoff values of all symmetryfunctions.
        cutoffs = list(set(sf.cutoff for sf in self.symmetryfunctions))

        # If the list is empty, return None instead.
        if len(cutoffs) == 0:
            return None

        return cutoffs

    def append(
        self,
        blob: Union['SymmetryFunctionSet', SymmetryFunction]
    ) -> None:
        """Append a data `blob` to the internal storage."""
        if isinstance(blob, SymmetryFunctionSet):
            self._sets.append(blob)

        elif isinstance(blob, SymmetryFunction):
            self._symmetryfunctions.append(blob)

        else:
            raise Exception(f'{self.__class__.__name__} can only store data of'
                           + 'type SymmetryFunctionSet or SymmetryFunction.')
