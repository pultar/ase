"""Implementation of readers and writers for RuNNer.

This module provides a Python I/O-interface to the RuNNer Neural Network Energy
Representation (RuNNer), a framework for the  construction of high-dimensional
neural network potentials developed in the group of Prof. Dr. Jörg Behler at
Georg-August-Universität Göttingen.

Reference
---------
- [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
- Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Optional, Union, Iterator, Tuple, List, Any, Dict

import os
import io
import numpy as np

from ase.io import read
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.calculators.calculator import Parameters
from ase.units import Hartree, Bohr
from ase.data import atomic_numbers

from ase.calculators.runner.runnersinglepoint import RunnerSinglePointCalculator
from . import defaultoptions as do
from .storageclasses import (RunnerScaling, RunnerFitResults,
                             RunnerSplitTrainTest, RunnerWeights,
                             RunnerSymmetryFunctionValues,
                             RunnerResults,
                             SymmetryFunction, SymmetryFunctionSet)

# Custom type for numpy arrays. This maintains backwards compatibility with
# numpy >= 1.20 as a type similar to this is natively available since the
# introduction of the numpy.typing module.
NDArray = np.ndarray


class UnrecognizedKeywordError(Exception):
    """Error class for marking format mistakes in input.nn parameter files."""

    def __init__(self, keyname: str) -> None:
        """Initialize exception with custom error message."""
        super().__init__(f"The keyword {keyname} is not recognized.")


class FileFormatError(Exception):
    """Generic error class for marking format mistakes in RuNNer files."""

    def __init__(self, message: Optional[str] = None) -> None:
        """Initialize exception with custom error message."""
        if not message:
            message = "File formatted incorrectly."

        super().__init__(message)


# A lot of structural information has to be stored.
# pylint: disable=too-many-instance-attributes
class TempAtoms:
    """Container for storing data about one atomic structure.

    This class is intended for fast I/O of structural data. It is required as
    the builtin ASE `Atoms` and `Atom` classes are a bottleneck for large data
    files. This is mostly because adding an `Atom` to an `Atoms` object over
    and over again comes with a lot of overhead because the attached arrays have
    to be checked and copied.
    In constrast, it is very efficient to store atomic positions, symbols, etc.
    in long lists and simply create one ASE `Atoms` object once all atoms have
    been collected. In summary, `TempAtoms` is simply a container to hold all
    these lists in one convenient place.
    """

    def __init__(self, atoms: Optional[Atoms] = None) -> None:
        """Initialize the object."""
        self.positions: List[List[float]] = []
        self.symbols: List[str] = []
        self.charges: List[float] = []
        self.magmoms: List[float] = []

        self.cell: List[List[float]] = []

        self.energy: float = np.NaN
        self.totalcharge: float = np.NaN
        self.forces: List[List[float]] = []

        # If given, read values from a given ASE Atoms object.
        if atoms is not None:
            self.from_aseatoms(atoms)

    def __iter__(
        self
    ) -> Iterator[Tuple[List[float], str, float, float, List[float]]]:
        """Iterate over tuples of information for each atom in storage."""
        for idx, xyz in enumerate(self.positions):
            element = self.symbols[idx]
            charge = self.charges[idx]
            atomenergy = self.magmoms[idx]
            force_xyz = self.forces[idx]

            yield (xyz, element, charge, atomenergy, force_xyz)

    @property
    def pbc(self) -> List[bool]:
        """Show the periodicity of the object along the Cartesian axes."""
        pbc = [False, False, False]
        for idx, vector in enumerate(self.cell):
            if len(vector) == 3:
                pbc[idx] = True

        return pbc

    def to_aseatoms(self) -> Atoms:
        """Convert the object to an ASE Atoms object."""
        atoms = Atoms(
            positions=self.positions,
            symbols=self.symbols,
            pbc=self.pbc
        )

        # Add the unit cell information.
        if any(self.pbc):
            atoms.set_cell(self.cell)

        atoms.set_initial_charges(self.charges)
        atoms.set_initial_magnetic_moments(self.magmoms)

        calc = RunnerSinglePointCalculator(
            atoms,
            energy=self.energy,
            forces=self.forces,
            totalcharge=self.totalcharge
        )

        # The calculator has to be attached at the very end,
        # otherwise, it is overwritten by `set_cell()`, `set_pbc()`, ...
        atoms.calc = calc

        return atoms

    def from_aseatoms(self, atoms: Atoms) -> None:
        """Convert an ASE Atoms object to this class."""
        self.positions = list(atoms.positions)
        self.symbols = list(atoms.symbols)
        self.charges = list(atoms.get_initial_charges())
        self.magmoms = list(atoms.get_initial_magnetic_moments())

        if any(atoms.pbc):
            self.cell = atoms.cell

        if atoms.calc is not None:
            self.energy = atoms.get_potential_energy()
            self.forces = list(atoms.get_forces())
            if isinstance(atoms.calc, RunnerSinglePointCalculator):
                self.totalcharge = atoms.calc.get_property('totalcharge')
            else:
                self.totalcharge = np.sum(atoms.get_initial_charges())
        else:
            self.energy = 0.0
            self.forces = [[0.0, 0.0, 0.0] for i in self.positions]
            self.totalcharge = 0.0

    def convert(self, input_units: str, output_units: str) -> None:
        """Convert all data from `input_units` to `output_units`."""
        # Transform lists into numpy arrays for faster processing.

        if input_units == output_units:
            pass

        elif input_units == 'atomic' and output_units == 'si':
            self.positions = [[i * Bohr for i in xyz] for xyz in self.positions]
            self.cell = [[i * Bohr for i in xyz] for xyz in self.cell]
            self.energy *= Hartree
            self.forces = [[i * Hartree * Bohr for i in xyz] for xyz in self.forces]

        elif input_units == 'si' and output_units == 'atomic':
            self.positions = [[i / Bohr for i in xyz] for xyz in self.positions]
            self.cell = [[i / Bohr for i in xyz] for xyz in self.cell]
            self.energy /= Hartree
            self.forces = [[i / Hartree * Bohr for i in xyz] for xyz in self.forces]


@reader
def read_runnerdata(
    infile: io.TextIOWrapper,
    index: Union[int, slice] = -1,
    input_units: str = 'atomic',
    output_units: str = 'si'
) -> Iterator[Atoms]:
    """Parse all structures within a RuNNer input.data file.

    input.data files contain all structural information needed to train a
    Behler-Parrinello-type neural network potential, e.g. Cartesian coordinates,
    atomic forces, and energies. This function reads the file object `infile`
    and returns the slice of structures given by `index`. All structures will
    be converted to SI units by default.

    Parameters
    ----------
    infile : TextIOWrapper
        Python fileobj with the target input.data file.
    index : int or slice, _default_ -1
        The slice of structures which should be returned. Returns only the last
        structure by default.
    input_units : str
        The given input units. Can be 'si' or 'atomic'.
    output_units : str
        The desired output units. Can be 'si' or 'atomic'.

    Returns
    --------
    Atoms : Atoms
        All information about the structures within `index` of `infile`,
        including symbols, positions, atomic charges, and cell lattice. Every
        `Atoms` object has a `RunnerSinglePointCalculator` attached with
        additional information on the total energy, atomic forces, and total
        charge.

    References
    ----------
    Detailed information about the RuNNer input.data file format can be found
    in the program's
    [documentation](https://runner.pages.gwdg.de/runner/reference/files/#inputdata)
    """
    # Container for all images in the file.
    images: List[Atoms] = []

    for line in infile:

        # Jump over blank lines and comments.
        if not line.strip() or line.strip().startswith('#'):
            continue

        # Split the line into the keyword and the arguments.
        keyword, arguments = line.split()[0], line.split()[1:]

        # 'begin' marks the start of a new structure.
        if keyword == 'begin':
            atoms = TempAtoms()

        # Read one atom.
        elif keyword == 'atom':
            xyz = [float(i) for i in arguments[:3]]
            symbol = arguments[3].lower().capitalize()
            charge = float(arguments[4])
            magmom = float(arguments[5])
            force_xyz = [float(i) for i in arguments[6:9]]

            atoms.positions.append(xyz)
            atoms.symbols.append(symbol)
            atoms.charges.append(charge)
            atoms.magmoms.append(magmom)
            atoms.forces.append(force_xyz)

        # Read one cell lattice vector.
        elif keyword == 'lattice':
            atoms.cell.append([float(i) for i in arguments[:3]])

        # Read the total energy of the structure.
        elif keyword == 'energy':
            atoms.energy = float(arguments[0])

        # Read the total charge of the structure.
        elif keyword == 'charge':
            atoms.totalcharge = float(arguments[0])

        # 'end' statement marks the end of a structure.
        elif keyword == 'end':

            # Convert all data to the specified output units.
            atoms.convert(input_units, output_units)

            # Append the structure to the list of all structures.
            aseatoms = atoms.to_aseatoms()
            images.append(aseatoms)

    for aseatoms in images[index]:
        yield aseatoms


@writer
def write_runnerdata(
    outfile: io.TextIOWrapper,
    images: List[Atoms],
    comment: str = '',
    fmt: str = '16.10f',
    input_units: str = 'si'
) -> None:
    """Write series of ASE Atoms to a RuNNer input.data file.

    For further details see the `read_runnerdata` routine.

    Parameters
    ----------
    outfile : TextIOWrapper
        Python fileobj with the target input.data file.
    images : array-like
        List of `Atoms` objects.
    comment : str
        A comment message to be added to each structure.
    fmt : str
        A format specifier for float values.
    input_units : str
        The given input units. Can be 'si' or 'atomic'.

    Raises
    -------
    ValueError : exception
        Raised if the comment line contains newline characters.
    """
    # Preprocess the comment.
    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line cannot contain line breaks.')

    for atoms in images:

        # Transform into a TempAtoms object.
        tempatoms = TempAtoms(atoms)

        # Convert, if necessary.
        tempatoms.convert(input_units=input_units, output_units='atomic')

        # Begin writing this structure to file.
        outfile.write('begin\n')

        if comment != '':
            outfile.write(f'comment {comment}\n')

        # Write lattice vectors if the structure is marked as periodic.
        if any(tempatoms.pbc):
            for vector in tempatoms.cell:
                outfile.write(f'lattice {vector[0]:{fmt}} {vector[1]:{fmt}} '
                            + f'{vector[2]:{fmt}}\n')

        for xyz, element, charge, atomenergy, force_xyz in tempatoms:
            outfile.write(f'atom {xyz[0]:{fmt}} {xyz[1]:{fmt}} {xyz[2]:{fmt}} '
                        + f'{element:2s} {charge:{fmt}} {atomenergy:{fmt}} '
                        + f'{force_xyz[0]:{fmt}} {force_xyz[1]:{fmt}} '
                        + f'{force_xyz[2]:{fmt}}\n')

        # Write the energy and total charge, then end this structure.
        outfile.write(f'energy {tempatoms.energy:{fmt}}\n')
        outfile.write(f'charge {tempatoms.totalcharge:{fmt}}\n')
        outfile.write('end\n')


def read_runnerase(
    label: str
) -> Tuple[Union[Atoms, List[Atoms], None], Parameters]:
    """Read structure and parameter options from a previous calculation.

    Parameters
    ----------
    label : str
        The ASE-internal calculation label, i.e. the prefix of the ASE
        parameters file.

    Returns
    --------
    atoms : ASE Atoms or List[Atoms]
        A list of all structures associated with the given calculation.
    parameters : Parameters
        A dictionary containing all RuNNer settings of the calculation.
    """
    # Extract the directory path.
    directory = ''.join(label.split('/')[:-1])

    # Join the paths to all relevant files.
    inputdata_path = os.path.join(directory, 'input.data')
    aseparams_path = f'{label}.ase'
    inputnn_path = os.path.join(directory, 'input.nn')

    # Read structural information from the input.data file.
    if os.path.exists(inputdata_path):
        atoms: Optional[Union[Atoms, List[Atoms]]] = read(
            inputdata_path, ':', format='runnerdata'
        )
    else:
        atoms = None

    # Read RuNNer options first from the ASE parameter file and second from
    # the input.nn file.
    if os.path.exists(aseparams_path):
        parameters: Parameters = Parameters.read(aseparams_path)
        if 'symfunction_short' in parameters:
            parameters['symfunction_short'] = SymmetryFunctionSet(parameters['symfunction_short'])

    elif os.path.exists(inputnn_path):
        parameters = read_runnerconfig(inputnn_path)

    else:
        parameters = Parameters()

    return atoms, parameters


def check_valid_keywords(keyword_dict):
    """Check whether a keyword is valid before accepting it as a parameter."""
    if not set(keyword_dict).issubset(do.RUNNERCONFIG_DEFAULTS.keys()):
        raise UnrecognizedKeywordError(keyword_dict)


def _format_argument(argument: Union[bool, float, int, str]) -> str:
    """Format one argument value when writing the input.nn file."""
    if isinstance(argument, bool):
        argument_formatted = ''

    elif isinstance(argument, float):
        argument_formatted = f'{argument:.8f}'

    elif isinstance(argument, (int, str)):
        argument_formatted = f'{argument}'

    else:
        raise FileFormatError(f"Unknown argument type of argument '{argument}'")

    return argument_formatted


def _format_keyword(
    outfile: io.TextIOWrapper,
    keyword: str,
    arguments: Union[List[Union[int, float, bool, str]],
                     int, float, bool, str,
                     SymmetryFunctionSet]
) -> None:
    """Write one keyword-value pair in input.nn format.

    Parameters
    ----------
    outfile : TextIOWrapper
        The fileobj where the data will be written.
    keyword : str
        The keyword to which the `arguments` belong.
    arguments : int or float or bool or str or list thereof
        List of arguments or single argument, unformatted.
    """
    # If `arguments` contains a list of arguments, each entry is formatted
    # individually and joined to one large string.
    if isinstance(arguments, list):
        arguments_formatted = ' '.join([_format_argument(i) for i in arguments])
        outfile.write(f'{keyword:30}')
        outfile.write(f'{arguments_formatted}\n')

    # `SymmetryFunction`s have their own to_runner() write routine.
    elif isinstance(arguments, SymmetryFunctionSet):
        for symmetryfunction in arguments:
            outfile.write(f'{keyword:30}')
            outfile.write(f'{symmetryfunction.to_runner()}\n')

    # All other arguments are simply formatted once and written.
    else:
        arguments_formatted = _format_argument(arguments)
        outfile.write(f'{keyword:30}')
        outfile.write(f'{arguments_formatted}\n')


@writer
def write_runnerconfig(
    outfile: io.TextIOWrapper,
    parameters: Parameters
) -> None:
    """Write the central RuNNer parameter file input.nn.

    The routine iterates over all keywords in `parameters` and calls
    `_format_keyword` for each entry. `_format_keyword` in turn calls
    `_format_argument` for each argument belonging to the keyword.

    Parameters
    ----------
    outfile : TextIOWrapper
        The fileobj to which the parameters will be written.
    parameters : Parameters
        A dict-like collection of RuNNer parameters.
    """
    # Store all RuNNer default parameters.
    defaults = do.RUNNERCONFIG_DEFAULTS

    # Write the header.
    outfile.write("### This input file for RuNNer was generated with ASE.\n")

    for keyword, arguments in parameters.items():

        # Skip Boolean keywords which are set to False.
        if isinstance(arguments, bool) and arguments is False:
            continue

        # Some keywords can occur multiple times. This is indicated by the
        # 'allow_multiple' flag in the defaults dictionary. In that case,
        # `arguments` will hold a list of arguments for each occurence of the
        # `keyword`.
        # Some keywords only occur once but are still a list, e.g. the list
        # of elements in the system.
        if isinstance(arguments, list) and defaults[keyword]['allow_multiple']:
            for occurence in arguments:
                _format_keyword(outfile, keyword, occurence)

        else:
            _format_keyword(outfile, keyword, arguments)


def _read_arguments(
    keyword: str,
    arguments: List[str]
) -> Union[bool, int, float, str,
           List[Union[bool, int, float, str]],
           SymmetryFunction]:

    # Get the default arguments belonging to this keyword.
    # Type is ignored because the default's dictionary is not typed yet.
    defaults = do.RUNNERCONFIG_DEFAULTS[keyword]['arguments'].items()  # type: ignore

    # Iterate over all the default arguments and check the user-defined
    # arguments against them.
    # Some arguments also have optional parameters depending on their value.
    # For such arguments, the 'count_parameters' counter is needed
    # to offset the `idx` in the for-loop below.
    arguments_formatted = []
    count_parameters = 0
    for idx, (_, properties) in enumerate(defaults):

        # Get the variable type of this argument and format the value accordingly.
        argument_type = properties['type']

        # Some keywords have only one argument that can, however, occur multiple
        # times. This is indicated by `allow_multiple`.
        # In this case, we read all arguments here and return directly.
        if 'allow_multiple' in properties and properties['allow_multiple']:
            arguments_formatted = [argument_type(i) for i in arguments]
            return arguments_formatted

        # In all other cases, we only read the value of a single argument and
        # append it to the final list of formatted arguments.
        argumentvalue = arguments[idx + count_parameters]

        # Catch legacy float format with Fortran double precision marker.
        if 'd0' in argumentvalue:
            argumentvalue = argumentvalue.split('d0')[0]

        argumentvalue = argument_type(argumentvalue)
        arguments_formatted.append(argumentvalue)

        # Some keywords can only take predefined options.
        if 'options' in properties:

            # Check if this option is set to a valid value.
            if argumentvalue not in properties['options'].keys():
                raise FileFormatError(f"'{argumentvalue}' is not a valid value "
                                      + f"for keyword '{keyword}'.")

            # Some options require additional parameters to be set.
            optionsettings = properties['options'][argumentvalue]
            if 'parameters' in optionsettings:
                for parameter in optionsettings['parameters'].values():
                    parameter_type = parameter['type']

                    # Read in the next value in input.nn, which is actually
                    # a parameter to this argument option.
                    parametervalue = parameter_type(
                        arguments[idx + count_parameters + 1]
                    )

                    # Append to the final list of arguments and increase the
                    # counter of parameters that have been read.
                    arguments_formatted.append(parametervalue)
                    count_parameters += 1

    # If there is only one argument, return the value without the surrounding list.
    if len(arguments_formatted) == 1:
        return arguments_formatted[0]

    # Treat symmetry function separately.
    if 'symfunction' in keyword:
        # After running through this routine, all 'symfuncion_*' keywords
        # will have the correct list formatting.
        return SymmetryFunction(sflist=arguments_formatted)  # type: ignore

    return arguments_formatted


@reader
def read_runnerconfig(
    infile: io.TextIOWrapper,
    check: bool = True
) -> Parameters:
    """Read an input.nn file and store the contained dictionary data.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj in RuNNer input.nn format from which the data will be read.
    check : bool, _default_ True
        A flag for controlling whether keywords will be checked against the
        default's dictionary. If `True`, only known keywords are accepted.
    """
    parameters: Parameters = Parameters()

    # Initialize the symmetry function containers.
    parameters['symfunction_short'] = SymmetryFunctionSet()

    for line in infile:

        # Strip all comments (all text after a sharp sign) and skip blank lines.
        line = line.split('#')[0]
        if line.strip() == '':
            continue

        # Extract the keyword, it is always the first word in a line.
        spline = line.split()
        keyword = spline[0]

        # Check if the keyword is a valid RuNNer keyword, i.e. if it is in the
        # `defaults` dictionary.
        if check is True and keyword not in do.RUNNERCONFIG_DEFAULTS:
            raise FileFormatError(f"'{keyword}' is an unknown keyword.")

        # Format the parameters to this keyword correctly. If the line only
        # has the keyword and no parameters, it is a Boolean and set to `True`.
        if len(spline) != 1:
            parameter = _read_arguments(keyword, spline[1:])
        else:
            parameter = True

        # Append keywords which can occur more than once to their list.
        if do.RUNNERCONFIG_DEFAULTS[keyword]['allow_multiple']:
            if keyword not in parameters:
                parameters[keyword] = []

            parameters[keyword].append(parameter)
        else:
            parameters[keyword] = parameter

    return parameters


# RuNNer operates in several modi, all of which take different arguments.
# For clarity it is intentionally chosen to pass these explicitely, even though
# it increases the number of parameters.
# pylint: disable=too-many-arguments
def write_all_inputs(
    atoms: Union[Atoms, List[Atoms]],
    parameters: Parameters,
    label: str = 'runner',
    directory: str = '.',
    scaling: Optional[RunnerScaling] = None,
    weights: Optional[RunnerWeights] = None,
    splittraintest: Optional[RunnerSplitTrainTest] = None,
    sfvalues: Optional[RunnerSymmetryFunctionValues] = None
) -> None:
    """Write all necessary input files for performing a RuNNer calculation."""
    # All functions take a list of atoms objects as input.
    if not isinstance(atoms, list):
        atoms = [atoms]

    # Write all parameters to a .ase parameters file.
    parameters.write(f'{label}.ase')

    # Write the input.data file containing all structures.
    path = os.path.join(directory, 'input.data')
    write_runnerdata(path, atoms)

    # Write the input.nn file containing all parameters.
    path = os.path.join(directory, 'input.nn')
    write_runnerconfig(path, parameters)

    # Write scaling data.
    if scaling is not None:
        path = os.path.join(directory, 'scaling.data')
        write_scaling(path, scaling)

    # Write weights data.
    if weights is not None:
        write_weights(weights, path=directory)

    if splittraintest is not None:
        path = os.path.join(directory, 'function.data')
        write_functiontestingdata(path, sfvalues, splittraintest.train)

        path = os.path.join(directory, 'testing.data')
        write_functiontestingdata(path, sfvalues, splittraintest.test)

        path = os.path.join(directory, 'trainstruct.data')
        write_trainteststruct(path, atoms, splittraintest.train)

        path = os.path.join(directory, 'teststruct.data')
        write_trainteststruct(path, atoms, splittraintest.test)

        path = os.path.join(directory, 'trainforces.data')
        write_traintestforces(path, atoms, splittraintest.train)

        path = os.path.join(directory, 'testforces.data')
        write_traintestforces(path, atoms, splittraintest.test)


@reader
def read_scaling(infile: io.TextIOWrapper) -> RunnerScaling:
    """Read symmetry function scaling data contained in 'scaling.data' files.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj from which the data will be read.

    Returns
    -------
    RunnerScaling : RunnerScaling
        A RunnerScaling object containing the data in `infile`.
    """
    return RunnerScaling(infile)


@writer
def write_scaling(outfile: io.TextIOWrapper, scaling: RunnerScaling) -> None:
    """Write symmetry function scaling data to RuNNer 'scaling.data' format.

    Parameters
    ----------
    outfile : TextIOWrapper
        The fileobj to which the data will be written.
    scaling : RunnerScaling
        The scaling data.
    """
    scaling.write(outfile)


def read_weights(
    path: str = '.',
    elements: Optional[List[str]] = None,
    prefix: str = 'weights',
    suffix: str = 'data'
) -> RunnerWeights:
    """Read the weights of atomic neural networks.

    Parameters
    ----------
    path : str, optional, _default_ '.'
        Data will be read from all weight files under the given directory.
    elements : List[str], optional, _default_ `None`
        A selection of chemical symbols for which the weights under `path`
        will be read.
    prefix : str, optional, _default_ `weights`
        The filename prefix of weight files under `path`.
    suffix : str, optional, _default_ `data`
        The filename suffix of weight files under `path`.

    Returns
    -------
    RunnerWeights : RunnerWeights
        A RunnerWeights object containing the data of all weights, ordered by
        element.
    """
    return RunnerWeights(path=path, elements=elements, prefix=prefix,
                         suffix=suffix)


def write_weights(
    weights: RunnerWeights,
    path: str = '.',
    elements: Optional[List[str]] = None,
    prefix: str = 'weights',
    suffix: str = 'data'
) -> None:
    """Write the weights of atomic neural networks in RuNNer format.

    Parameters
    ----------
    weights : RunnerWeights
        The weights data.
    path : str, optional, _default_ '.'
        Data will be read from all weight files under the given directory.
    elements : List[str], optional, _default_ `None`
        A selection of chemical symbols for which the weights under `path`
        will be read.
    prefix : str, optional, _default_ `weights`
        The filename prefix of weight files under `path`.
    suffix : str, optional, _default_ `data`
        The filename suffix of weight files under `path`.
    """
    weights.write(path, elements, prefix, suffix)


@reader
def read_fitresults(infile: io.TextIOWrapper) -> RunnerFitResults:
    """Read training process results from stdout of RuNNer Mode 2.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj from which the data will be read.

    Returns
    -------
    RunnerFitResults : RunnerFitResults
        A RunnerFitResults object containing the data in `infile`.
    """
    return RunnerFitResults(infile)


@reader
def read_splittraintest(infile: io.TextIOWrapper):
    """Read the split between train and test set from stdout of RuNNer Mode 1.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj from which the data will be read.

    Returns
    -------
    RunnerSplitTrainTest : RunnerSplitTrainTest
        A RunnerSplitTrainTest object containing the data in `infile`.
    """
    return RunnerSplitTrainTest(infile)


@reader
def read_functiontestingdata(
    infile: io.TextIOWrapper
) -> RunnerSymmetryFunctionValues:
    """Read symmetry function values from function.data or testing.data files.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj from which the data will be read.

    Returns
    -------
    RunnerSymmetryFunctionValues : RunnerSymmetryFunctionValues
        A RunnerSymmetryFunctionValues object containing the data in `infile`.
    """
    return RunnerSymmetryFunctionValues(infile)


@writer
def write_functiontestingdata(
    outfile: io.TextIOWrapper,
    sfvalues: RunnerSymmetryFunctionValues,
    index: Union[int, slice] = slice(0, None),
    fmt: str = '22.12f'
) -> None:
    """Write symmetry function values to function.data or testing.data files.

    Parameters
    ----------
    outfile : TextIOWrapper
        The fileobj from which the data will be read.
    sfvalues : RunnerSymmetryFunctionValues
        The symmetry function values.
    index : int or slice, default `slice(0, None)`
        A selection of structures for which the data will be written. By
        default, all data in storage is written.
    fmt : str
        A format specifier for float values.
    """
    sfvalues.write(outfile, index, fmt)


def read_results_mode1(label: str, directory: str) -> Dict[str, object]:
    """Read all results of RuNNer Mode 1.

    Parameters
    ----------
    label : str
        The ASE calculator label of the calculation. Typically this is the
        joined path of the `directory` and the .ase parameter file prefix.
    directory : str
        The path of the directory which holds the calculation files.

    Returns
    -------
    dict : RunnerResults
        A dictionary with two entries
            - sfvalues : RunnerSymmetryFunctionValues
                The symmetry function values.
            - splittraintest : RunnerSplitTrainTest
                The split between train and test set.
    """
    sfvalues_train = read_functiontestingdata(f'{directory}/function.data')
    sfvalues_test = read_functiontestingdata(f'{directory}/testing.data')
    splittraintest = read_splittraintest(f'{label}.out')

    # Store only one symmetry function value container.
    sfvalues = sfvalues_train + sfvalues_test
    sfvalues.sort(splittraintest.train + splittraintest.test)

    return {'sfvalues': sfvalues, 'splittraintest': splittraintest}


def read_results_mode2(label: str, directory: str) -> Dict[str, object]:
    """Read all results of RuNNer Mode 2.

    Parameters
    ----------
    label : str
        The ASE calculator label of the calculation. Typically this is the
        joined path of the `directory` and the .ase parameter file prefix.
    directory : str
        The path of the directory which holds the calculation files.

    Returns
    -------
    dict : RunnerResults
        A dictionary with three entries
            - fitresults : RunnerFitResults
                Details about the fitting process.
            - weights : RunnerWeights
                The atomic neural network weights and bias values.
            - scaling : RunnerScaling
                The symmetry function scaling data.
    """
    # Store training results, weights, and symmetry function scaling data.
    # Mode 2 writes best weights to the optweights.XXX.out file.
    fitresults = read_fitresults(f'{label}.out')
    weights = read_weights(path=directory, prefix='optweights', suffix='out')
    scaling = read_scaling(f'{directory}/scaling.data')

    return {'fitresults': fitresults, 'weights': weights, 'scaling': scaling}


def read_results_mode3(directory: str) -> Dict[str, object]:
    """Read all results of RuNNer Mode 3.

    Parameters
    ----------
    directory : str
        The path of the directory which holds the calculation files.

    Returns
    -------
    dict : RunnerResults
        A dictionary with two entries
            - energy : float or np.ndarray
                The total energy of all structures. In case of a single
                structure only one float value is returned.
            - forces : np.ndarray
                The atomic forces of all structures.
    """
    # Read predicted structures from the output.data file.
    # `read` automatically converts all properties to SI units.
    path = f'{directory}/output.data'
    predicted_structures = read(path, ':', format='runnerdata')
    energies = np.array([i.get_potential_energy() for i in predicted_structures])
    forces = np.array([i.get_forces() for i in predicted_structures])

    # For just one structure, flatten the energy and force arrays.
    if forces.shape[0] == 1:
        forces = forces[0, :, :]

    if energies.shape[0] == 1:
        return {'energy': float(energies[0]), 'forces': forces}

    return {'energy': energies, 'forces': forces}


@writer
def write_trainteststruct(
    outfile: io.TextIOWrapper,
    images: Union[Atoms, List[Atoms]],
    index: Union[int, slice] = slice(0, None),
    fmt: str = '16.10f',
    input_units: str = 'si'
) -> None:
    """Write a series of ASE Atoms to trainstruct.data / teststruct.data format.

    Parameters
    ----------
    outfile : TextIOWrapper
        The fileobj where the data will be written.
    images : List[Atoms]
        List of ASE `Atoms` objects.
    index : int or slice, _default_ `slice(0, None)`
        Only the selection of `images` given by `index` will be written.
    fmt : str, _default_ '16.10f'
        A format specifier for float values.
    input_units : str, _default_ 'si'
        The units within `images`. Can be 'si' or 'atomic'. All data will
        automatically be converted to atomic units.
    """
    # Filter the images which should be printed according to `index`.
    if isinstance(index, (int, slice)):
        images = images[index]
    else:
        images = [images[i] for i in index]

    for idx_atoms, atoms in enumerate(images):

        # Transform into a TempAtoms object and do unit conversion, if needed.
        tempatoms = TempAtoms(atoms)
        tempatoms.convert(input_units=input_units, output_units='atomic')

        # Write structure index. White space at the end is important.
        outfile.write(f'{idx_atoms + 1:8} ')

        # Write lattice vectors for periodic structures.
        if any(tempatoms.pbc):
            outfile.write('T\n')
            for vector in tempatoms.cell:
                outfile.write(f'{vector[0]:{fmt}} {vector[1]:{fmt}} '
                            + f'{vector[2]:{fmt}} \n')
        else:
            outfile.write('F\n')

        # Write atomic data to file.
        for xyz, element, charge, atomenergy, force_xyz in tempatoms:
            atomic_number = atomic_numbers[element]
            outfile.write(f'{atomic_number:4d} {xyz[0]:{fmt}} {xyz[1]:{fmt}} '
                        + f'{xyz[2]:{fmt}} '
                        + f'{charge:{fmt}} {atomenergy:{fmt}} '
                        + f'{force_xyz[0]:{fmt}} {force_xyz[1]:{fmt}} '
                        + f'{force_xyz[2]:{fmt}}\n')


@writer
def write_traintestforces(
    outfile: io.TextIOWrapper,
    images: Union[Atoms, List[Atoms]],
    index: Union[int, slice, List[int]] = slice(0, None),
    fmt: str = '16.10f',
    input_units: str = 'si'
) -> None:
    """Write a series of ASE Atoms to trainforces.data / testforces.data format.

    Parameters
    ----------
    outfile : TextIOWrapper
        The fileobj where the data will be written.
    images : List[Atoms]
        List of ASE `Atoms` objects.
    index : int or slice, _default_ `slice(0, None)`
        Only the selection of `images` given by `index` will be written.
    fmt : str, _default_ '16.10f'
        A format specifier for float values.
    input_units : str, _default_ 'si'
        The units within `images`. Can be 'si' or 'atomic'. All data will
        automatically be converted to atomic units.
    """
    # Filter the images which should be printed according to `index`.
    if isinstance(index, (int, slice)):
        images = images[index]
    else:
        images = [images[i] for i in index]

    for idx_atoms, atoms in enumerate(images):

        # Transform into a TempAtoms object and do unit conversion, if needed.
        tempatoms = TempAtoms(atoms)
        tempatoms.convert(input_units=input_units, output_units='atomic')

        # Write structure index. White space at the end is important.
        outfile.write(f'{idx_atoms + 1:8}\n')

        # Write atomic data to file.
        for _, _, _, _, force_xyz in tempatoms:
            outfile.write(f'{force_xyz[0]:{fmt}} {force_xyz[1]:{fmt}} '
                        + f'{force_xyz[2]:{fmt}}\n')


@reader
def read_traintestpoints(
    infile: io.TextIOWrapper,
    input_units: str = 'atomic',
    output_units: str = 'si'
) -> NDArray:
    """Read RuNNer trainpoint.XXXXXX.out / testpoint.XXXXXX.out.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj where the data will be read.
    input_units : str, _default_ 'atomic'
        The units within `images`. Can be 'si' or 'atomic'. All data will
        automatically be converted to `output_units`.
    output_units : str, _default_ 'si'
        The desired units in `data`. Can be 'si' or 'atomic'.

    Returns
    -------
    data : np.ndarray
        An array holding the following columns: image ID, number of atoms,
        reference energy, neural network energy, difference between ref. and
        neural network energy.
    """
    # The first row holds the column names.
    data: NDArray = np.loadtxt(infile, skiprows=1)

    # Unit conversion.
    if input_units == 'atomic' and output_units == 'si':
        data[:, 2:] *= Hartree

    elif input_units == 'si' and output_units == 'atomic':
        data[:, 2:] /= Hartree

    return data


@reader
def read_traintestforces(
    infile: io.TextIOWrapper,
    input_units: str = 'atomic',
    output_units: str = 'si'
) -> NDArray:
    """Read RuNNer trainforces.XXXXXX.our / testpoint.XXXXXX.out.

    Parameters
    ----------
    infile : TextIOWrapper
        The fileobj where the data will be read.
    input_units : str, _default_ 'atomic'
        The units within `images`. Can be 'si' or 'atomic'. All data will
        automatically be converted to `output_units`.
    output_units : str, _default_ 'si'
        The desired units in `data`. Can be 'si' or 'atomic'.

    Returns
    -------
    data : np.ndarray
        An array holding the following columns: image ID, atom ID, reference
        force x, reference force y, reference force z, neural network force x,
        neural network force y, neural network force z.
    """
    # The first row holds the column names.
    data: NDArray = np.loadtxt(infile, skiprows=1)

    # Unit conversion.
    if input_units == 'atomic' and output_units == 'si':
        data[:, 2:] *= Hartree * Bohr

    elif input_units == 'si' and output_units == 'atomic':
        data[:, 2:] /= Hartree * Bohr

    return data
