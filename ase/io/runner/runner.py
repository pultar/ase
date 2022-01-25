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

import os
import re
import numpy as np
from ase.io import read
from ase.atoms import Atoms
from ase.utils import reader, writer
from ase.calculators.calculator import PropertyNotImplementedError, Parameters
from ase.data import atomic_numbers, chemical_symbols

from ase.calculators.runner.runnersinglepoint import RunnerSinglePointCalculator
from . import defaultoptions as do


class UnrecognizedKeywordError(Exception):
    """Raised if a format mistake is encountered in the input.data file."""

    def __init__(self, keyname):
        """Initialize exception with custom error message."""
        super().__init__(f"The keyword {keyname} is not recognized.")


class FileFormatError(Exception):
    """Raised if a formatting error is encountered while parsing a file."""

    def __init__(self, message=None):
        """Initialize exception with custom error message."""
        if not message:
            message = "File formatted incorrectly."
        super().__init__(message)


def reset_structure():
    """Reset per-structure arrays and variables while reading input.data."""
    symbols = []
    positions = []
    cell = []
    charges = []
    magmoms = []
    forces = []
    periodic = np.array([False, False, False])
    totalenergy = []
    totalcharge = []
    latticecount = 0

    return (symbols, positions, cell, charges, magmoms, forces, periodic,
            totalenergy, totalcharge, latticecount)


def read_runnerdata(fd, index):
    """Parse all structures within a RuNNer input.data file.

    input.data files contain all structural information needed to train a
    Behler-Parrinello-type neural network potential, e.g. Cartesian coordinates,
    atomic forces, and energies. This function reads the file object `fd` and
    returns the slice of structures given by `index`.

    Parameters
    ----------
    fd : fileobj
        Python file object with the target input.data file.
    index : int
        The slice of structures which should be returned.

    Returns
    --------
    Atoms: ase.atoms.Atoms object
        All information about the structures within `index` of `fd`,
        including symbols, positions, atomic charges, and cell lattice. Every
        `Atoms` object has a calculator `calc` attached to it with additional
        information on the total energy, atomic forces, and total charge.

    Raises
    -------
    FileFormatError : exception
        Raised when a format error in `fd` is encountered.

    References
    ----------
    Detailed information about the RuNNer input.data file format can be found
    in the program's
    [documentation](https://runner.pages.gwdg.de/runner/reference/files/#inputdata)

    """

    # Container for all images in the file.
    images = []

    # Set all per-structure containers and variables.
    (symbols, positions, cell, charges, magmoms, forces, periodic,
     totalenergy, totalcharge, latticecount) = reset_structure()

    for lineidx, line in enumerate(fd):
        # Jump over blank lines.
        if line.strip() == "":
            continue

        # First word of each line must be a valid keyword.
        keyword = line.split()[0]

        if keyword not in do.RUNNERDATA_KEYWORDS:
            raise FileFormatError(
                f"File {fd.name} is not a valid input.data file. "
                f"Illegal keyword '{keyword}' in line {lineidx+1}."
            )

        # 'begin' marks the start of a new structure.
        if keyword == 'begin':
            # Check if anything appeared in between this new structure and the
            # previous one, e.g. a poorly formatted third structure.
            if any(symbols):
                raise FileFormatError(
                    f"Structure {len(images)} beginning in line {lineidx+1}"
                    f"appears to be preceded by a poorly formatted structure."
                )

            # Set all per-structure containers and variables.
            (symbols, positions, cell, charges, magmoms, forces, periodic,
             totalenergy, totalcharge, latticecount) = reset_structure()

        # Read one atom.
        elif keyword == 'atom':
            x, y, z, symbol, charge, magmom, fx, fy, fz = line.split()[1:10]

            # Convert and process.
            symbol = symbol.lower().capitalize()

            # Append to related arrays.
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
            charges.append(float(charge))
            forces.append([float(fx), float(fy), float(fz)])
            magmoms.append(float(magmom))

        # Read one cell lattice vector.
        elif keyword == 'lattice':
            lx, ly, lz = line.split()[1:4]
            cell.append([float(lx), float(ly), float(lz)])

            periodic[latticecount] = True
            latticecount += 1

        # Read the total energy of the structure.
        elif keyword == 'energy':
            energy = float(line.split()[1])
            totalenergy.append(energy)

        # Read the total charge of the structure.
        elif keyword == 'charge':
            charge = float(line.split()[1])
            totalcharge.append(charge)

        # 'end' statement marks the end of a structure.
        elif keyword == 'end':
            # Check if there is at least one atom in the structure.
            if len(symbols) == 0:
                raise FileFormatError(
                    f'Structure {len(images)} ending in line {lineidx+1} does '
                    f'not contain any atoms.'
                )

            # Check if a charge has been specified for the structure.
            if len(totalcharge) != 1:
                raise FileFormatError(
                    f'Structure {len(images)} ending in line {lineidx+1} does '
                    f'not have exactly one total charge.'
                )

            # Check if an energy has been specified for the structure.
            if len(totalenergy) != 1:
                raise FileFormatError(
                    f'Structure {len(images)} ending in line {lineidx+1} does '
                    f'not have exactly one total energy.'
                )

            # If all checks clear, set the atoms object.
            atoms = Atoms(
                symbols=symbols,
                positions=positions
            )

            # Optional: set periodic structure properties.
            if periodic.any():
                # Check if there are exactly three lattice vectors.
                if len(cell) != 3:
                    raise FileFormatError(
                        f'Structure {len(images)} ending in line {lineidx+1}'
                        f'does not have exactly three lattice vectors.'
                    )

                atoms.set_cell(cell)
                atoms.set_pbc(periodic)

            # Optional: set magnetic moments for each atom.
            if any(magmoms):
                atoms.set_initial_magnetic_moments(magmoms)

            # Optional: set atomic charges.
            if any(charges):
                atoms.set_initial_charges(charges)

            # CAUTION: The calculator has to be attached at the very end,
            # otherwise, it is overwritten by `set_cell()`, `set_pbc()`, ...
            atoms.calc = RunnerSinglePointCalculator(
                atoms=atoms,
                energy=totalenergy[0],
                forces=forces,
                totalcharge=totalcharge[0]
            )

            # Finally, append the structure to the list of all structures.
            images.append(atoms)

            # Reset all per-structure containers and variables.
            (symbols, positions, cell, charges, magmoms, forces, periodic,
             totalenergy, totalcharge, latticecount) = reset_structure()

    # Check whether there are any structures in the file.
    if len(images) == 0:
        raise FileFormatError(f"No structures in file '{fd.name}'.")

    for atoms in images[index]:
        yield atoms


def read_runnerase(label):
    """Read structure and parameter options from a previous calculation.

    Parameters
    ----------
    label : str
        The ASE-internal calculation label, i.e. the prefix of the ASE
        parameters file.

    Returns
    --------
    atoms: ase.atoms.Atoms object
        All information about the structures within `index` of `fd`,
        including symbols, positions, atomic charges, and cell lattice. Every
        `Atoms` object has a calculator `calc` attached to it with additional
        information on the total energy, atomic forces, and total charge.
    parameters: dict
        A Python-dictionary contain all calculation options.

    """
    # Get the path to the input.data file.
    directory = ''.join(label.split('/')[:-1])

    # Read structural information from the input.data file.
    atoms = read(directory + '/input.data', ':', format='runnerdata')

    # Read RuNNer options from the ASE parameter file.
    if os.path.exists(label + '.ase'):
        parameters = Parameters.read(label + '.ase')

    elif os.path.exists(directory + '/input.nn'):
        with open(directory + '/input.nn', 'r') as fd:
            parameters = read_runnerconfig(fd)

    return atoms, parameters


def write_all_inputs(atoms, properties, parameters, raise_exception=True,
                     label='runner', *, v8_legacy_format=True, scaling=None,
                     weights=None, splittraintest=None, sfvalues=None,
                     directory='.'):
    """Write all necessary input files for performing a RuNNer calculation."""
    # Write all parameters to a .ase parameters file.
    parameters.write(label + '.ase')

    # All functions take a list of atoms objects as input.
    if not isinstance(atoms, list):
        atoms = [atoms]

    # Write the input.data file containing all structures.
    with open(directory + '/input.data', 'w') as fd:
        write_runnerdata(fd, atoms)

    # Write the input.nn file.
    with open(directory + '/input.nn', 'w') as fd:
        write_runnerconfig(fd, parameters)

    # Write scaling data.
    if scaling is not None:
        with open(directory + '/scaling.data', 'w') as fd:
            write_scaling(fd, scaling)

    # Write weights data.
    if weights is not None:
        write_weights(weights, path=directory)

    if splittraintest is not None:

        # Write the function.data and testing.data files.
        with open(directory + '/function.data', 'w') as fd:
            write_functiontestingdata(fd, sfvalues, splittraintest['training'])

        with open(directory + '/testing.data', 'w') as fd:
            write_functiontestingdata(fd, sfvalues, splittraintest['testing'])

        with open(directory + '/trainstruct.data', 'w') as fd:
            write_trainteststruct(fd, atoms, splittraintest['training'])

        with open(directory + '/teststruct.data', 'w') as fd:
            write_trainteststruct(fd, atoms, splittraintest['testing'])

        with open(directory + '/trainforces.data', 'w') as fd:
            write_traintestforces(fd, atoms, splittraintest['training'])

        with open(directory + '/testforces.data', 'w') as fd:
            write_traintestforces(fd, atoms, splittraintest['testing'])


def write_runnerdata(fd, images, comment='', fmt='%22.12f'):
    """Write series of ASE Atoms to a RuNNer input.data file.

    Parameters
    ----------
    fd : fileobj
        Python file object with the target input.data file.
    images : array-like
        List of `Atoms` objects.
    comment : str
        A comment message to be added to each structure.

    Raises
    -------
    ValueError : exception
        Raised if the comment line contains newline characters.

    """
    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line should not have line breaks.')

    for atoms in images:
        fd.write('begin\n')

        if comment != '':
            fd.write('comment %s\n' % (comment))

        # Write lattice vectors.
        if atoms.get_pbc().any():
            for vector in atoms.cell:
                lx, ly, lz = vector
                fd.write('lattice %s %s %s\n' % (fmt % lx, fmt % ly, fmt % lz))

        if atoms.calc is None:
            energy = 0.0
            totalcharge = 0.0
            forces = np.zeros(atoms.positions.shape)
        else:
            energy = atoms.get_potential_energy()
            try:
                totalcharge = atoms.calc.get_property('totalcharge')
            except PropertyNotImplementedError:
                totalcharge = np.sum(atoms.get_initial_charges())
            forces = atoms.get_forces()
        # Write atoms.
        atom = zip(
            atoms.symbols,
            atoms.positions,
            atoms.get_initial_charges(),
            atoms.get_initial_magnetic_moments(),
            forces
        )

        for s, (x, y, z), q, m, (fx, fy, fz) in atom:
            fd.write(
                'atom %s %s %s %-2s %s %s %s %s %s\n'
                % (fmt % x, fmt % y, fmt % z, s, fmt % q, fmt % m,
                   fmt % fx, fmt % fy, fmt % fz)
            )

        fd.write('energy %s\n' % (fmt % (energy)))

        # Exception handling is necessary as long as totalcharge property is not
        # finalized.
        try:
            fd.write('charge %s\n' % (fmt % totalcharge))
        except PropertyNotImplementedError:
            fd.write('charge %s\n' % (fmt % 0.0))
        fd.write('end\n')


def _format_argument(arg):
    """Format one argument value when writing the input.nn file."""
    if isinstance(arg, bool):
        farg = ''

    elif isinstance(arg, float):
        farg = f'{arg:.8f}'

    elif isinstance(arg, int) or isinstance(arg, str):
        farg = f'{arg}'

    else:
        raise Exception(f"Unknown argument type of argument '{arg}'")

    return farg


def check_valid_keywords(dict):
    """Check the validity of a keyword before accepting it as a parameter."""
    if not set(dict).issubset(do.RUNNERCONFIG_DEFAULTS.keys()):
        raise UnrecognizedKeywordError(dict)


def _format_keyword(fd, keyword, args):
    """Format one keyword-value pair when writing the input.nn file."""
    if isinstance(args, list):
        fargs = '  '.join([_format_argument(i) for i in args])

    else:
        fargs = _format_argument(args)

    fd.write(f'{keyword:30}')
    fd.write(f'{fargs}')
    fd.write('\n')


def write_runnerconfig(fd, parameters=None):
    """Write the central RuNNer parameter file input.nn."""
    if parameters is None:
        raise Exception('No parameters specified.')

    defaults = do.RUNNERCONFIG_DEFAULTS

    # Write the header.
    fd.write("### This input file for RuNNer was generated with ASE.\n")

    for keyword, arguments in parameters.items():

        # For Boolean keywords, check if they are True. If not, skip writing.
        if isinstance(arguments, bool):
            if not arguments:
                continue

        # Some keywords can occur multiple times. In that case, `arguments` will
        # hold a list of arguments for each occurence of the `keyword`.
        # Some keywords only occur once but are still a list, e.g. the list of
        # elements in the system.
        if isinstance(arguments, list) and defaults[keyword]['allow_multiple']:
            for occurence in arguments:
                _format_keyword(fd, keyword, occurence)

        else:
            _format_keyword(fd, keyword, arguments)


def _read_arguments(keyword, arguments):

    # Get the default arguments belonging to this keyword.
    defaults = do.RUNNERCONFIG_DEFAULTS[keyword]['arguments'].items()

    # Iterate over all the default arguments and check the ser-defined arguments
    # against them.
    # Some arguments also have optional parameters depending on their value.
    # For such arguments, we need a counter to offset the `idx` in the for-loop
    # below.

    arguments_formatted = []
    count_parameters = 0
    for idx, (argument, properties) in enumerate(defaults):

        # Get the variable type of this argument and format the value accordingly.
        argument_type = properties['type']

        # Some keywords have only one argument that can, however, occur multiple
        # times. This is indicated by `allow_multiple`.
        # In this case, we read all arguments here and return directly.
        if 'allow_multiple' in properties.keys() and properties['allow_multiple']:
            arguments_formatted = [argument_type(i) for i in arguments]
            return arguments_formatted

        # In all other cases, we only read the value of a single argument and
        # append it to the final list of formatted arguments.
        argumentvalue = argument_type(arguments[idx + count_parameters])
        arguments_formatted.append(argumentvalue)

        # Some keywords can only take predefined options.
        if 'options' in properties:

            # Check if this option is set to a valid value.
            if argumentvalue not in properties['options'].keys():
                raise FileFormatError(f"'{argumentvalue}' is not a valid value \
                for keyword '{keyword}'.")

            # Some options require additional parameters to be set.
            optionsettings = properties['options'][argumentvalue]
            if 'parameters' in optionsettings:
                for parameter in optionsettings['parameters'].values():
                    parameter_type = parameter['type']

                    # Read in the next value in input.nn, which is actually
                    # a parameter to this argument option.
                    parametervalue = parameter_type(arguments[idx + count_parameters + 1])

                    # Append to the final list of arguments and increase the
                    # counter of parameters that have been read.
                    arguments_formatted.append(parametervalue)
                    count_parameters += 1

    # If there is only one argument, return the value without the surrounding list.
    if len(arguments_formatted) == 1:
        return arguments_formatted[0]
    else:
        return arguments_formatted


def _read_keyword(line):
    """Read one keyword from a RuNNer input.nn file."""
    # Extract the keyword, it is always the first word in a line.
    spline = line.split()
    keyword = spline[0]

    # Check if the keyword is a valid RuNNer keyword, i.e. if it is in the
    # `defaults` dictionary.
    if keyword not in do.RUNNERCONFIG_DEFAULTS:
        raise FileFormatError(f"'{keyword}' is an unknown keyword.")

    # Boolean keywords do not have any arguments, so they are already treated
    # here.
    if len(spline) == 1:
        return True
    else:
        arguments = _read_arguments(keyword, spline[1:])

        return arguments


@reader
def read_runnerconfig(fd):
    """Read an input.nn file and store the contained dictionary data."""
    parameters = {}

    for line in fd.readlines():

        # Strip all comments, i.e. all text after a sharp sign.
        line = line.split('#')[0]

        # Skip blank lines.
        if line.strip() == '':
            continue

        # Extract the keyword, it is always the first word in a line.
        keyword = line.split()[0]
        parameter = _read_keyword(line)

        # Append keywords which can occur more than once to their list.
        if do.RUNNERCONFIG_DEFAULTS[keyword]['allow_multiple']:
            try:
                parameters[keyword].append(parameter)
            except KeyError:
                parameters[keyword] = [parameter]
        else:
            parameters[keyword] = parameter

    return Parameters(**parameters)


@reader
def read_scaling(fd):
    """Read symmetry function scaling data contained in `scaling.data` files.

    This function offers to read the scaling parameters used by RuNNer to
    transform symmetry functions. By default, these are stored in the
    `scaling.data` files generated as part of the training process.

    Parameters
    ----------
    fd: file object
        A readable Python file object (e.g. opened with the `open()` function)
        which contains the symmetry function scaling data in RuNNer format.

    Returns
    -------
    scaling: dict
        A dictionary with the three keys
        * 'scaling': np.ndarray
            The symmetry function scaling data in the following order:

            element ID, symmetry function ID, symmetry function minimum,
            symmetry function maximum, symmetry function average.

            Please note that this array structure is essentially the transpose
            of the storage format within the `scaling.data` files itself.
        * 'target_min': np.float
            The minimum value of the training target property (either energy or
            charge, depending on the training mode).
        * 'target_max': np.float
            The maximum value of the training target property (either energy or
            charge, depending on the training mode).

    Resources
    ---------
    For more information on the `scaling.data` file format in RuNNer please
    visit the
    [documentation](https://theochemgoettingen.gitlab.io/RuNNer/1.3/reference/files/#scalingdata).

    Example
    -------

    >>> with open('scaling.data', 'r') as infile:
    >>>     scalingdata = read_scaling(infile)
    >>> print(scalingdata.keys)
    ['scaling', 'target_min', 'target_max']

    """
    # Prepare the target dictionary.
    scaling = {'scaling': [], 'target_min': 0.0, 'target_max': 0.0}

    for line in fd.readlines():

        data = line.split()

        # Lines of length five hold the scaling data for each symmetry function.
        if len(data) == 5:
            # Extract the data.
            idx_element, idx_symfun, min, max, avg = data

            # Convert to the correct variable types.
            idx_element = np.int(idx_element)
            idx_symfun = np.int(idx_symfun)
            min, max, avg = np.float(min), np.float(max), np.float(avg)

            # Append to 'scaling' array in the dictionary.
            scaling['scaling'].append([idx_element, idx_symfun, min, max, avg])

        # The final line holds only two values, the minimum and maximum of the
        # target property.
        elif len(data) == 2:
            scaling['target_min'] = np.float(data[0])
            scaling['target_max'] = np.float(data[1])

        else:
            raise FileFormatError('Format error encountered while reading \
                scaling data.')

    # Convert the scaling data to a numpy array and transpose for more
    # convenient processing.
    scaling['scaling'] = np.array(scaling['scaling']).T

    return scaling


@writer
def write_scaling(fd, scaling):
    """Write symmetry function scaling data.

    This function writes the scaling parameters for each symmetry function to a
    file object `fd`, typically labeled 'scaling.data'.

    Parameters
    ----------
    fd: file object
        A writable Python file object (e.g. opened with the `open()` function).
    scaling: dict
        A dictionary containing three key-value pairs:
        * 'scaling': np.ndarray
            The symmetry function scaling data in the following order:

            element ID, symmetry function ID, symmetry function minimum,
            symmetry function maximum, symmetry function average.

            Please note that this array structure is essentially the transpose
            of the storage format within the `scaling.data` files itself.
        * 'target_min': np.float
            The minimum value of the training target property (either energy or
            charge, depending on the training mode).
        * 'target_max': np.float
            The maximum value of the training target property (either energy or
            charge, depending on the training mode).

    Resources
    ---------
    For more information on the `scaling.data` file format in RuNNer please
    visit the
    [documentation](https://theochemgoettingen.gitlab.io/RuNNer/1.3/reference/files/#scalingdata).

    Example
    -------

    >>> with open('scaling.data', 'w') as infile:
    >>>     write_scaling(infile, scalingdata)

    """
    # RuNNer lists the scaling data ordered by symmetry functions, which is the
    # transpose format of how it is stored here.
    scalingdata = scaling['scaling'].T

    # First, write the scaling data for each symmetry function.
    np.savetxt(fd, scalingdata, fmt='%4d %5d %18.9f %18.9f %18.9f')

    # The last line contains the minimum and maximum of the target property.
    fd.write(f"{scaling['target_min']:18.9f} {scaling['target_max']:18.9f}\n")


def read_weights(path='.', elements=None, prefix='weights', suffix='data'):
    """Read weights of atomic neural networks.

    This function read the weights of atomic neural networks, typically
    optimized during training of a neural network potential with RuNNer in Mode
    2.
    RuNNer stores weights in two different formats which can both be read
    by this routine:
    * The optimized weights stored either in 'weights.XXX.data' or
      'optweights.XXX.data' files, where 'XXX' stands for the atomic number of
      the element.
    * The intermediate weights stored in 'YYYYYY.short.XXX.out' files with the
      atomic number 'XXX' and the training epoch 'YYYYYY'.

    In the latter case, this method will only read the first column of the file
    as it contains the actual weight values.

    In contrast to most other I/O routines in the RuNNer module, this one
    deliberately does not take a file object as the input parameter. This is
    because reading weights typically means reading more than one file (one
    for each element).

    Optional Parameters
    -------------------
    path : string, _default_='.'
        The base path where all weight files to be read reside.
    elements : list of strings, _default_=`None`
        The chemical symbols of the elements for which the weights should be
        read. The default behaviour (= no list is supplied) is to read all files
        under `path` matching the name format `prefix`.XXX.`suffix` where XXX
        stands for any atomic number.
    prefix : string, _default_='weights'
        The file name prefix. Typically, weight files are labelled according to
        the scheme `prefix`.XXX.`suffix` where XXX stands for any chemical
        symbol.
    suffix : string, _default_='data'
        The file name suffix. Typically, weight files are labelled according to
        the scheme `prefix`.XXX.`suffix` where XXX stands for any chemical
        symbol.

    Returns
    -------
    weights: dict
        A dictionary holding one key-value-pair for each requested `element`.
        The keys are the chemical symbols, while the values are np.ndarrays
        containing the weights for that element.

        If no weight files are found (e.g. due to a fit without improvement)
        an empty dictionary is returned.

    Resources
    ---------
    For more information on the `weights.XXX.data` file format in RuNNer please
    visit the
    [documentation](https://theochemgoettingen.gitlab.io/RuNNer/1.3/reference/files/#weightsxxxdata).

    Example
    -------

    >>> weights = read_weights('mode2', elements=['C'])
    {'C': [...]}

    """
    weights = {}

    # If a list of elements was supplied, only read the corresponding weight
    # files (faster than reading all weight files and sorting later).
    if elements:
        for element in elements:

            # Obtain the atomic number of the element and the path to the file.
            id = atomic_numbers[element]
            fullpath = os.path.join(path, prefix + f'.{id:03d}.' + suffix)

            # Store the weights as a np.ndarray.
            weights[element] = np.genfromtxt(fullpath, usecols=(0))

    # If no elements where supplied, read in all weight files that can be found.
    else:
        for file in os.listdir(path):
            if file.startswith(prefix):
                fullpath = os.path.join(path, file)

                # Transform the atomic numbers into the chemical symbol.
                element = chemical_symbols[int(file.split('.')[1])]

                # Store the weights as a np.ndarray.
                weights[element] = np.genfromtxt(fullpath, usecols=(0))

    return weights


def write_weights(weights, path='.', elements=None, prefix='weights', suffix='data'):
    """Write weights of atomic neural networks.

    This function writes the weights of atomic neural networks contained in
    `weights_dict`. By default, this data is stored in `weights.XXX.data` files
    where `XXX` stands for the atomic number of the element in question.

    In contrast to most other I/O routines in the RuNNer module, this one
    deliberately does not take a file object as the input parameter. This is
    because writing weights typically means writing more than one file (one
    for each element).

    Parameters
    ----------
    weights_dict : dict
        A dictionary holding one key-value for each `element` that shall be
        written. The keys should be the chemical symbol while the value should
        be np.ndarrays holding the weights.

    Optional Parameters
    -------------------
    path : string, _default_='.'
        The base path where all weight files to be read reside.
    elements : list of strings, _default_=`None`
        The chemical symbols of the elements for which the weights should be
        written.
    prefix : string, _default_='weights'
        The file name prefix. Typically, weight files are labelled according to
        the scheme `prefix`.XXX.`suffix` where XXX stands for any chemical
        symbol.
    suffix : string, _default_='data'
        The file name suffix. Typically, weight files are labelled according to
        the scheme `prefix`.XXX.`suffix` where XXX stands for any chemical
        symbol.

    Resources
    ---------
    For more information on the `weights.XXX.data` file format in RuNNer please
    visit the
    [documentation](https://theochemgoettingen.gitlab.io/RuNNer/1.3/reference/files/#weightsxxxdata).

    """
    # If requested, choose only the weights of `elements` for writing.
    if elements:
        weights = {element: weights[element] for element in elements}

    for element, weights in weights.items():
        id = atomic_numbers[element]
        np.savetxt(f'{path}/{prefix}.{id:03d}.{suffix}', weights, fmt='%.10f')


def read_results_mode1(label, directory):

    with open(f'{directory}/function.data', 'r') as fd:
        sfvalues_training = read_functiontestingdata(fd)

    with open(f'{directory}/testing.data', 'r') as fd:
        sfvalues_testing = read_functiontestingdata(fd)

    with open(f'{label}.out', 'r') as fd:
        splittraintest = read_splittraintest(fd)

    # Fill the full list of sfvalues.
    sfvalues = []
    trainidx = 0
    testidx = 0
    for i in range(len(sfvalues_training) + len(sfvalues_testing)):
        if i in splittraintest['training']:
            sfvalues.append(sfvalues_training[trainidx])
            trainidx += 1
        else:
            sfvalues.append(sfvalues_testing[testidx])
            testidx += 1

    return sfvalues, splittraintest


def read_results_mode2(label, directory):

    # Store the results of the training process.
    with open(f'{label}.out', 'r') as fd:
        fitresults = read_fitresults(fd)

    # Store the weights of the atomic neural networks.
    # Mode 2 writes them to the optweights.XXX.out file.
    weights = read_weights(path=directory, prefix='optweights', suffix='out')

    # Store the symmetry function scaling data.
    with open(f'{directory}/scaling.data', 'r') as fd:
        scaling = read_scaling(fd)

    return fitresults, weights, scaling


def read_results_mode3(label, directory):

    # Read predicted structures from the output.data file.
    predicted_structures = read(directory + '/output.data', ':', format='runnerdata')
    energy = 0.0
    forces = 0.0

    return predicted_structures, energy, forces


def read_fitresults(fd):

    # Initialize the fitresults dictionary.
    fitresults = {'epochs': [],
                  'rmse_energy_training': [], 'rmse_energy_testing': [],
                  'rmse_forces_training': [], 'rmse_forces_testing': [],
                  'rmse_charge_training': [], 'rmse_charge_testing': [],
                  'unit_rmse_energy': '', 'unit_rmse_force': ''}

    for line in fd.readlines():
        spline = line.split()

        # Read the RMSEs of energies, forces and charges, and the corresponding
        # epochs.
        if line.strip().startswith('ENERGY'):
            if '*****' in line:
                epoch, rmse_train, rmse_test = None, None, None
            else:
                epoch = int(spline[1])
                rmse_train, rmse_test = float(spline[2]), float(spline[3])

            fitresults['epochs'].append(epoch)
            fitresults['rmse_energy_training'].append(rmse_train)
            fitresults['rmse_energy_testing'].append(rmse_test)

        elif line.strip().startswith('FORCES'):
            if '*****' in line:
                rmse_train, rmse_test = None, None
            else:
                rmse_train, rmse_test = float(spline[2]), float(spline[3])

            fitresults['rmse_forces_training'].append(rmse_train)
            fitresults['rmse_forces_testing'].append(rmse_test)

        elif line.strip().startswith('CHARGE'):
            rmse_train, rmse_test = spline[2:4]
            fitresults['rmse_charge_training'].append(float(rmse_train))
            fitresults['rmse_charge_testing'].append(float(rmse_test))

        # Read the fitting units, indicated by the heading 'RMSEs'.
        if 'RMSEs' in line:
            # Use regular expressions to find the units. All units conveniently
            # start with two letters ('Ha', or 'eV'), followed by a slash and
            # some more letters (e.g. 'Bohr', or 'atom').
            units = re.findall(r'\w{2}/\w+', line)
            fitresults['unit_rmse_energy'] = units[0]
            fitresults['unit_rmse_force'] = units[0]

        # Read in the epoch where the best fit was obtained.
        if 'Best short range fit has been obtained in epoch' in line:
            fitresults['opt_rmse_epoch'] = int(spline[-1])

        # Explicitely handle the special case that the fit did not yield any
        # improvement. This also means that no weights were written.
        if 'No improvement' in line:
            fitresults['opt_rmse_epoch'] = None

    return fitresults


def read_splittraintest(fd):

    lines = fd.readlines()

    traintest = {'training': [], 'testing': []}
    for line in lines:
        if 'Point is used for' in line:
            # In Python, indices start at 0, therefore we subtract 1.
            point_idx = int(line.split()[0]) - 1
            family = line.split()[5]

            traintest[family].append(point_idx)

    return traintest


def read_functiontestingdata(fd):

    lines = fd.readlines()

    data = []

    for line in lines:
        spline = line.split()

        # Line of length 1 marks a new structure and holds the # of atoms.
        if len(spline) == 1:
            natoms = int(spline[0])
            structure = {
                'energy_total': 0.0,
                'energy_short': 0.0,
                'energy_elec': 0.0,
                'charge': 0.0,
                'sfvalues': [],
                'elements': []
            }

        # Line of length 4 marks the end of a structure.
        elif len(spline) == 4:
            structure['totalcharge'] = float(spline[0])
            structure['energy_total'] = float(spline[1])
            structure['energy_short'] = float(spline[2])
            structure['energy_elec'] = float(spline[3])

            if len(structure['sfvalues']) != natoms:
                raise FileFormatError('function.data appears malformatted.')

            data.append(structure)

        # All other lines hold symmetry function values.
        else:
            sfvalues = np.array([float(i) for i in spline[1:]])
            structure['sfvalues'].append(sfvalues)
            structure['elements'].append(int(spline[0]))

    return data


def write_functiontestingdata(fd, images, index=':', fmt='22.12f'):

    images = [images[i] for i in index]

    for image in images:

        q = image['totalcharge']
        e_tot = image['energy_total']
        e_short = image['energy_short']
        e_elec = image['energy_elec']

        fd.write(f"{len(image['sfvalues']):6}\n")

        for (element, sf) in zip(image['elements'], image['sfvalues']):
            fd.write(f'{element:3}')
            fd.write(''.join(f'{i:{fmt}}' for i in sf))
            fd.write('\n')

        fd.write(f'{q:{fmt}} {e_tot:{fmt}} {e_short:{fmt}} {e_elec:{fmt}}\n')


def write_trainteststruct(fd, images, index=":", fmt='22.12f'):
    """Write series of ASE Atoms to a RuNNer input.data file.

    Parameters
    ----------
    fd : fileobj
        Python file object with the target input.data file.
    images : array-like
        List of `Atoms` objects.
    comment : str
        A comment message to be added to each structure.

    Raises
    -------
    ValueError : exception
        Raised if the comment line contains newline characters.

    """
    # Index starts counting at 1.
    images = [images[i] for i in index]

    for idx_atoms, atoms in enumerate(images):

        # Write structure index. White space at the end is important.
        fd.write(f'{idx_atoms + 1:8} ')

        # Write lattice vectors for periodic structures.
        if atoms.get_pbc().any():
            fd.write('T\n')
            for (lx, ly, lz) in atoms.cell:
                fd.write(f'lattice {lx:{fmt}} {ly:{fmt}} {lz:{fmt}} \n')
        else:
            fd.write('F\n')

        if atoms.calc is None:
            forces = np.zeros(atoms.positions.shape)
        else:
            forces = atoms.get_forces()

        # Write atoms.
        atom = zip(
            [atomic_numbers[i] for i in atoms.symbols],
            atoms.positions,
            atoms.get_initial_charges(),
            atoms.get_initial_magnetic_moments(),
            forces
        )

        for s, (x, y, z), q, m, (fx, fy, fz) in atom:
            fd.write(
                f'{s:3} {x:{fmt}} {y:{fmt}} {z:{fmt}} {q:{fmt}} {m:{fmt}} '
                + f'{fx:{fmt}} {fy:{fmt}} {fz:{fmt}}\n'
            )


def write_traintestforces(fd, images, index=":", fmt='22.12f'):
    """Write series of ASE Atoms to a RuNNer input.data file.

    Parameters
    ----------
    fd : fileobj
        Python file object with the target input.data file.
    images : array-like
        List of `Atoms` objects.
    comment : str
        A comment message to be added to each structure.

    Raises
    -------
    ValueError : exception
        Raised if the comment line contains newline characters.

    """
    # Index starts counting at 1.
    images = [images[i] for i in index]

    for idx_atoms, atoms in enumerate(images):

        # Write structure index.
        fd.write(f'{idx_atoms + 1:8}\n')

        if atoms.calc is None:
            forces = np.zeros(atoms.positions.shape)
        else:
            forces = atoms.get_forces()

        for (fx, fy, fz) in forces:
            fd.write(
                f'{fx:{fmt}} {fy:{fmt}} {fz:{fmt}}\n'
            )
