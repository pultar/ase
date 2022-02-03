"""Implementation of a calculator for machine-learning with RuNNer.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Contains
--------

calc_radial_symfuns : utility function
    Returns a list of RuNNer radial symmetry functions.
calc_angular_symfuns : utility function
    Returns a list of RuNNer angular symmetry functions.

Runner : FileIOCalculator
    The main calculator for training and evaluating HDNNPs with RuNNer.

Reference
---------
* [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
* Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from itertools import combinations_with_replacement, product

import numpy as np

from ase.calculators.calculator import (PropertyNotImplementedError,
                                        FileIOCalculator,
                                        compare_atoms)
from ase.geometry import get_distances
from ase.data import atomic_numbers

import ase.io.runner.runner as io

# Sensible default parameters for running RuNNer with ASE.
DEFAULT_PARAMETERS: dict = {
    # General for all modes.
    'runner_mode': 1,                   # Default should be to start a new fit.
    'elements': None,                   # Auto-set by ASE when attaching Atoms.
    'number_of_elements': None,         # Auto-set by ASE when attaching Atoms.
    'bond_threshold': 0.5,              # Default OK but system-dependent.
    'nn_type_short': 1,                 # Most people use atomic NNs.
    #'nnp_gen': 2,                       # 2Gs remain the most common use case.
    'use_short_nn': True,               # Short-range fitting is the default.
    'optmode_charge': 1,                # Default OK but option is relevant.
    'optmode_short_energy': 1,          # Default OK but option is relevant.
    'optmode_short_force': 1,           # Default OK but option is relevant.
    'points_in_memory': 1000,           # Default value is legacy.
    'scale_symmetry_functions': True,   # Scaling is used by almost everyone.
    'cutoff_type': 1,                   # Default OK, but important.
    # Mode 1.
    'test_fraction': 0.1,               # Default too small, more common.
    # Mode 1 and 2.
    'random_seed': 0,                   # Set via utility function set_seed().
    'use_short_forces': True,           # Force fitting is standard procedure.
    # Mode 1 and 3.
    'remove_atom_energies': True,       # Everyone only fits binding energies.
    'atom_energy': [],                  # Dependency of `remove_atom_energies`.
    # Mode 2.
    'epochs': 30,                       # Default is 0, 30 is common use case.
    'kalman_lambda_short': 0.98000,     # Typical use case.
    'kalman_nue_short': 0.99870,        # Typical use case.
    'mix_all_points': True,             # This is a standard option for most.
    'nguyen_widrow_weights_short': True,  # Typically improves the fit.
    'repeated_energy_update': True,       # Default is False, but usage common.
    'short_energy_error_threshold': 0.1,  # Use only energies > 0.1*RMSE.
    'short_energy_fraction': 1.0,         # All energies are used.
    'short_force_error_threshold': 1.0,   # All forces are used.
    'short_force_fraction': 0.1,        # 10% of the forces are used.
    'use_old_weights_charge': False,    # Relevant for calculation restart.
    'use_old_weights_short': False,     # Relevant for calculation restart.
    'write_weights_epoch': 5,           # Default is 1, very verbose.
    # Mode 2 and 3.
    'center_symmetry_functions': True,  # This is standard procedure.
    'precondition_weights': True,       # This is standard procedure.
    # Mode 3.
    'calculate_forces': False,          # Auto-set by ASE but important.
    'calculate_stress': False,          # Auto-set by ASE but important.
    # ---------- 2G-specific keywords. ---------------------------------------#
    # All modes.
    'symfunction_short': [],
    # Mode 2 and 3.
    'global_activation_short': [None],
    'global_hidden_layers_short': None,
    'global_nodes_short': [None],
    # ### PairNN-specific keywords.
    # # All modes.
    # 'element_pairsymfunction_short': [None, None, None],
    # 'global_pairsymfunction_short': [None, None, None],
    # # Mode 2 and 3.
    # 'element_activation_pair': [None, None, None, None],
    # 'element_hidden_layers_pair': [None, None],
    # 'element_nodes_pair': [None, None, 'global_nodes_pair'],
    # 'global_activation_pair': None,
    # 'global_hidden_layers_pair': None,
    # 'global_nodes_pair': None,
    # # 3G-/4G-specific keywords.
    # 'electrostatic_type': 1,            # Default ok, but should be visible.
    # 'use_fixed_charges': False,
    # 'use_gausswidth': False,
    # 'fixed_gausswidth': [None, -99.0],
    # 'element_symfunction_electrostatic': [None, None, None],
    # 'global_symfunction_electrostatic': [None, None, None],
    # 'symfunction_electrostatic': [None, None, None],
    #
    # 'element_activation_electrostatic': [None, None, None, None],
    # 'element_activation_pair': [None, None, None, None],
    # 'element_activation_short': [None, None, None, None],
    # 'element_hidden_layers_electrostatic': [None, None],
    # 'element_hidden_layers_pair': [None, None],
    # 'element_hidden_layers_short': [None, None],
    # 'element_nodes_electrostatic': [None, None, 'global_nodes_electrostatic'],
    # 'element_nodes_pair': [None, None, 'global_nodes_pair'],
    # 'element_nodes_short': [None, None, 'global_nodes_short'],
    # 'global_activation_electrostatic': None,
    # 'global_activation_pair': None,
    # 'global_activation_short': None,
    # 'global_hidden_layers_electrostatic': None,
    # 'global_hidden_layers_pair': None,
    # 'global_hidden_layers_short': None,
    # 'global_nodes_electrostatic': None,
    # 'global_nodes_pair': None,
    # 'global_nodes_short': None,
    # # Mode 1.
    # 'enforce_totcharge': 1,
    # # Mode 2.
    # 'kalman_lambda_charge': 0.98000,           # Very common choice.
    # 'kalman_nue_charge': 0.99870,              # Very common choice.
    # 'charge_error_threshold': 0.0,             # Default is ok, but changed often.
    # 'charge_fraction': 1.0,                    # Default is 0, must be changed.
    # 'nguyen_widrow_weights_ewald': True,       # Typically improves the fit.
    # 'regularize_fit_param': 0.00001,           # Very important for charge fitting.
    # # Mode 2 and 3.
    # 'use_electrostatics': False,
    # 'ewald_alpha': 0.0,
    # 'ewald_cutoff': 0.0,
    # 'ewald_kmax': 0,
    # 'ewald_prec': -1.0,
    # 'fixed_charge': [None, 0.0],
    # 'use_atom_charges': True,
    # 'element_activation_electrostatic': [None, None, None, None],
    # 'element_hidden_layers_electrostatic': [None, None],
    # 'element_nodes_electrostatic': [None, None, 'global_nodes_electrostatic'],
    # 'global_activation_electrostatic': None,
    # 'global_hidden_layers_electrostatic': None,
    # 'global_nodes_electrostatic': None
}


class DatasetUndefiniedError(Exception):
    """Raise this error if no structure data is available to a calculator."""

    def __init__(self, message=None):
        """Initialize the class.

        Optional Arguments:
        -------------------
        message : string, _default_=`None`
            The error message to be shown to the user.
        """
        # Define a default message.
        if message is None:
            message = 'No training dataset defined yet.'

        # Raise.
        super().__init__(message)


# pylint: disable=R0913
def calc_symfun_coefficients(dataset, elements, cutoff, rmins=None,
                             n_radial=6, n_angular=4, algorithm='half'):
    """Calculate is the docstring."""
    if dataset is None:
        raise DatasetUndefiniedError()

    if elements is None:
        raise Exception('No elements defined yet.')

    # If no rmin was specified, try to determine it ourselves.
    if rmins is None:
        rmins = get_minimum_distances(dataset, elements)

    symmetryfunctions = []

    # Calculate radial symmetry functions = two-body terms.
    for elements_group in get_element_groups(elements, 2):
        label = '-'.join(elements_group)
        coefficients = calc_radial_symfuns(cutoff, rmins[label],
                                           n_radial, algorithm)

        for coefficient in coefficients:
            new_symfun = coefficient_to_symfun(label, coefficient, cutoff)
            symmetryfunctions.append(new_symfun)

    # Calculate angular symmetry functions = three-body terms.
    for elements_group in get_element_groups(elements, 3):
        label = '-'.join(elements_group)

        coefficients = calc_angular_symfuns(cutoff, n_angular, 'literature')

        for lambd, zetas in coefficients:
            for zeta in zetas:
                new_symfun = coefficient_to_symfun(label, (lambd, zeta), cutoff)
                symmetryfunctions.append(new_symfun)

    return symmetryfunctions


def coefficient_to_symfun(label, coefficient, cutoff):
    """Calculate this is the docstring."""
    elements = label.split('-')

    # Two elements indicates a radial symmetry function.
    if len(elements) == 2:
        elem1, elem2 = elements
        eta = coefficient
        symmetryfunction = [elem1, 2, elem2, eta, 0.0, cutoff]

    elif len(elements) == 3:
        elem1, elem2, elem3 = elements
        lamb, zeta = coefficient
        symmetryfunction = [elem1, 3, elem2, elem3, 0.0, lamb, zeta, cutoff]

    return symmetryfunction


def calc_radial_symfuns(cutoff, rmin, n_radial, algorithm):
    """Calculate the coefficients of radial symmetry functions."""
    dturn = 0.5 * cutoff - rmin
    interval = dturn / (n_radial - 1)

    #ETAS = [0.0, 0.013, 0.034, 0.078, 0.189, 0.667]
    #ETAS = [0.0, 0.008, 0.022, 0.051, 0.131, 0.475]
    eta = np.zeros(n_radial)
    for i in range(n_radial):
        rturn = 0.5 * cutoff - interval * i

        # Equally spaced at G(r) = 0.5.
        if algorithm == 'half':
            eta[i] = np.log(np.cos(np.pi * rturn/cutoff) + 1) / rturn**2

        # Equally spaced turning points.
        elif algorithm == 'turn':
            w = np.pi * rturn / cutoff
            p = 2 * (np.cos(w) + 1)
            q = 8 * p * rturn**2
            t = 2 * p - 4 * w * np.sin(w)
            sqrtterm = np.sqrt(t**2 + q * (np.pi**2/cutoff**2) * np.cos(w))
            eta[i] = (t + sqrtterm) / q

        #y = np.exp(-eta*x**2) * 0.5 * (np.cos(np.pi*x/cutoff) + 1)

    return eta


def calc_angular_symfuns(cutoff, n_angular, algorithm):
    """Calculate the coefficients of angular symmetry functions."""
    # Hard-coded literature values for the zeta parameter.
    zeta_lit = [1.0, 4.0, 8.0, 16.0, 64.0]
    lambdas = [1.0, -1.0]

    # Calculate the interval between reference points.
    interval = 160.0 / n_angular

    # Calculate the zeta values for each symmetry function.
    params = []

    for lamb in lambdas:
        zeta = np.zeros(n_angular)
        for i in range(n_angular):

            # Get the next point of reference.
            tturn = (160.0 - interval * i) / 180.0 * np.pi

            # Equally spaced at G(r) = 1.0.
            if algorithm == 'half':
                w = 1.0 + lamb * np.cos(tturn)
                zeta[i] = -np.log(2) / (np.log(w) - np.log(2))

            # Equally spaced turning points.
            elif algorithm == 'turn':
                w = 1.0 + lamb * np.cos(tturn)
                zeta[i] = 1 + (np.cos(tturn) / np.sin(tturn)**2) * w / lamb

            # Literature turning points.
            elif algorithm == 'literature':
                if n_angular > 5:
                    raise PropertyNotImplementedError('This')
                zeta[i] = zeta_lit[i]

            #y = 2**(1-zeta[i]) * (1.0 + lamb * np.cos(theta))**zeta[i]

        params.append([lamb, zeta])

    return params


def get_element_groups(elements, groupsize):
    """Create doubles or triplets of elements from all `elements`."""
    # Build pairs of elements.
    if groupsize == 2:
        groups = product(elements, repeat=2)
    
    # Build triples of elements.
    elif groupsize == 3:
        groups = combinations_with_replacement(elements, 2)
        groups = product(groups, elements)
        groups = [(a, b, c) for (a, b), c in groups]

    return groups


def get_minimum_distances(dataset, elements):
    """Calculate min. distance between all `elements` pairs in `dataset`."""
    minimum_distances = {}
    for elem1, elem2 in get_element_groups(elements, 2):
        for structure in dataset:

            elems = structure.get_chemical_symbols()

            # All positions of one element.
            pos1 = structure.positions[np.array(elems) == elem1]
            pos2 = structure.positions[np.array(elems) == elem2]

            distmatrix = get_distances(pos1, pos2)[1]

            # Remove same atom interaction.
            flat = distmatrix.flatten()
            flat = flat[flat > 0.0]

            dmin = min(flat)
            label = '-'.join([elem1, elem2])
            try:
                if minimum_distances[label] > dmin:
                    minimum_distances[label] = dmin
            except KeyError:
                minimum_distances[label] = dmin

    return minimum_distances

#---------- A calculator for RuNNer -------------------------------------------#


class Runner(FileIOCalculator):  # pylint: disable=too-many-ancestors
    """Class for training and evaluating neural network potentials with RuNNer.

    The RuNNer Neural Network Energy Representation is a framework for the
    construction of high-dimensional neural network potentials developed in the
    group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

    The default parameters are mostly those of the RuNNer Fortran code.
    There are, however, a few exceptions. This is supposed to facilitate
    typical application cases when using RuNNer via ASE. These changes are
    documented in the default_parameters of this class.

    RuNNer operates in three different modes:
        - Mode 1: Calculation of symmetry function values. Symmetry functions
                  are many-body descriptors for the chemical environment of an
                  atom.
        - Mode 2: Fitting of the potential energy surface.
        - Mode 3: Prediction. Use the previously generated high-dimensional
                  potential energy surface to predict the energy and force
                  of an unknown chemical configuration.

    The different modes generate a lot of output:
        - Mode 1:
            - sfvalues:       The values of the symmetry functions for each
                              atom.
            - splittraintest: which structures belong to the training and which
                              to the testing set. ASE needs this
                              information to generate the relevant input files
                              for RuNNer Mode 2.
        - Mode 2:
            - fit:            The performance of the training process.
            - weights:        The neural network weights.
            - scaling:        The symmetry function scaling factors.

        - Mode 3: predict the total energy and atomic forces for a structure.

    """

    implemented_properties = ['energy', 'forces', 'stress', 'charges',
                              'sfvalues', 'splittraintest', 'fit']
    command = 'RuNNer.x > PREFIX.out'

    # FIXME: is this really what we want?
    discard_results_on_any_change = False

    default_parameters = DEFAULT_PARAMETERS

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='runner', atoms=None,
                 v8_legacy_format=None,
                 **kwargs):
        """Construct RuNNer-calculator object.

        Parameters
        ----------
        restart: str
            Directory and label of an existing calculation for restarting.
        ignore_bad_restart_file: deprecated
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'runner'.
        atoms: ASE Atoms
            Atoms object to be attached to this calculator.
        kwargs: dict
            Arbitrary key-value pairs can be passed to this class upon
            initialization. The RuNNer-specific keys `dataset`, `weights`,
            `scaling`, `sfvalues`, and `splittraintest` will be processed as
            part of this routine, all other keys are passed on for base class
            initialiation.

        Examples
        --------
        Run Mode 1 from scratch with existing input.nn and input.data files.

        >>> dataset = read('input.data', ':', format='runnerdata')
        >>> with open('input.nn', 'r') as fd:
        >>>     options = read_runnerconfig(fd)
        >>> RO = Runner(dataset=dataset, **options)
        >>> RO.mode1()

        Restart Mode 1:

        >>> RO = Runner(restart='mode1/mode1')
        >>> print(RO.results['splittraintest'])

        Run Mode 2:

        >>> RO = Runner(restart='mode1/mode1')
        >>> RO.mode2()
        >>> print(RO.results['fit'])

        Update some input parameters:

        >>> RO.set(epochs=20)
        >>> RO.set(use_old_weights_short=False)

        Restart Mode 2 and run Mode 3:

        >>> RO = Runner(restart='mode2/mode2')
        >>> RO.mode3()

        Run Mode 3 with self-defined weights:

        >>> RO = Runner(
        >>>    scaling=scaling,
        >>>    weights=weights,
        >>>    dataset=dataset,
        >>>    **options
        >>> )
        """
        self.v8_legacy_format = v8_legacy_format
        self.dataset = None
        self.weights = None
        self.scaling = None
        self.sfvalues = None
        self.splittraintest = None

        # Remove the RuNNer-specific, user-supplied keywords before base class
        # initialization.
        dataset = kwargs.pop('dataset', None)
        scaling = kwargs.pop('scaling', None)
        weights = kwargs.pop('weights', None)
        sfvalues = kwargs.pop('sfvalues', None)
        splittraintest = kwargs.pop('splittraintest', None)

        # Initialize the parent class.
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                                  atoms, **kwargs)

        # Load additional parameters depending on the NN type unless they were
        # read from a restart file.
        # if self.parameters['use_short_nn'] and not restart:
        #     self.set(**self.default_parameters_short)

        if dataset is not None:
            self.set_dataset(dataset)
        if scaling is not None:
            self.set_scaling(scaling)
        if weights is not None:
            self.set_weights(weights)
        if sfvalues is not None:
            self.set_sfvalues(sfvalues)
        if splittraintest is not None:
            self.set_splittraintest(splittraintest)

    def set(self, **kwargs):
        """Update `self.parameters` with `kwargs`.

        Extends the base class (FileIOCalculator) routine to change
        `self.parameters` with keyword validation.
        """
        # Catch invalid keywords.
        if isinstance(kwargs, dict):
            io.check_valid_keywords(kwargs)

        changed_parameters = FileIOCalculator.set(self, **kwargs)

        return changed_parameters

    def set_dataset(self, dataset):
        """Store a training dataset. symmetry function scaling data.

        This method stores symmetry function scaling data as results of the
        calculator and also attaches the same information to the calculator as
        a property.
        This is required as RuNNer operates in several modes, thus some
        calculation results serve as input parameters to other modes.

        Parameters
        ----------
        dataset : list of RunnerSinglePointCalculator objects
            A dataset of structures used for training the neural network
            potential.
        """
        self.dataset = dataset

    def set_elements(self):
        """Store chemical symbols of elements in the parameters dictionary.

        This method extracts the chemical symbols of all elements connected to
        this calculator and stores them in the parameters dictionary.
        The elements are either extracted from the training `dataset` or from
        an ASE `atoms` object that has been attached to the calculator.

        Raises
        ------
        DatasetUndefiniedError : exception
            Raised if neither a `dataset` nor an `Atoms` object have been
            defined.

        """
        # Elements will either be extracted from the training dataset or from
        # the `Atoms` object to which the calculator has been attached.
        atoms = self.dataset or [self.atoms]
        if atoms is None:
            raise DatasetUndefiniedError()

        # Get the chemical symbol of all elements.
        elements = [i.get_chemical_symbols() for i in atoms]

        # Remove repeated elements.
        elements = list(set(elements))

        # Sort the elements by atomic number.
        elements.sort(key=lambda i: atomic_numbers[i])

        self.parameters['elements'] = elements

    def set_scaling(self, scaling):
        """Store symmetry function scaling data.

        This method stores symmetry function scaling data as results of the
        calculator and also attaches the same information to the calculator as
        a property.
        This is required as RuNNer operates in several modes, thus some
        calculation results serve as input parameters to other modes.

        Parameters
        ----------
        scaling : dict
            The symmetry function scaling data. See the `ase.io.runner` module
            for details in the structure of this dictionary.

        """
        self.results['scaling'] = scaling
        self.scaling = scaling

    def set_weights(self, weights):
        """Store weights of the atomic neural networks.

        This method stores the weights of the atomic neural networks as results
        of the calculator and also attaches the same information to the
        calculator as a property.
        This is required as RuNNer operates in several modes, thus some
        calculation results serve as input parameters to other modes.

        Parameters
        ----------
        weights : dict
            A dictionary containing one key-value pair for each element in the
            system and the corresponding atomic neural network weights, provided
            in the form of a np.ndarray. See the `ase.io.runner` module for
            details in the structure of this dictionary.

        """
        self.results['weights'] = weights
        self.weights = weights

    def set_symmetryfunctions(self,
                              symmetryfunctions=None,
                              coeff_radial=None, coeff_angular=None,
                              cutoff=12.0, rmins=None,
                              n_radial=6, n_angular=4, type='turn'):
        """Set symmetry function parameters in RuNNer.

        This routine offers several different options for setting the
        'symfunction_short' keyword in RuNNer. The symmetry functions can be
        calculated from scratch, generated from a list of existing coefficients,
        or the user can simply provide finished symmetry functions.

        Parameters
        ----------
        weights : dict
            A dictionary containing one key-value pair for each element in the
            system and the corresponding atomic neural network weights, provided
            in the form of a np.ndarray. See the `ase.io.runner` module for
            details in the structure of this dictionary.

        """
        # If the user supplied finished symmetry functions.
        if symmetryfunctions is None:

            if self.parameters.elements is None:
                self.set_elements()

            symmetryfunctions = calc_symfun_coefficients(
                self.dataset,
                self.parameters.elements,
                cutoff
            )

        # FIXME: If the user provided coefficients.
        # if coeff_radial is not None or coeff_angular is not None:
        #     pass

        for symfun in symmetryfunctions:
            self.parameters.symfunction_short.append(symfun)
            # self.set(**{'symfunction_short': symfun})

    def set_sfvalues(self, sfvalues):
        """Store symmetry function values.

        This method stores the symmetry function values, which were typically
        calculated during RuNNer Mode 1, as results of the calculator and also
        attaches the same information to the calculator as a property.
        This is required as RuNNer operates in several modes, thus some
        calculation results serve as input parameters to other modes.

        Parameters
        ----------
        sfvalues : list of dicts
            Each dictionary in this list contains the symmetry function values
            for one structure in the dataset. See the `ase.io.runner` module for
            details on the structure of these dictionaries.

        """
        self.results['sfvalues'] = sfvalues
        self.sfvalues = sfvalues

    def set_splittraintest(self, splittraintest):
        """Store the splitting data between training and testing set.

        This method stores the symmetry splitting data between training and
        testing set, which was typically alculated during RuNNer Mode 1, as
        results of the calculator and also attaches the same information to the
        calculator as a property.
        This is required as RuNNer operates in several modes, thus some
        calculation results serve as input parameters to other modes.

        Parameters
        ----------
        splittraintest : dict
            This dictionary contains one np.ndarray for the training and testing
            data set, respectively, holding the ID of the included structures.

        """
        self.results['splittraintest'] = splittraintest
        self.splittraintest = splittraintest

    def set_fitresults(self, fitresults):
        """Store the results of a RuNNer training process.

        This method stores the fitting results, typically obtained during RuNNer
        Mode 2, as results of the calculator.

        Parameters
        ----------
        fitresults : dict
            This dictionary contains the results of the training process. See
            the `ase.io.runner` module for details in the structure of this
            dictionary.

        """
        self.results['fit'] = fitresults

    def set_energy(self, energy):
        """Store the predicted energy of a structure.

        This method stores the predicted energy of a structure, typically
        obtained during RuNNer Mode 3, as result of the calculator.

        Parameters
        ----------
        energy : float
            The energy of the predicted structure.

        """
        self.results['energy'] = energy

    def set_forces(self, forces):
        """Store the predicted forces of a structure.

        This method stores the predicted forces of a structure, typically
        obtained during RuNNer Mode 3, as result of the calculator.

        Parameters
        ----------
        forces : np.ndarray
            The force components for each atom of the predicted structure.

        """
        self.results['forces'] = forces

    def set_prediction(self, prediction):
        """Store the predicted structure.

        This method stores the predicted structure, typically
        obtained during RuNNer Mode 3, as a result of the calculator.

        Parameters
        ----------
        prediction : ASE Atoms object or List of ASE Atoms objects
            The predicted structures.

        """
        self.results['prediction'] = prediction

    def set_atoms(self, atoms):
        """Attach an ASE `atoms` object to the calculator."""
        # If users try to attach a list instead of an Atoms object,
        # remind them that list of Atoms are typically used as the training
        # dataset.
        if isinstance(atoms, list):
            raise Exception('This seems to be a list. Please set it as \
            the `dataset` instead.')

        self.atoms = atoms

    def mode1(self, label='mode1/mode1'):
        """Calculate symmetry functions and dataset split, i.e. RuNNer Mode 1.

        This method runs the first mode of RuNNer, in which two properties are
        calculated:
            - symmetry function values for all structures.
            - The split between the training and testing set.

        Optional Parameters
        -------------------
        label : string, _default_='mode1/mode1'
            The label of the calculation. By default, RuNNer Mode 1 calculations
            are stored in a separate folder with the name 'mode1' and output
            files carry the `PREFIX` 'mode1'.

        Raises
        ------
        DatasetUndefiniedError : exception
            Raised if neither `self.dataset` nor `self.atoms` have been defined.
            This would mean that no structure data is available at all.

        """
        # Set the correct calculation label.
        self.label = label

        # RuNNer Mode 1 can either be called for a single ASE Atoms object to
        # which the calculator has been attached (`self.atoms`) or for the
        # whole dataset (`self.dataset`).
        # `dataset` takes precedence over the attached `atoms` object.
        atoms = self.dataset or self.atoms

        # If neither `self.dataset` nor `self.atoms` has been defined yet, raise.
        if atoms is None:
            raise DatasetUndefiniedError('Please either set a dataset for this '
                + 'calculator or attach it to an `Atoms` object.')

        # Make sure that we run a Mode 1 calculation.
        self.set(runner_mode=1)

        # Start the calculation by calling the `get_property` method of the
        # parent class. We here ask for the symmetry function values, knowing
        # that this will also return the splitting between training and testing
        # set.
        self.get_property('sfvalues', atoms=atoms)

    def mode2(self, label='mode2/mode2'):
        """Train a neural network potential, i.e. RuNNer Mode 2.

        This method runs the second mode of RuNNer, in which a neural network
        potential is trained. Three results are obtained:
            * Metrics of the training process performance, i.e. the fit results.
            * The weights of the atomic neural networks.
            * The symmetry function scaling data.

        Optional Parameters
        -------------------
        label : string, _default_='mode2/mode2'
            The label of the calculation. By default, RuNNer Mode 2 calculations
            are stored in a separate folder with the name 'mode2' and output
            files carry the `PREFIX` 'mode2'.

        Raises
        ------
        DatasetUndefiniedError : exception
            Raised if neither `self.dataset` nor `self.atoms` have been defined.
            This would mean that no structure data is available at all.

        """
        # Set the correct calculation label.
        self.label = label

        # Mode 2 always needs a dataset of structure to run successfully.
        atoms = self.dataset

        if atoms is None:
            raise DatasetUndefiniedError('Please either set a dataset for this'
                                       + ' calculator to train a neural network'
                                       + ' potential.')

        # Make sure that we run a Mode 2 calculation.
        self.set(runner_mode=2)

        # Start the calculation by calling the `get_property` method of the
        # parent class. We here ask for the fit results, knowing that this will
        # also return the weights and scaling data.
        self.get_property('fit', atoms=atoms)

    def read(self, label=None):
        """Read atoms, parameters and calculated properties from output file(s).

        This extension of the `FileIOCalculator.read()` routine reads
        calculation results within the `self.label` directory.

        Optional Parameters
        -------------------
        label : string, _default_=`None`
            The label of the calculation.

        Sets
        ----
        atoms: Atoms object
            The state of the atoms from last calculation.
        parameters: Parameters object
            The parameter dictionary.
        results: dict
            Calculated properties like energy and forces.

        """
        # Call the method of the parent class, which will handle the correct
        # treatment of the `label`.
        FileIOCalculator.read(self, label)

        # Read in the dataset, the parameters and the results.
        self.dataset, self.parameters = io.read_runnerase(self.label)
        self.read_results()

    def read_results(self):
        """Read calculation results and store them as class properties."""
        # If successful, RuNNer Mode 1 returns symmetry function values for each
        # structure and the information, which structure belongs to the training
        # and which to the testing set.
        if self.parameters.runner_mode == 1:
            sfvalues, splittraintest = io.read_results_mode1(self.label,
                                                             self._directory)
            self.set_sfvalues(sfvalues)
            self.set_splittraintest(splittraintest)

        # If successful, RuNNer Mode 2 returns the weights of the atomic neural
        # networks, the symmetry function scaling data, and the results of the
        # fitting process.
        if self.parameters.runner_mode == 2:
            fitresults, weights, scaling = io.read_results_mode2(self.label,
                                                                 self._directory)
            self.set_fitresults(fitresults)
            self.set_weights(weights)
            self.set_scaling(scaling)

        # If successful, RuNNer Mode 3 returns the energy and forces of the
        # structure for which it was executed.
        if self.parameters.runner_mode == 3:
            prediction, energy, forces = io.read_results_mode3(self.label,
                                                               self._directory)

            # For just one structure, flatten the energy and force arrays.
            if energy.shape[0] == 1:
                energy = float(energy[0])

            if forces.shape[0] == 1:
                forces = forces[0, :, :]

            self.set_prediction(prediction)
            self.set_energy(energy)
            self.set_forces(forces)

    def write_input(self, atoms, properties, system_changes):
        """Write relevant RuNNer input file(s) to the calculation directory.

        Parameters
        ----------
        atoms : ASE Atoms object or list of ASE Atoms objects
            A single structure or a list of structures for which the symmetry
            functions shall be calculated (= RuNNer Mode 1), for which atomic
            properties like energies and forces will be calculated (= RuNNer
            Mode 3) or on which a neural network potential will be trained
            (= RuNNer Mode 2).
        properties : list of strings
            The target properties which shall be returned. See
            `implemented_properties` for a list of options.
        system_changes : FIXME

        """
        # Per default, `io.write_all_inputs` will only write the input.data and
        # input.nn files (= all that is needed for RuNNer Mode 1).
        # Therefore, all other possible input data is set to `None`.
        scaling = None
        weights = None
        splittraintest = None
        sfvalues = None

        # RuNNer Mode 2 additionally requires symmetry function values and the
        # information, which structure within input.data belongs to the training
        # and which to the testing set.
        if self.parameters.runner_mode == 2:
            sfvalues = self.sfvalues
            splittraintest = self.splittraintest

        # RuNNer Mode 3 requires the symmetry function scaling data and the
        # neural network weights which were obtained as the results of
        # RuNNer Mode 2.
        elif self.parameters.runner_mode == 3:
            scaling = self.scaling
            weights = self.weights

        # Call the method from the parent function, so that directories are
        # created automatically.
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # Write the relevant files to the calculation directory.
        io.write_all_inputs(atoms, properties, parameters=self.parameters,
                            label=self.label,
                            v8_legacy_format=self.v8_legacy_format,
                            scaling=scaling, weights=weights,
                            splittraintest=splittraintest, sfvalues=sfvalues,
                            directory=self._directory)

    def check_state(self, atoms_new):
        """Check for any changes since the last calculation."""
        if isinstance(atoms_new, list):
            system_changes = []
            for idx, at in enumerate(atoms_new):
                system_changes.append(compare_atoms(at, self.dataset[idx]))

        else:
            system_changes = compare_atoms(self.atoms, atoms_new)

        return [change for structure in system_changes for change in structure]
