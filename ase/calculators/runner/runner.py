"""Implementation of a calculator for machine-learning with RuNNer.

The RuNNer Neural Network Energy Representation is a framework for the
construction of high-dimensional neural network potentials developed in the
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

Provides
--------
get_elements : utility function
    Get a list of all elements in a list of images.

Runner : FileIOCalculator
    The main calculator for training and evaluating HDNNPs with RuNNer.

Reference
---------
* [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
* Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Union, Optional, Dict, List, Any

import os

import numpy as np

import ase.io.runner.runner as io

from ase.atoms import Atoms
from ase.data import atomic_numbers

from ase.calculators.calculator import (FileIOCalculator,
                                        compare_atoms,
                                        CalculatorSetupError)

from ase.io.runner.storageclasses import (RunnerSymmetryFunctionValues,
                                          RunnerSplitTrainTest,
                                          RunnerWeights,
                                          RunnerScaling,
                                          SymmetryFunctionSet)

from ase.io.runner.defaultoptions import DEFAULT_PARAMETERS


def get_elements(images: List[Atoms]) -> List[str]:
    """Extract a list of elements from a given list of ASE Atoms objects.

    Parameters
    ----------
    images : List[Atoms]
        A list of ASE atoms objects.

    Returns
    -------
    elements : List[str]
        A list of all elements contained in `images`.

    """
    # Get the chemical symbol of all elements.
    elements: List[str] = []
    for atoms in images:
        for element in atoms.get_chemical_symbols():
            elements.append(element)

    # Remove repeated elements.
    elements = list(set(elements))

    # Sort the elements by atomic number.
    elements.sort(key=lambda i: atomic_numbers[i])
    return elements


# RuNNer class inherits from FileIOCalculator and therefore requires a lot
# of instance attributes and ancestors.
# pylint: disable=[too-many-ancestors, too-many-instance-attributes]
class Runner(FileIOCalculator):
    """Class for training and evaluating neural network potentials with RuNNer.

    The RuNNer Neural Network Energy Representation is a framework for the
    construction of high-dimensional neural network potentials developed in the
    group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.

    The default parameters are mostly those of the RuNNer Fortran code.
    There are, however, a few exceptions. This is supposed to facilitate
    typical application cases when using RuNNer via ASE. These changes are
    documented in `DEFAULT_PARAMETERS` (in `ase.io.runner.defaultoptions`).

    RuNNer operates in three different modes:
        - Mode 1: Calculation of symmetry function values. Symmetry functions
                  are many-body descriptors for the chemical environment of an
                  atom.
        - Mode 2: Fitting of the potential energy surface.
        - Mode 3: Prediction. Use the previously generated high-dimensional
                  potential energy surface to predict the energy and force
                  of unknown chemical configurations.

    The different modes generate a lot of output:
        - Mode 1:
            - sfvalues:       The values of the symmetry functions for each
                              atom.
            - splittraintest: which structures belong to the training and which
                              to the testing set. ASE needs this
                              information to generate the relevant input files
                              for RuNNer Mode 2.
        - Mode 2:
            - fitresults:     The performance of the training process.
            - weights:        The neural network weights.
            - scaling:        The symmetry function scaling factors.

        - Mode 3:
            - energy
            - forces

    """

    implemented_properties = ['energy', 'forces', 'charges',
                              'sfvalues', 'splittraintest',
                              'fitresults', 'weights', 'scaling']
    command = 'RuNNer.x > PREFIX.out'

    discard_results_on_any_change = True

    default_parameters = DEFAULT_PARAMETERS

    # Explicit is better than implicit. By passing all arguments explicitely,
    # the argument types can be narrowly specified.
    # pylint: disable=too-many-arguments
    def __init__(
        self,
        restart: Optional[str] = None,
        ignore_bad_restart_file: object = FileIOCalculator._deprecated,
        label: Optional[str] = None,
        directory: Optional[str] = '.',
        atoms: Optional[Atoms] = None,
        dataset: Optional[List[Atoms]] = None,
        weights: Optional[RunnerWeights] = None,
        scaling: Optional[RunnerScaling] = None,
        sfvalues: Optional[RunnerSymmetryFunctionValues] = None,
        splittraintest: Optional[RunnerSplitTrainTest] = None,
        **kwargs: Dict[Any, Any]
    ) -> None:
        """Construct RuNNer-calculator object.

        Parameters
        ----------
        restart : str, optional, _default_ `None`
            Directory and label of an existing calculation for restarting.
        ignore_bad_restart_file: deprecated
        label : str, optional, _default_ `None`
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'runner'.
        directory : str, optional, _default_ `None`
            Path to the calculation directory.
        atoms : Atoms
            Atoms object to be attached to this calculator.
        dataset : List[Atoms]
            List of ASE Atoms objects used for training the neural network
            potential. Mandatory for RuNNer Mode 2.
        weights : RunnerWeights
            Weights and bias values of atomic neural networks.
        scaling : RunnerScaling
            Symmetry function scaling data.
        sfvalues : RunnerSymmetryFunctionValues
            Symmetry function values.
        splittraintest : RunnerSplitTrainTest
            The assignment of structures in `dataset` to a train and test set.
        kwargs : dict
            Arbitrary key-value pairs can be passed to this class upon
            initialization. They are passed on to the base class. Useful for
            passing RuNNer options to the `parameters` dictionary.

        Examples
        --------
        Run Mode 1 from scratch with existing input.nn and input.data files.

        >>> dataset = read('input.data', ':', format='runnerdata')
        >>> options = read_runnerconfig('input.nn')
        >>> RO = Runner(dataset=dataset, **options)
        >>> RO.run(mode=1)

        Restart Mode 1:

        >>> RO = Runner(restart='mode1/mode1')
        >>> print(RO.results['sfvalues'])

        Run Mode 2:

        >>> RO = Runner(restart='mode1/mode1')
        >>> RO.run(mode=2)
        >>> print(RO.results['fitresults'])

        Update some input parameters:

        >>> RO.set(epochs=20)

        Restart Mode 2 and run Mode 3:

        >>> RO = Runner(restart='mode2/mode2')
        >>> RO.run(mode=3)

        Run Mode 3 with self-defined weights and scaling data:

        >>> RO = Runner(
        >>>    scaling=scaling,
        >>>    weights=weights,
        >>>    dataset=dataset,
        >>>    **options
        >>> )
        >>> RO.atoms = Bulk('C')
        >>> RO.get_potential_energy()
        """
        # Store optional input parameters as class properties.
        self.dataset = dataset
        self.weights = weights
        self.scaling = scaling
        self.sfvalues = sfvalues
        self.splittraintest = splittraintest

        # Initialize the base class.
        FileIOCalculator.__init__(
            self,
            restart=restart,
            ignore_bad_restart_file=ignore_bad_restart_file,
            label=label,
            atoms=atoms,
            directory=directory,
            **kwargs)

    @property
    def elements(self) -> List[str]:
        """Show the elements pertaining to this calculator.

        Store chemical symbols of elements in the parameters dictionary.

        This routine returns (in decreasing order of relevance):
            1. the elements given in `self.parameters['elements']`,
            2. all elements in `self.dataset`,
            3. all elements in `self.atoms`

        If `self.parameters['elements']` was previously not set, it will be
        `set` with the information extracted from `self.dataset` or
        `self.atoms`.
        This means that the calculator will automatically grep elements from
        the attached dataset upon starting a calculation, because it will
        eventually try to access `self.elements`.

        Raises
        ------
        CalculatorSetupError : Exception
            Raised if neither a `dataset` nor an `Atoms` object have been
            defined and `self.parameters['elements']` is empty.

        Returns
        ---------
        elements : List[str]
            Elements can either be set directly from a list of strings, or
            they are automatically extracted from the attached dataset or Atoms
            objects on the calculator.

        """
        elements = self.parameters['elements']

        if elements is None:
            self.set_elements()
            elements = self.parameters['elements']

        return elements

    @elements.setter
    def elements(self, elements: List[str]) -> None:
        """Store chemical symbols of elements in the parameters dictionary.

        This routine adjusts the RuNNer keywords 'elements'
        and 'number_of_elements' based on the provided `elements`.

        Arguments
        ---------
        elements : list of str
            The list of elements which will be stored.

        """
        self.parameters['elements'] = elements
        self.parameters['number_of_elements'] = len(elements)

    def set_elements(self, elements: Optional[List[str]] = None) -> None:
        """Set the `elements` attribute based on `dataset` or `atoms`.

        This routine automatically determined the elements of this calculator
        based on either the attached dataset or the attached Atoms object.
        One may optionally supply a list of elements that will be set.

        Arguments
        ---------
        elements : list[str], optional, _default_ `None`
            The list of elements which will be stored.

        """
        # If no elements were specified, set them automatically based on the
        # attached dataset or Atoms object.
        if elements is None:
            images = self.dataset or [self.atoms]

            # Elements will either be extracted from the training dataset or
            # from the `Atoms` object to which the calculator has been attached.
            if images is None:
                raise CalculatorSetupError('Please specify a custom list of '
                    + 'elements, or attach a dataset or Atoms object to the '
                    + 'calculator.')

            elements = get_elements(images)

        self.elements = elements

    @property
    def symmetryfunctions(self) -> SymmetryFunctionSet:
        """Show the specified short-range symmetry function parameters.

        Returns
        -------
        symfunction_short : SymmetryFunctionSet
            The collection of symmetry functions stored under the RuNNer
            parameter 'symfunction_short'.

        """
        return self.parameters['symfunction_short']

    @symmetryfunctions.setter
    def symmetryfunctions(
        self,
        sfset: SymmetryFunctionSet
    ) -> None:
        """Add symmetry functions to the currently stored ones.

        This routine add either a single symmetry function or a whole set of
        symmetry functions to the storage container under
        `self.parameters['symfunction_short'].`

        """
        self.parameters['symfunction_short'] = sfset

    def set(self, **kwargs) -> Dict[str, object]:
        """Update `self.parameters` with `kwargs`.

        Adds the ability to do keyword validation before calling the base
        class (FileIOCalculator) routine. For this purpose, the argument
        `validate` will be popped from `kwargs`.
        **This means that `validate` can never be set as a RuNNer keyword!**

        Parameters
        ----------
        kwargs : dict
            A dictionary of options that will be added to `self.parameters`.
        """
        # Catch invalid keywords.
        validate: Optional[bool] = kwargs.pop('validate', None)
        if isinstance(kwargs, dict) and validate is True:
            io.check_valid_keywords(kwargs)

        changed_parameters: Dict[str, object] = super().set(**kwargs)

        return changed_parameters

    def run(
        self,
        mode: Optional[int] = None,
        label: Optional[str] = None
    ) -> None:
        """Execute RuNNer Mode 1, 2, or 3.

        Parameters
        -------------------
        mode : int, optional, _default_ None
            The RuNNer mode that will be executed. If not given,
            `self.parameters.runner_mode` will be evaluated.
        label : string, optional, _default_ `None`
            The label of the calculation. By default, RuNNer Mode X calculations
            are stored in a separate folder with the name 'modeX' and output
            files carry the `PREFIX` 'modeX'.

        Raises
        ------
        CalculatorSetupError : exception
            Raised if neither `self.dataset` nor `self.atoms` have been defined.
            This would mean that no structure data is available at all.

        """
        # If not given, use the mode stored in the `parameters`. Otherwise,
        # update parameters with the requested `mode`.
        if mode is None:
            mode = self.parameters['runner_mode']
        else:
            self.set(runner_mode=mode)

        # Set the correct calculation label.
        if label is None:
            label = os.path.join(self.directory, f'mode{mode}', f'mode{mode}')

        self.label = label

        # Set the correct elements of the system.
        self.set_elements()

        # RuNNer can either be called for a single ASE Atoms object to
        # which the calculator has been attached (`self.atoms`) or for the
        # whole dataset (`self.dataset`).
        # `dataset` takes precedence over the attached `atoms` object.
        atoms = self.dataset or self.atoms

        # If neither `self.dataset` nor `self.atoms` has been defined yet,
        # raise an error.
        if atoms is None:
            raise CalculatorSetupError('Please attach a training dataset '
                                       'or an Atoms object to this calculator.')

        # If no seed was specified yet, choose a random value.
        if 'random_seed' not in self.parameters.keys():
            self.set(random_seed=np.random.randint(1, 1000))

        # Start the calculation by calling the `get_property` method of the
        # parent class. Each mode will return more than one possible result,
        # but specifying one is sufficient at this point.
        properties = ['sfvalues', 'fitresults', 'energy']
        self.get_property(properties[mode - 1], atoms=atoms)

    def read(self, label: Optional[str] = None) -> None:
        """Read atoms, parameters and calculated properties from output file(s).

        Parameters
        ----------
        label : string, optional, _default_ `None`
            The label of the calculation whose output will be parsed.

        """
        if label is None:
            label = self.label

        if label is None:
            raise CalculatorSetupError('Please provide a valid label.')

        # Call the method of the parent class, which will handle the correct
        # treatment of the `label`.
        FileIOCalculator.read(self, label)

        # Read in the dataset, the parameters and the results.
        structures, self.parameters = io.read_runnerase(label)
        if isinstance(structures, list):
            self.dataset = structures
        else:
            self.atoms = structures

        self.read_results()

    def read_results(self) -> None:
        """Read calculation results and store them on the calculator."""
        # Call an IO function to read all results expected in this mode.
        # results will be a dictionary.
        mode = self.parameters['runner_mode']
        if mode == 1:
            results = io.read_results_mode1(self.label, self._directory)
        elif mode == 2:
            results = io.read_results_mode2(self.label, self._directory)
        elif mode == 3:
            results = io.read_results_mode3(self._directory)

        # Add the results to the 'results' dictionary of the calculator.
        self.results.update(results)

        # Store the results as instance properties so they may serve as input
        # parameters in future runs.
        # This also keeps them from being overwritten when changes are detected
        # on the keywords.
        for key, value in results.items():
            self.__dict__[f'{key}'] = value

    def write_input(
        self,
        atoms: Union[Atoms, List[Atoms]],
        properties: Optional[List[str]] = None,
        system_changes: Optional[List[str]] = None
    ) -> None:
        """Write relevant RuNNer input file(s) to the calculation directory.

        Parameters
        ----------
        atoms : Atoms or List[Atoms]
            A single structure or a list of structures for which the symmetry
            functions shall be calculated (= RuNNer Mode 1), for which atomic
            properties like energies and forces will be calculated (= RuNNer
            Mode 3) or on which a neural network potential will be trained
            (= RuNNer Mode 2).
        properties : List[str]
            The target properties which shall be returned. See
            `implemented_properties` for a list of options.
        system_changes : List[str]
            A list of changes in `atoms` in comparison to the previous run.
        """
        # Per default, `io.write_all_inputs` will only write the input.data and
        # input.nn files (= all that is needed for RuNNer Mode 1).
        # Therefore, all other possible input options are set to `None`.
        scaling = None
        weights = None
        splittraintest = None
        sfvalues = None

        # RuNNer Mode 2 additionally requires symmetry function values and the
        # information, which structure within input.data belongs to the training
        # and which to the testing set.
        if self.parameters['runner_mode'] == 2:
            sfvalues = self.sfvalues or self.results['sfvalues']
            splittraintest = self.splittraintest or self.results['splittraintest']
            scaling = self.scaling
            weights = self.weights

        # RuNNer Mode 3 requires the symmetry function scaling data and the
        # neural network weights which were obtained as the results of
        # RuNNer Mode 2.
        elif self.parameters['runner_mode'] == 3:
            scaling = self.scaling or self.results['scaling']
            weights = self.weights or self.results['weights']

        # Call the method from the parent function, so that directories are
        # created automatically.
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        # Write the relevant files to the calculation directory.
        io.write_all_inputs(atoms, parameters=self.parameters, label=self.label,
                            scaling=scaling, weights=weights,
                            splittraintest=splittraintest, sfvalues=sfvalues,
                            directory=self._directory)

    def check_state(
        self,
        atoms: Union[Atoms, List[Atoms]],
        tol: float = 1e-15
    ) -> List[str]:
        """Check for any changes since the last calculation.

        Check whether any of the parameters in atoms differs from those in
        `self.atoms`. Overrides the `Calculator` routine to extend the
        functionality to List[Atoms], i.e. RuNNer datasets.

        Parameters
        ----------
        atoms : Atoms or List[Atoms]
            A RuNNer dataset or an ASE Atoms object which will be compared
            to the calculator storage.
        tol : float, _default_ 1e-15
            The tolerance for float comparisons.

        """
        # If more than one Atoms object is passed, check_state for each `Atoms`.
        if isinstance(atoms, list):

            # If no dataset has been defined yet, no changes will be found.
            if self.dataset is None:
                return []

            system_changes = []
            for idx, structure in enumerate(atoms):
                structure_changes = compare_atoms(structure, self.dataset[idx],
                                                  tol=tol)
                system_changes.append(structure_changes)

            # Unfold the list of found changes and return.
            return [change for structure in system_changes
                           for change in structure]

        return compare_atoms(self.atoms, atoms, tol=tol)

    def get_forces(
        self,
        atoms: Optional[Atoms] = None
    ) -> np.ndarray:
        """Calculate the atomic forces.

        Overrides the `Calculator` routine to ensure that the `calculate_forces`
        keyword is set and RuNNer is in prediction mode (Mode 3).
        Otherwise, RuNNer does not yield atomic forces.

        Parameters
        ----------
        atoms : Atoms, optional, _default_ `None`
            The Atoms object for which the forces will be calculated.
        Returns
        -------
        stress : np.ndarray[np.float64]
            The atomic stress in a [Nx3] array where N is the number of atoms
            in the system.
        """
        self.parameters['runner_mode'] = 3
        self.parameters['calculate_forces'] = True
        return super().get_forces(atoms)

    def get_stress(
        self,
        atoms: Optional[Atoms] = None
    ) -> np.ndarray:
        """Calculate the atomic stress.

        Overrides the `Calculator` routine to ensure that the `calculate_stress`
        keyword is set and RuNNer is in prediction mode (Mode 3).
        Otherwise, RuNNer does not yield stress values.

        Parameters
        ----------
        atoms : Atoms, optional, _default_ `None`
            The Atoms object for which the stress will be calculated.
        Returns
        -------
        stress : np.ndarray[np.float64]
            The atomic stress in a [Nx3x3] array where N is the number of atoms
            in the system.
        """
        self.parameters['runner_mode'] = 3
        self.parameters['calculate_stress'] = True
        return super().get_stress(atoms)
