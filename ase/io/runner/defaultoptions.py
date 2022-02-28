"""Default dictionaries of RuNNer parameters.

Provides
--------
RunnerOptions : TypedDict
    Type specifications for all dictionaries of RuNNer parameters. This is
    mainly used for `DEFAULT_PARAMETERS` at the moment and NOT a complete list
    of all possible parameters as given in `RUNNERCONFIG_DEFAULTS`.
DEFAULT_PARAMETERS : RunnerOptions
    A selection of those keywords in `RUNNERCONFIG_DEFAULTS` which are
        - either mandatory for RuNNer usage,
        - or very commonly used,
        - or have a default in RuNNer that is rarely seen in applications.
    This dictionary is used when initializing a RuNNer calculator object.
RUNNERDATA_KEYWORDS : List[str]
    A list of the possible allowed keywords in an input.nn file.
RUNNERCONFIG_DEFAULTS : nested dict
    A dictionary of all RuNNer options, complete with
        - type specification
        - verbose description
        - formatting statement
        - RuNNer mode specification
        - arguments with description, type specification, and possible options
        - default value
        - switch whether the parameter can occur multiple times in input.nn

Reference
---------
- [The online documentation of RuNNer](https://theochem.gitlab.io/runner)

Contributors
------------
- Author: [Alexander Knoll](mailto:alexander.knoll@chemie.uni-goettingen.de)

"""

from typing import Optional, Dict, List
from .storageclasses import SymmetryFunctionSet


# Originally inherited from TypedDict, but this was removed for now to retain
# backwards compatibility with Python 3.6 and 3.7.
# class RunnerOptions(TypedDict, total=False)
class RunnerOptions:
    """Type specifications for RuNNer default options."""

    runner_mode: int
    symfunction_short: SymmetryFunctionSet
    elements: Optional[List[str]]
    number_of_elements: int
    bond_threshold: float
    nn_type_short: int
    use_short_nn: bool
    optmode_charge: int
    optmode_short_energy: int
    optmode_short_force: int
    points_in_memory: int
    scale_symmetry_functions: bool
    cutoff_type: int
    # Mode 1.
    test_fraction: float
    # Mode 1 and 2.
    use_short_forces: bool
    # Mode 1 and 3.
    remove_atom_energies: bool
    atom_energy: Dict[str, float]
    # Mode 2.
    epochs: int
    kalman_lambda_short: float
    kalman_nue_short: float
    mix_all_points: bool
    nguyen_widrow_weights_short: bool
    repeated_energy_update: bool
    short_energy_error_threshold: float
    short_energy_fraction: float
    short_force_error_threshold: float
    short_force_fraction: float
    use_old_weights_charge: bool
    use_old_weights_short: bool
    write_weights_epoch: int
    # Mode 2 and 3.
    center_symmetry_functions: bool
    precondition_weights: bool
    global_activation_short: List[str]
    global_hidden_layers_short: int
    global_nodes_short: List[int]
    # Mode 3.
    calculate_forces: bool
    calculate_stress: bool


DEFAULT_PARAMETERS: Dict[str, object] = {
    # General for all modes.
    'runner_mode': 1,                     # Default is starting a new fit.
    # All modes.
    'symfunction_short': SymmetryFunctionSet(),  # Auto-set if net provided.
    'elements': None,                     # Auto-set by `set_atoms()`.
    'number_of_elements': 0,              # Auto-set by `set_atoms()`.
    'bond_threshold': 0.5,                # Default OK but system-dependent.
    'nn_type_short': 1,                   # Most people use atomic NNs.
    #'nnp_gen': 2,                        # 2Gs remain the most common use case.
    'use_short_nn': True,                 # Short-range fitting is the default.
    'optmode_charge': 1,                  # Default OK but option is relevant.
    'optmode_short_energy': 1,            # Default OK but option is relevant.
    'optmode_short_force': 1,             # Default OK but option is relevant.
    'points_in_memory': 1000,             # Default value is legacy.
    'scale_symmetry_functions': True,     # Scaling is used by almost everyone.
    'cutoff_type': 1,                     # Default OK, but important.
    # Mode 1.
    'test_fraction': 0.1,                 # Default too small, more common.
    # Mode 1 and 2.
    'use_short_forces': True,             # Force fitting is standard.
    # Mode 1 and 3.
    'remove_atom_energies': True,         # Standard use case.
    'atom_energy': {},                    # `remove_atom_energies` dependency.
    # Mode 2.
    'epochs': 30,                         # Default is 0, 30 is common.
    'kalman_lambda_short': 0.98000,       # No Default, this is sensible value.
    'kalman_nue_short': 0.99870,          # No Default, this is sensible value.
    'mix_all_points': True,               # Standard option.
    'nguyen_widrow_weights_short': True,  # Typically improves the fit.
    'repeated_energy_update': True,       # Default is False, but usage common.
    'short_energy_error_threshold': 0.1,  # Use only energies > 0.1*RMSE.
    'short_energy_fraction': 1.0,         # All energies are used.
    'short_force_error_threshold': 1.0,   # All forces are used.
    'short_force_fraction': 0.1,          # 10% of the forces are used.
    'use_old_weights_charge': False,      # Relevant for calculation restart.
    'use_old_weights_short': False,       # Relevant for calculation restart.
    'write_weights_epoch': 5,             # Default is 1, very verbose.
    # Mode 2 and 3.
    'center_symmetry_functions': True,    # This is standard procedure.
    'precondition_weights': True,         # This is standard procedure.
    'global_activation_short': ['t', 't', 'l'],  # tanh / linear activ. func.
    'global_hidden_layers_short': 2,      # 2 hidden layers
    'global_nodes_short': [15, 15],       # 15 nodes per hidden layer.
}


RUNNERDATA_KEYWORDS: List[str] = ['begin', 'comment', 'lattice', 'atom',
                                  'charge', 'energy', 'end']


RUNNERCONFIG_DEFAULTS = {
    'analyze_composition': {
        'type': bool,
        'description': r'Print detailed information about the element com'
                       + r'position of the data set in  `input.data`.',
        'format': r'analyze_composition',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': True,
        'allow_multiple': False,
    },
    'analyze_error': {
        'type': bool,
        'description': r'Print detailed information about the training er'
                       + r'ror.',
        'format': r'analyze_error',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'analyze_error_charge_step': {
        'description': r'When a detailed analysis of the training error w'
                       + r'ith analyze_error is performed, this keyword all'
                       + r'ows for the definition of the interval in which '
                       + r' atoms with the same charge error are grouped to'
                       + r'gether.',
        'format': r'analyze_error_charge_step a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The interval in which atoms with the sam'
                               + r'e charge error are grouped together (uni'
                               + r't: electron charge).',
                'type': float,
                'default_value': 0.001,
            },
        },
        'allow_multiple': False,
    },
    'analyze_error_energy_step': {
        'description': r'When a detailed analysis of the training error w'
                       + r'ith analyze_error is performed, this keyword all'
                       + r'ows for the definition of the interval in which '
                       + r' atoms with the same energy error are grouped to'
                       + r'gether.',
        'format': r'analyze_error_energy_step a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The interval in which atoms with the sam'
                               + r'e energy error are grouped together (uni'
                               + r't: Hartree).',
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'analyze_error_force_step': {
        'description': r'When a detailed analysis of the training error w'
                       + r'ith analyze_error is performed, this keyword all'
                       + r'ows for the definition of the interval in which '
                       + r' atoms with the same total force error are group'
                       + r'ed together.',
        'format': r'analyze_error_force_step a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The interval in which atoms with the sam'
                               + r'e force error are grouped together (unit'
                               + r': Hartree per Bohr).',
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'atom_energy': {
        'description': r'Specification of the energies of the free atoms.'
                       + r' This keyword must be used for each element if t'
                       + r'he keyword  remove_atom_energies is used.  In ru'
                       + r'nner_mode 1 the atomic energies are removed from'
                       + r' the total energies, in  runner_mode 3 the atomi'
                       + r'c energies are added to the fitted energy to yie'
                       + r'ld the correct total energy. Internally, `RuNNer'
                       + r'` always works with binding energies, if  remove'
                       + r'_atom_energies is specified.',
        'format': r'atom_energy element energy',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'Element symbol.',
                'type': str,
                'default_value': None,
            },
            'energy': {
                'description': r'Atomic reference energy in Hartree.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': True,
    },
    'biasweights_max': {
        'description': r'`RuNNer` allows for a separate random initializa'
                       + r'tion of the bias weights at the beginning of  ru'
                       + r'nner_mode 2 through separate_bias_ini_short. In '
                       + r'that case the bias weights are randomly initiali'
                       + r'zed on an interval between  biasweights_max  and'
                       + r' biasweights_min. ',
        'format': r'biasweights_max a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The maximum default_value that is assign'
                               + r'ed to bias weights during initialization'
                               + r' of the weights.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'biasweights_min': {
        'description': r'`RuNNer` allows for a separate random initializa'
                       + r'tion of the bias weights at the beginning of  ru'
                       + r'nner_mode 2 through separate_bias_ini_short. In '
                       + r'that case the bias weights are randomly initiali'
                       + r'zed on an interval between  biasweights_max  and'
                       + r' biasweights_min.',
        'format': r'biasweights_min a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The maximum default_value that is assign'
                               + r'ed to bias weights during initialization'
                               + r' of the weights.',
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'bond_threshold': {
        'description': r'Threshold for the shortest bond in the structure'
                       + r' in Bohr units. If a shorter bond occurs `RuNNer'
                       + r'` will stop with an error message in  runner_mod'
                       + r'e 2 and 3. In  runner_mode 1 the  structure will'
                       + r' be eliminated from the data set.',
        'format': r'bond_threshold a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'The minimum bond length between any two '
                               + r'atoms in the structure (unit: Bohr).',
                'type': float,
                'default_value': 0.5,
            },
        },
        'allow_multiple': False,
    },
    'calculate_final_force': {
        'type': bool,
        'description': r'Print detailed information about the forces in t'
                       + r'he training and testing set at  the end of the N'
                       + r'NP training process. ',
        'format': r'calculate_final_force',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'calculate_forces': {
        'type': bool,
        'description': r'Calculate the atomic forces in  runner_mode 3 an'
                       + r'd write them to the files  runner.out nnforces.o'
                       + r'ut',
        'format': r'calculate_forces',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'calculate_hessian': {
        'type': bool,
        'description': r'Calculate the Hessian matrix in  runner_mode 3. '
                       + r' <!-- The implementation is currently in progres'
                       + r's and the keyword is not yet ready for use. -->',
        'format': r'calculate_hessian',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'calculate_stress': {
        'type': bool,
        'description': r'Calculate the stress tensor (only for periodic s'
                       + r'ystems) in  runner_mode 3 and  write it to the f'
                       + r'iles runner.out nnstress.out This is at the mome'
                       + r'nt only implemented for the short range part and'
                       + r' for the contributions to the stress tensor thro'
                       + r'ugh vdW interactions.',
        'format': r'calculate_stress',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'center_symmetry_functions': {
        'type': bool,
        'description': r'Shift the symmetry function default_values indiv'
                       + r'idually for each symmetry function such that the'
                       + r' average is moved to zero. This may have numeric'
                       + r'al advantages, because  zero is the center of th'
                       + r'e non-linear regions of most activation function'
                       + r's.',
        'format': r'center_symmetry_functions',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'charge_error_threshold': {
        'description': r'Threshold default_value for the error of the cha'
                       + r'rges in units of the RMSE of the previous epoch.'
                       + r' A default_value of 0.3 means that only charges '
                       + r'with an error larger than 0.3RMSE will be used f'
                       + r'or the weight update. Large default_values (abou'
                       + r't 1.0) will speed up the first epochs, because o'
                       + r'nly a few points will be used. ',
        'format': r'charge_error_threshold a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of charge RMSE that a charge ne'
                               + r'eds to reach to be used in the weight up'
                               + r'date.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'charge_fraction': {
        'description': r'Defines the random fraction of atomic charges us'
                       + r'ed for fitting the electrostatic weights in  run'
                       + r'ner_mode 2.',
        'format': r'charge_fraction a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of atomic charges used for fitt'
                               + r'ing of the electrostatic weights. 100% ='
                               + r' 1.0.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'charge_group': {
        'description': r'Do not update the electrostatic NN weights after'
                       + r' the presentation of an  individual atomic charg'
                       + r'e, but average the derivatives with respect to t'
                       + r'he  weights over the specified number of charges'
                       + r' for each element.',
        'format': r'charge_group i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Number of atomic charges per group. The '
                               + r'maximum is given by points_in_memory.',
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
    'check_forces': {
        'type': bool,
        'description': r'This keyword allows to check if the sum of all N'
                       + r'N force vectors is zero, It is for debugging pur'
                       + r'poses only, but does not cost much CPU time.',
        'format': r'check_forces',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'check_input_forces': {
        'description': r'Check, if the sum of all forces of the training '
                       + r'structures is sufficiently close to the zero vec'
                       + r'tor.',
        'format': r'check_input_forces a0',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Threshold for the absolute default_value'
                               + r' of the sum of all force vectors per ato'
                               + r'm.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'cutoff_type': {
        'description': r'This keyword determines the cutoff function to b'
                       + r'e used for the symmetry functions.',
        'format': r'cutoff_type i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Threshold for the absolute default_value'
                               + r' of the sum of all force vectors per ato'
                               + r'm.',
                'type': int,
                'default_value': 1,
                'options': {
                    0: {
                        'description': r'Hard function: $1$',
                    },
                    1: {
                        'description': r'Cosine function: $\frac{1}{2}[\c'
                                       + r'os(\pi x)+ 1]$',
                    },
                    2: {
                        'description': r'Hyperbolic tangent function 1: $'
                                       + r'\tanh^{3} (1-\frac{R_{ij}}{R_{\m'
                                       + r'athrm{c}}})$',
                    },
                    3: {
                        'description': r'Hyperbolic tangent function 2: $'
                                       + r'(\frac{e+1}{e-1})^3 \tanh^{3}(1-'
                                       + r'\frac{R_{ij}}{R_{\mathrm{c}}})$',
                    },
                    4: {
                        'description': r'Exponential function: $\exp(1-\f'
                                       + r'rac{1}{1-x^2})$',
                    },
                    5: {
                        'description': r'Polynomial function 1: $(2x -3)x'
                                       + r'^2+1$',
                    },
                    6: {
                        'description': r'Polynomial function 2: $((15-6x)'
                                       + r'x-10)x^3+1$',
                    },
                    8: {
                        'description': r'Polynomial function 3: $(x(x(20x'
                                       + r'-70)+84)-35)x^4+1$',
                    },
                    '8': {
                        'description': r'Polynomial function 4: $(x(x(x(3'
                                       + r'15-70x)-540)+420)-126)x^5+1$',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'data_clustering': {
        'description': r'Performs an analysis of all symmetry function ve'
                       + r'ctors of all atoms and groups the atomic environ'
                       + r'ments to clusters with a maximum distance of  `a'
                       + r'0` between the symmetry function vectors. If  `a'
                       + r'1` is larger than 1.0 the assignment of each ato'
                       + r'm will be  printed.',
        'format': r'data_clustering a0 a1',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Maximum distance between the symmetry fu'
                               + r'nction vectors of two clusters of atomic'
                               + r' environments.',
                'type': float,
                'default_value': 1.0,
            },
            'a1': {
                'description': r'If `a1 > 1.0`, print the group that each'
                               + r' atom has been assigned to.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'debug_mode': {
        'type': bool,
        'description': r'If switched on, this option can produce a lot of'
                       + r' output and is meant for debugging new developme'
                       + r'nts only!!!',
        'format': r'debug_mode',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'detailed_timing': {
        'type': bool,
        'description': r'Write detailed timing information for the indivi'
                       + r'dual parts of `RuNNer` at the end of the run. Th'
                       + r'is feature has to be used with some care because'
                       + r' often the implementation of the time measuremen'
                       + r't lacks behind in code development.',
        'format': r'detailed_timing',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'detailed_timing_epoch': {
        'type': bool,
        'description': r'Write detailed timing information in each epoch '
                       + r'in  runner_mode 2.  This feature has to be used '
                       + r'with some care because often the implementation '
                       + r'of the time measurement lacks behind in code dev'
                       + r'elopment.',
        'format': r'detailed_timing_epoch',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'detect_saturation': {
        'type': bool,
        'description': r'For each training epoch, checks whether the defa'
                       + r'ult_value of a node in any hidden layer  exceeds'
                       + r'  saturation_threshold and prints a warning.',
        'format': r'detect_saturation',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'dynamic_force_grouping': {
        'type': bool,
        'description': r'Do not update the short-range NN weights after t'
                       + r'he presentation of an  individual atomic force v'
                       + r'ector, but average the derivatives with respect '
                       + r'to the  weights over the number of force vectors'
                       + r' for each element specified by short_force_group',
        'format': r'dynamic_force_grouping',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'electrostatic_type': {
        'description': r'This keyword determines the cutoff function to b'
                       + r'e used for the symmetry functions.',
        'format': r'electrostatic_type i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Threshold for the absolute default_value'
                               + r' of the sum of all force vectors per ato'
                               + r'm.',
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': r'There is a separate set of atomi'
                                       + r'c NNs to fit the atomic charges '
                                       + r'as a function of the chemical en'
                                       + r'vironment.',
                    },
                    2: {
                        'description': r'The atomic charges are obtained '
                                       + r'as a second output node of the s'
                                       + r'hort range atomic NNs. **This is'
                                       + r' not yet implemented.**',
                    },
                    3: {
                        'description': r'Element-specific fixed charges a'
                                       + r're used that are specified in th'
                                       + r'e input.nn file by the keyword f'
                                       + r'ixed_charge.',
                    },
                    4: {
                        'description': r'The charges are fixed but can be'
                                       + r' different for each atom in the '
                                       + r'system. They are specified in th'
                                       + r'e file `charges.in`.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'element_activation_electrostatic': {
        'description': r'Set the activation function for a specified node'
                       + r' of a specified element in the electrostatic NN.'
                       + r' The default is set by the keyword global_activa'
                       + r'tion_electrostatic.',
        'format': r'element_activation_electrostatic element layer node type',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r' whose atomic NN the activation function'
                               + r' shall be applied to.',
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': r'The number of the layer of the target no'
                               + r'de.',
                'type': int,
                'default_value': None,
            },
            'node': {
                'description': r'The number of the target node in layer `'
                               + r'layer`.',
                'type': int,
                'default_value': None,
            },
            'type': {
                'description': r'The kind of activation function. Options'
                               + r' are listed under global_activation_shor'
                               + r't.',
                'type': str,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_activation_pair': {
        'description': r'Set the activation function for a specified node'
                       + r' of a specified element pair in  the pairwise NN'
                       + r'. The default is set by the keyword global_activ'
                       + r'ation_pair.',
        'format': r'element_activation_pair element element layer node type',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the second '
                               + r'element in the pair whose short-range pa'
                               + r'ir NN the activation function shall be a'
                               + r'pplied to.',
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': r'The number of the layer of the target no'
                               + r'de.',
                'type': int,
                'default_value': None,
            },
            'node': {
                'description': r'The number of the target node in layer `'
                               + r'layer`.',
                'type': int,
                'default_value': None,
            },
            'type': {
                'description': r'The kind of activation function. Options'
                               + r' are listed under global_activation_shor'
                               + r't.',
                'type': str,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_activation_short': {
        'description': r'Set the activation function for a specified node'
                       + r' of a specified element in the short range NN. T'
                       + r'he default is set by the keyword global_activati'
                       + r'on_short.',
        'format': r'element_activation_short element layer node type',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r' whose atomic NN the activation function'
                               + r' shall be applied to.',
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': r'The number of the layer of the target no'
                               + r'de.',
                'type': int,
                'default_value': None,
            },
            'node': {
                'description': r'The number of the target node in layer `'
                               + r'layer`.',
                'type': int,
                'default_value': None,
            },
            'type': {
                'description': r'The kind of activation function. Options'
                               + r' are listed under global_activation_shor'
                               + r't.',
                'type': str,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_decoupled_forces_v2': {
        'type': bool,
        'description': r'This is a more sophisticated version of the elem'
                       + r'ent decoupled Kalman filter for force fitting (s'
                       + r'witched on by the keyword element_decoupled_kalm'
                       + r'an.',
        'format': r'element_decoupled_forces_v2',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'element_decoupled_kalman': {
        'type': bool,
        'description': r'Use the element decoupled Kalman filter for the '
                       + r'short range energy and force update (if use_shor'
                       + r't_forces is switched on). This is implemented on'
                       + r'ly for the atomic energy case. A more sophistica'
                       + r'ted algorithm for the force fitting can be activ'
                       + r'ated by using additionally the keyword  element_'
                       + r'decoupled_forces_v2.  One important parameter fo'
                       + r'r force fitting is  force_update_scaling,  which'
                       + r' determines the magnitude of the force update co'
                       + r'mpared to the energy update. Usually 1.0 is a go'
                       + r'od default default_value.',
        'format': r'element_decoupled_kalman',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'element_hidden_layers_electrostatic': {
        'description': r'Overwrite the global default number of hidden la'
                       + r'yers given by global_hidden_layers_electrostatic'
                       + r' for a specific element. Just a reduction of the'
                       + r' number of hidden layers is possible. .',
        'format': r'element_hidden_layers_electrostatic element layers',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r' whose atomic NN hidden layer number wil'
                               + r'l be set.',
                'type': str,
                'default_value': None,
            },
            'layers': {
                'description': r'The number of hidden layers for this ele'
                               + r'ment.',
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_hidden_layers_pair': {
        'description': r'Overwrite the global default number of hidden la'
                       + r'yers given by global_hidden_layers_pair for a sp'
                       + r'ecific element. Just a reduction of the number o'
                       + r'f hidden layers is possible. .',
        'format': r'element_hidden_layers_pair element element layers',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element1': {
                'description': r'The periodic table symbol of the first p'
                               + r'air element whose short-range pair NN hi'
                               + r'dden layer number will be set.',
                'type': str,
                'default_value': None,
            },
            'element2': {
                'description': r'The periodic table symbol of the second '
                               + r'pair element whose short-range pair NN h'
                               + r'idden layer number will be set.',
                'type': str,
                'default_value': None,
            },
            'layers': {
                'description': r'The number of hidden layers for this ele'
                               + r'ment pair.',
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_hidden_layers_short': {
        'description': r'Overwrite the global default number of hidden la'
                       + r'yers given by global_hidden_layers_short for a s'
                       + r'pecific element. Just a reduction of the number '
                       + r'of hidden layers is possible.',
        'format': r'element_hidden_layers_short element layers',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r' whose atomic NN hidden layer number wil'
                               + r'l be set.',
                'type': str,
                'default_value': None,
            },
            'layers': {
                'description': r'The number of hidden layers for this ele'
                               + r'ment.',
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_nodes_electrostatic': {
        'description': r'Overwrite the global default number of nodes in '
                       + r'the specified hidden layer of an elecrostatic NN'
                       + r' given by  global_nodes_electrostatic  for a spe'
                       + r'cific element.  Just a reduction of the number o'
                       + r'f nodes is possible. ',
        'format': r'element_nodes_electrostatic element layer i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r'.',
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': r'The number of the hidden layer for which'
                               + r' the number of nodes is set.',
                'type': int,
                'default_value': None,
            },
            'i0': {
                'description': r'The number of nodes to be set.',
                'type': int,
                'default_value': r'global_nodes_electrostatic',
            },
        },
        'allow_multiple': True,
    },
    'element_nodes_pair': {
        'description': r'Overwrite the global default number of nodes in '
                       + r'the specified hidden layer of a pair NN given by'
                       + r'  global_nodes_pair  for a specific element.  Ju'
                       + r'st a reduction of the number of nodes is possibl'
                       + r'e. ',
        'format': r'element_nodes_pair element element layer i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the second '
                               + r'pair element.',
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': r'The number of the hidden layer for which'
                               + r' the number of nodes is set.',
                'type': int,
                'default_value': None,
            },
            'i0': {
                'description': r'The number of nodes to be set.',
                'type': int,
                'default_value': r'global_nodes_pair',
            },
        },
        'allow_multiple': True,
    },
    'element_nodes_short': {
        'description': r'Overwrite the global default number of nodes in '
                       + r'the specified hidden layer of an short-range ato'
                       + r'mic NN given by  global_nodes_short  for a speci'
                       + r'fic element.  Just a reduction of the number of '
                       + r'nodes is possible. ',
        'format': r'element_nodes_short element layer i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r'.',
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': r'The number of the hidden layer for which'
                               + r' the number of nodes is set.',
                'type': int,
                'default_value': None,
            },
            'i0': {
                'description': r'The number of nodes to be set.',
                'type': int,
                'default_value': r'global_nodes_short',
            },
        },
        'allow_multiple': True,
    },
    'element_pairsymfunction_short': {
        'description': r'Set the symmetry functions for one element pair '
                       + r'for the short-range pair NN.',
        'format': r'element_pairsymfunction_short element element type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element1': {
                'description': r'The periodic table symbol of the first p'
                               + r'air element.',
                'type': str,
                'default_value': None,
            },
            'element2': {
                'description': r'The periodic table symbol of the second '
                               + r'pair element.',
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires parame'
                                       + r'ters `eta` and `rshift`. ```runn'
                                       + r'er-config element_pairsymfunctio'
                                       + r'n_short element element 2 eta rs'
                                       + r'hift cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, `lambda`, and `ze'
                                       + r'ta`. ```runner-config element_pa'
                                       + r'irsymfunction_short element elem'
                                       + r'ent 3 eta lambda zeta cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'lambda': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'zeta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires parame'
                                       + r'ter `eta`. ```runner-config elem'
                                       + r'ent_pairsymfunction_short elemen'
                                       + r't element 4 eta cutoff ```',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig element_pairsymfunction_short'
                                       + r' element element 5 eta ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config element_pairsymfunction'
                                       + r'_short element element 6 cutoff '
                                       + r'```',
                    },
                    8: {
                        'description': r'Not implemented.',
                    },
                    '8': {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, and `rshift`. ```'
                                       + r'runner-config element_pairsymfun'
                                       + r'ction_short element element 8 et'
                                       + r'a rshift cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`. ```runner-config '
                                       + r'element_pairsymfunction_short el'
                                       + r'ement element 9 eta cutoff ```.',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'elements': {
        'description': r'The element symbols of all elements in the syste'
                       + r'm in arbitrary order.  The number of specified e'
                       + r'lements must fit to the default_value of the key'
                       + r'word number_of_elements.',
        'format': r'elements element [element...]',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element [element...]': {
                'description': r'The periodic table symbols of all the el'
                               + r'ement in the system.',
                'type': str,
                'default_value': [None],
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'element_symfunction_electrostatic': {
        'description': r'Set the symmetry functions for one element with '
                       + r'all possible neighbor element combinations for t'
                       + r'he electrostatics NN. The variables are the same'
                       + r' as for the  keyword  global_symfunction_electro'
                       + r'static and are explained in more detail there.',
        'format': r'element_symfunction_electrostatic element type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r'.',
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires parame'
                                       + r'ters `eta` and `rshift`. ```runn'
                                       + r'er-config element_symfunction_el'
                                       + r'ectrostatic element 2 eta rshift'
                                       + r' cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, `lambda`, and `ze'
                                       + r'ta`. ```runner-config element_sy'
                                       + r'mfunction_electrostatic element '
                                       + r'3 eta lambda zeta cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'lambda': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'zeta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires parame'
                                       + r'ter `eta`. ```runner-config elem'
                                       + r'ent_symfunction_electrostatic el'
                                       + r'ement 4 eta cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig element_symfunction_electrost'
                                       + r'atic element 5 eta ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config element_pairsymfunction'
                                       + r'_short element 6 cutoff ```',
                    },
                    8: {
                        'description': r'Not implemented.',
                    },
                    '8': {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, and `rshift`. ```'
                                       + r'runner-config element_pairsymfun'
                                       + r'ction_short element 8 eta rshift'
                                       + r' cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`. ```runner-config '
                                       + r'element_pairsymfunction_short el'
                                       + r'ement 9 eta cutoff ```.',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_symfunction_short': {
        'description': r'Set the symmetry functions for one element with '
                       + r'all possible neighbor element combinations for t'
                       + r'he short-range NN. The variables are the same as'
                       + r' for the  keyword  global_symfunction_short and '
                       + r'are explained in more detail there.',
        'format': r'element_symfunction_short element type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r'.',
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires parame'
                                       + r'ters `eta` and `rshift`. ```runn'
                                       + r'er-config element_symfunction_sh'
                                       + r'ort element 2 eta rshift cutoff '
                                       + r'```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, `lambda`, and `ze'
                                       + r'ta`. ```runner-config element_sy'
                                       + r'mfunction_short element 3 eta la'
                                       + r'mbda zeta cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'lambda': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'zeta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires parame'
                                       + r'ter `eta`. ```runner-config elem'
                                       + r'ent_symfunction_short element 4 '
                                       + r'eta cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig element_symfunction_short ele'
                                       + r'ment 5 eta ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config element_pairsymfunction'
                                       + r'_short element 6 cutoff ```',
                    },
                    8: {
                        'description': r'Not implemented.',
                    },
                    '8': {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, and `rshift`. ```'
                                       + r'runner-config element_pairsymfun'
                                       + r'ction_short element 8 eta rshift'
                                       + r' cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`. ```runner-config '
                                       + r'element_pairsymfunction_short el'
                                       + r'ement 9 eta cutoff ```.',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'enable_on_the_fly_input': {
        'type': bool,
        'description': r'Read modified input.nn the fitting procedure fro'
                       + r'm a file labeled `input.otf`.',
        'format': r'enable_on_the_fly_input',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'energy_threshold': {
        'description': r'Set an energy threshold for fitting data. This k'
                       + r'eyword is only used in  runner_mode 1 for the de'
                       + r'cision if a point should be used in the training'
                       + r' or  test set or if it should be eliminated beca'
                       + r'use of its high energy.',
        'format': r'energy_threshold a0',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Threshold for the total energy of a stru'
                               + r'cture (unit: Hartree per atom).',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'enforce_max_num_neighbors_atomic': {
        'description': r'Set an upper threshold for the number of neighbo'
                       + r'rs an atom can have.',
        'format': r'enforce_max_num_neighbors_atomic i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Maximum number of neighbors for one atom'
                               + r'.',
                'type': int,
                'default_value': r'None',
            },
        },
        'allow_multiple': False,
    },
    'enforce_totcharge': {
        'description': r'Rescale the NN atomic charges to get a neutral s'
                       + r'ystem. An overall neutral system is required for'
                       + r' a correct calculation of the Ewald sum for peri'
                       + r'odic systems.  The additional error introduced b'
                       + r'y rescaling the NN charges is typically much sma'
                       + r'ller than the fitting error, but this should be '
                       + r'checked.',
        'format': r'enforce_totcharge i0',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Switch charge rescaling on (`1`) or off '
                               + r'(`0`).',
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'environment_analysis': {
        'type': bool,
        'description': r'Print a detailed analysis of the atomic environm'
                       + r'ents in  `trainstruct.data` and `teststruct.data'
                       + r'`.',
        'format': r'environment_analysis',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'epochs': {
        'description': r'The number of epochs for fitting. If `0` is spec'
                       + r'ified, `RuNNer` will calculate the error and ter'
                       + r'minate without adjusting weights.',
        'format': r'epochs i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Number of epochs.',
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_alpha': {
        'description': r'Parameter $lpha$ for the Ewald summation. Deter'
                       + r'mines the accuracy of the  electrostatic energy '
                       + r'and force evaluation for periodic systems togeth'
                       + r'er with  ewald_kmax and  ewald_cutoff. Recommend'
                       + r'ed settings are  (ewald_alpha = 0.2 and  ewald_k'
                       + r'max or (ewald_alpha = 0.5 and  ewald_kmax and a '
                       + r' sufficiently large ewald_cutoff.',
        'format': r'ewald_alpha a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'The default_value of the parameter $lph'
                               + r'a$ for the Ewald summation.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_cutoff': {
        'description': r'Parameter for the Ewald summation. Determines th'
                       + r'e accuracy of the  electrostatic energy and forc'
                       + r'e evaluation for periodic systems together with '
                       + r' ewald_kmax and  ewald_alpha. Must be chosen suf'
                       + r'ficiently large because it determines the number'
                       + r' of neighbors taken into account in the real spa'
                       + r'ce part of the Ewald summation (e.g. 15.0 Bohr)',
        'format': r'ewald_cutoff a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'The cutoff radius if the Ewald summation'
                               + r' (unit: Bohr).',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_kmax': {
        'description': r'Parameter for the reciprocal space part of the E'
                       + r'wald summation. Determines the accuracy of the e'
                       + r'lectrostatic energy and force evaluation for  pe'
                       + r'riodic systems together with  ewald_alpha and  e'
                       + r'wald_cutoff. Recommended settings are  (ewald_al'
                       + r'pha = 0.2 and  ewald_kmax or (ewald_alpha = 0.5 '
                       + r'and  ewald_kmax and a  sufficiently large ewald_'
                       + r'cutoff.',
        'format': r'ewald_kmax i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'k-space cutoff for the Ewald summation.',
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_prec': {
        'description': r'This parameter determines the error tolerance in'
                       + r' electrostatic energy and force  evaluation for '
                       + r'periodic systems when Ewald Summation is used. `'
                       + r'RuNNer` will  automatically choose the optimized'
                       + r'  ewald_alpha, ewald_kmax, and  ewald_cutoff.',
        'format': r'ewald_prec a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'The desired precision of the Ewald summa'
                               + r'tion. Recommended default_values are $10'
                               + r'^{-5}$ to $10^{-6}$.',
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'find_contradictions': {
        'description': r'This keyword can be used in  runner_mode 2 to te'
                       + r'st if the symmetry functions are able to disting'
                       + r'uish different atomic environments sufficiently.'
                       + r' If two atomic environments of a given element a'
                       + r're very similar, they will result in very simila'
                       + r'r symmetry function vectors. Therefore, the leng'
                       + r'th of the difference vector $$ \Delta G = \sqrt{'
                       + r'\sum_{i=1}^N (G_{i,1}-G_{i,2})^2} \,,\notag $$ ('
                       + r'$N$ runs over all individual symmetry functions)'
                       + r' will be close to zero. If the environments are '
                       + r'really similar, the absolute forces acting on th'
                       + r'e atom should be similar as well, which is measu'
                       + r'red by $$ \begin{align} \Delta F &= |\sqrt{F_{1,'
                       + r'x}^2+F_{1,y}^2+F_{1,z}^2}            -\sqrt{F_{2'
                       + r',x}^2+F_{2,y}^2+F_{2,z}^2}|\,,\notag\\          '
                       + r'&= |F_1-F_2| \notag\,. \end{align} $$ If the for'
                       + r'ces are different ($\Delta F >$ `a1`) but the sy'
                       + r'mmetry functions similar ($\Delta G <$ `a0`) for'
                       + r' an atom pair, a message will be printed in the '
                       + r'output file. The optimal choices for `a0` and `a'
                       + r'1` are system dependent and should be selected s'
                       + r'uch that only the most contradictory data is fou'
                       + r'nd. It is not recommended to keep this keyword s'
                       + r'witched on routinely, because it requires substa'
                       + r'ntial CPU time.',
        'format': r'find_contradictions a0 a1',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Symmetry function threshold $\Delta G$.',
                'type': float,
                'default_value': 0.0,
            },
            'a1': {
                'description': r'Force threshold $\Delta F$.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'fitting_unit': {
        'description': r'Set the energy unit that is printed to the outpu'
                       + r't files during training in runner_mode 2.',
        'format': r'fitting_unit i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Switch for different energy units.',
                'type': str,
                'default_value': r'eV',
                'options': {
                    'eV': {
                        'description': r'Unit: `eV`. The energy RMSE and '
                                       + r'MAD in the output file are given'
                                       + r' in eV/atom, the force error is '
                                       + r'given in eV/Bohr.',
                    },
                    'Ha': {
                        'description': r'Unit: `Ha`. The energy RMSE and '
                                       + r'MAD in the output file are given'
                                       + r' in Ha/atom, the force error is '
                                       + r'given in Ha/Bohr.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'fix_weights': {
        'type': bool,
        'description': r'Do not optimize all weights, but freeze some wei'
                       + r'ghts, which are specified by the keywords  weigh'
                       + r't_constraint and  weighte_constraint.',
        'format': r'fix_weights',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'fixed_charge': {
        'description': r'Use a fixed charge for all atoms of the specifie'
                       + r'd element independent of the chemical environmen'
                       + r't.',
        'format': r'fixed_charge element a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r'.',
                'type': str,
                'default_value': None,
            },
            'a0': {
                'description': r'The fixed charge of all atoms of this el'
                               + r'ement (unit: electron charge).',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'fixed_gausswidth': {
        'description': r'This keyword specifies the Gaussian width for ca'
                       + r'lculating the charges and  electrostatic energy '
                       + r'and forces in 4G-HDNNPs. ',
        'format': r'fixed_gausswidth element a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the element'
                               + r'.',
                'type': str,
                'default_value': None,
            },
            'a0': {
                'description': r'The Gaussian width for all atoms of this'
                               + r' element (unit: Bohr).',
                'type': float,
                'default_value': -99.0,
            },
        },
        'allow_multiple': True,
    },
    'fixed_short_energy_error_threshold': {
        'description': r'Only consider points in the weight update during'
                       + r'  runner_mode 2 for which the  absolute error of'
                       + r' the total energy is higher than  fixed_short_en'
                       + r'ergy_error_threshold.',
        'format': r'fixed_short_energy_error_threshold a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The lower threshold for the absolute err'
                               + r'or of the total energy.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'fixed_short_force_error_threshold': {
        'description': r'Only consider points in the weight update during'
                       + r'  runner_mode 2 for which the  absolute error of'
                       + r' the total force is higher than  fixed_short_for'
                       + r'ce_error_threshold.',
        'format': r'fixed_short_force_error_threshold a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The lower threshold for the absolute err'
                               + r'or of the total force.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'force_grouping_by_structure': {
        'type': bool,
        'description': r'Do not update the short-range NN weights after t'
                       + r'he presentation of an  individual atomic force v'
                       + r'ector, but average the derivatives with respect '
                       + r'to the  weights over the number of force vectors'
                       + r' per structure.',
        'format': r'force_grouping_by_structure',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'force_threshold': {
        'description': r'Set a force threshold for fitting data. If any f'
                       + r'orce component of a structure in the reference s'
                       + r'et is larger than `a0` then the point is not use'
                       + r'd and eliminated from the data set. ',
        'format': r'force_threshold a0',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The upper threshold for force components'
                               + r' (unit: Ha/Bohr)',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'force_update_scaling': {
        'description': r'Since for a given structure the number of forces'
                       + r' is much larger than the number of energies, the'
                       + r' force updates can have a dominant influence on '
                       + r'the fits. This can result in poor energy errors.'
                       + r' Using this option the relative strength of the '
                       + r'energy and the forces can be  adjusted. A defaul'
                       + r't_value of 0.1 means that the influence of the e'
                       + r'nergy is 10 times stronger than of a single forc'
                       + r'e. A negative default_value will automatically b'
                       + r'alance the strength of the energy and of the for'
                       + r'ces by taking into account the actual number of '
                       + r'atoms of each structures.',
        'format': r'force_update_scaling a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The relative strength of the energy and '
                               + r'forces for a weight update.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'global_activation_electrostatic': {
        'description': r'Set the default activation function for each hid'
                       + r'den layer and the output layer in the electrosta'
                       + r'tic NNs of all elements. ',
        'format': r'global_activation_electrostatic type [type...]',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [type...]': {
                'description': r'The kind of activation function. One `ty'
                               + r'pe` has to be given for each layer in th'
                               + r'e NN. Options are listed under global_ac'
                               + r'tivation_short.',
                'type': str,
                'default_value': [None],
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'global_activation_pair': {
        'description': r'Set the default activation function for each hid'
                       + r'den layer and the output layer in the NNs of all'
                       + r' element pairs. ',
        'format': r'global_activation_pair type [type...]',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [type...]': {
                'description': r'The kind of activation function. One `ty'
                               + r'pe` has to be given for each layer in th'
                               + r'e NN. Options are listed under global_ac'
                               + r'tivation_short.',
                'type': str,
                'default_value': [None],
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'global_activation_short': {
        'description': r'Set the activation function for each hidden laye'
                       + r'r and the output layer in the short range NNs of'
                       + r' all elements. ',
        'format': r'global_activation_short type [type...]',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [type...]': {
                'description': r'The kind of activation function. One `ty'
                               + r'pe` has to be given for each layer in th'
                               + r'e NN.',
                'type': str,
                'default_value': [None],
                'options': {
                    'c': {
                        'description': r'Cosine function: $\cos(x)$',
                    },
                    'e': {
                        'description': r'Exponential function: $\exp(-x)$',
                    },
                    'g': {
                        'description': r'Gaussian function: $\exp(-\alpha'
                                       + r' x^2)$',
                    },
                    'h': {
                        'description': r'Harmonic function: $x^2$.',
                    },
                    'l': {
                        'description': r'Linear function: $x$',
                    },
                    'p': {
                        'description': r'Softplus function: $\ln(1+\exp(x'
                                       + r'))$',
                    },
                    's': {
                        'description': r'Sigmoid function v1: $(1-\exp(-x'
                                       + r'))^{-1}$',
                    },
                    'S': {
                        'description': r'Sigmoid function v2: $1-(1-\exp('
                                       + r'-x))^{-1}$',
                    },
                    't': {
                        'description': r'Hyperbolic tangent function: $	a'
                                       + r'nh(x)$',
                    },
                },
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'global_hidden_layers_electrostatic': {
        'description': r'Set the default number of hidden layers in the e'
                       + r'lectrostatic NNs of all  elements. Internally 1 '
                       + r'is added to `maxnum_layers_elec`, which also inc'
                       + r'ludes  the output layer.',
        'format': r'global_hidden_layers_electrostatic layers',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'layers': {
                'description': r'The number of hidden layers.',
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_hidden_layers_pair': {
        'description': r'Set the default number of hidden layers in the N'
                       + r'Ns of all element pairs.  Internally 1 is added '
                       + r'to `maxnum_layers_short_pair`, which also includ'
                       + r'es  the output layer.',
        'format': r'global_hidden_layers_pair layers',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'layers': {
                'description': r'The number of hidden layers.',
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_hidden_layers_short': {
        'description': r'Set the default number of hidden layers in the s'
                       + r'hort-range NNs of all  elements. Internally 1 is'
                       + r' added to `maxnum_layers_short`, which also incl'
                       + r'udes  the output layer.',
        'format': r'global_hidden_layers_short layers',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'layers': {
                'description': r'The number of hidden layers.',
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_nodes_electrostatic': {
        'description': r'Set the default number of nodes in the hidden la'
                       + r'yers of the electrostatic NNs in case of  electr'
                       + r'ostatic_type 1. In the array, the entries `1 - ('
                       + r'maxnum_layerseelec - 1)` refer to the hidden  la'
                       + r'yers. The first entry (0) refers to the nodes in'
                       + r' the input layer and is  determined automaticall'
                       + r'y from the symmetry functions.',
        'format': r'global_nodes_electrostatic i0 [i1...]',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0 [i1...]': {
                'description': r'The number of nodes to be set in each la'
                               + r'yer.',
                'type': int,
                'default_value': [None],
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'global_nodes_pair': {
        'description': r'Set the default number of nodes in the hidden la'
                       + r'yers of the pairwise NNs in case of  nn_type_sho'
                       + r'rt 2.',
        'format': r'global_nodes_pair i0 [i1...]',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0 [i1...]': {
                'description': r'The number of nodes to be set in each la'
                               + r'yer.',
                'type': int,
                'default_value': [None],
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'global_nodes_short': {
        'description': r'Set the default number of nodes in the hidden la'
                       + r'yers of the short-range NNs in case of  nn_type_'
                       + r'short 1. In the array, the entries `1 - maxnum_l'
                       + r'ayersshort - 1` refer to the hidden  layers. The'
                       + r' first entry (0) refers to the nodes in the inpu'
                       + r't layer and is  determined automatically from th'
                       + r'e symmetry functions.',
        'format': r'global_nodes_short i0 [i1...]',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0 [i1...]': {
                'description': r'The number of nodes to be set in each la'
                               + r'yer.',
                'type': int,
                'default_value': [None],
                'allow_multiple': True,
            },
        },
        'allow_multiple': False,
    },
    'global_pairsymfunction_short': {
        'description': r'Specification of the global symmetry functions f'
                       + r'or all element pairs in the  pairwise NN.',
        'format': r'global_pairsymfunction_short type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires parame'
                                       + r'ters `eta` and `rshift`. ```runn'
                                       + r'er-config global_pairsymfunction'
                                       + r'_short 2 eta rshift cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, `lambda`, and `ze'
                                       + r'ta`. ```runner-config global_pai'
                                       + r'rsymfunction_short 3 eta lambda '
                                       + r'zeta cutoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'lambda': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'zeta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires parame'
                                       + r'ter `eta`. ```runner-config glob'
                                       + r'al_pairsymfunction_short 4 eta c'
                                       + r'utoff ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig global_pairsymfunction_short '
                                       + r'5 eta ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config global_pairsymfunction_'
                                       + r'short 6 cutoff ```',
                    },
                    8: {
                        'description': r'Not implemented.',
                    },
                    '8': {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, and `rshift`. ```'
                                       + r'runner-config global_pairsymfunc'
                                       + r'tion_short 8 eta rshift cutoff `'
                                       + r'``',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`. ```runner-config '
                                       + r'global_pairsymfunction_short 9 e'
                                       + r'ta cutoff ```.',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_symfunction_electrostatic': {
        'description': r'Specification of global symmetry functions for a'
                       + r'll elements and all element  combinations for th'
                       + r'e electrostatic NN.',
        'format': r'global_symfunction_electrostatic type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires parame'
                                       + r'ters `eta` and `rshift`. ```runn'
                                       + r'er-config global_symfunction_ele'
                                       + r'ctrostatic 2 eta rshift cutoff `'
                                       + r'``',
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0,
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, `lambda`, and `ze'
                                       + r'ta`. ```runner-config global_sym'
                                       + r'function_electrostatic 3 eta lam'
                                       + r'bda zeta cutoff ```',
                        'parameters': {
                            'eta': 0.0,
                            'lambda': 0.0,
                            'zeta': 0.0,
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires parame'
                                       + r'ter `eta`. ```runner-config glob'
                                       + r'al_symfunction_electrostatic 2 e'
                                       + r'ta cutoff ```',
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig global_symfunction_electrosta'
                                       + r'tic 5 eta ```',
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config global_symfunction_elec'
                                       + r'trostatic 6 cutoff ```',
                    },
                    8: {
                        'description': r'Not implemented.',
                    },
                    '8': {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, and `rshift`. ```'
                                       + r'runner-config global_symfunction'
                                       + r'_electrostatic 8 eta rshift cuto'
                                       + r'ff ```',
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0,
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`. ```runner-config '
                                       + r'global_symfunction_electrostatic'
                                       + r' element 9 eta cutoff ```.',
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_symfunction_short': {
        'description': r'Specification of global symmetry functions for a'
                       + r'll elements and all element  combinations for th'
                       + r'e short-range atomic NN.',
        'format': r'global_symfunction_short type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`.',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires parame'
                                       + r'ters `eta` and `rshift`. ```runn'
                                       + r'er-config global_symfunction_sho'
                                       + r'rt 2 eta rshift cutoff ```',
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0,
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, `lambda`, and `ze'
                                       + r'ta`. ```runner-config global_sym'
                                       + r'function_short 3 eta lambda zeta'
                                       + r' cutoff ```',
                        'parameters': {
                            'eta': 0.0,
                            'lambda': 0.0,
                            'zeta': 0.0,
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires parame'
                                       + r'ter `eta`. ```runner-config glob'
                                       + r'al_symfunction_short 2 eta cutof'
                                       + r'f ```',
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig global_symfunction_short 5 et'
                                       + r'a ```',
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config global_symfunction_shor'
                                       + r't 6 cutoff ```',
                    },
                    8: {
                        'description': r'Not implemented.',
                    },
                    '8': {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`, and `rshift`. ```'
                                       + r'runner-config global_symfunction'
                                       + r'_short 8 eta rshift cutoff ```',
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0,
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires `para'
                                       + r'meters` `eta`. ```runner-config '
                                       + r'global_symfunction_short 9 eta c'
                                       + r'utoff ```.',
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'growth_mode': {
        'description': r'If this keyword is used, not the full training s'
                       + r'et will be used in each epoch.  First, only a fe'
                       + r'w points will be used, and after a specified num'
                       + r'ber of epochs further points will be included an'
                       + r'd so on. ',
        'format': r'growth_mode i0 i1',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Number of points that will be added to t'
                               + r'he training set every `i1` steps.',
                'type': int,
                'default_value': 0,
            },
            'i1': {
                'description': r'Number of steps to wait before increasin'
                               + r'g the number of training points.',
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'initialization_only': {
        'type': bool,
        'description': r'With this keyword, which is active only in  runn'
                       + r'er_mode 2, `RuNNer` will stop  after the initial'
                       + r'ization of the run before epoch 0, i.e. no fit w'
                       + r'ill be done.  This is meant as an automatic stop'
                       + r' of the program in case only the analysis carrie'
                       + r'd out in the initialization of  runner_mode 2  i'
                       + r's of interest.',
        'format': r'initialization_only',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'ion_forces_only': {
        'type': bool,
        'description': r'If this keyword is set, for structures with a no'
                       + r'nzero net charge only the forces will be used fo'
                       + r'r fitting, the energies will be omitted. This ke'
                       + r'yword is  currently implemented only for the ato'
                       + r'mic short range part.',
        'format': r'ion_forces_only',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'joint_energy_force_update': {
        'type': bool,
        'description': r'This is an experimental keyword not fully tested'
                       + r'. For each atom only one weight update is done f'
                       + r'or an averaged set of gradients calculated from '
                       + r'the energy and all forces (not yet working well)'
                       + r'.',
        'format': r'joint_energy_force_update',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'kalman_damp_charge': {
        'description': r'Reduce the effective RMSE on the charges for the'
                       + r' Kalman filter update of the weights in the elec'
                       + r'trostatic NN.',
        'format': r'kalman_damp_charge a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of charge RMSE that is consider'
                               + r'ed for the weight update. 100% = 1.0.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_damp_force': {
        'description': r'Reduce the effective RMSE on the forces for the '
                       + r'Kalman filter update of the weights in the short'
                       + r'-range NN.',
        'format': r'kalman_damp_force a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of force RMSE that is considere'
                               + r'd for the weight update. 100% = 1.0.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_damp_short': {
        'description': r'Reduce the effective RMSE on the energies for th'
                       + r'e Kalman filter update of the weights in the sho'
                       + r'rt-range NN.',
        'format': r'kalman_damp_short a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of energy RMSE that is consider'
                               + r'ed for the weight update. 100% = 1.0.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_epsilon': {
        'description': r'Set the initialization parameter for the correla'
                       + r'tion matrix of the Kalman filter according to $$'
                       + r' P(0)=\epsilon^{-1} \mathcal{I}. $$ $\epsilon$ i'
                       + r's often set to the order of $10^{-3}$ to $10^{-2'
                       + r'}$.',
        'format': r'kalman_epsilon a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $\epsilon$.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_lambda_charge': {
        'description': r'Kalman filter parameter $\lambda$ for the electr'
                       + r'ostatic NN weight updates.',
        'format': r'kalman_lambda_charge a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $\lambda$.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_lambda_short': {
        'description': r'Kalman filter parameter $\lambda$ for the short '
                       + r'range NN weight updates.',
        'format': r'kalman_lambda_short a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $\lambda$.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_nue_charge': {
        'description': r'Kalman filter parameter $\lambda_0$ for the elec'
                       + r'trostatic NN weight updates.',
        'format': r'kalman_nue_charge a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $\lambda_0$.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'kalman_nue_short': {
        'description': r'Kalman filter parameter $\lambda_0$ for the shor'
                       + r't range weight updates.',
        'format': r'kalman_nue_short a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $\lambda_0$.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'kalman_q0': {
        'description': r'It is possible to add artificial process noise f'
                       + r'or the Kalman filter in the form of  $$ Q(t) =q('
                       + r't)\mathcal{I}, $$  with either a fixed $q(t)=q(0'
                       + r')$ or annealing from a higher $q(0)$ to  $q_{\ma'
                       + r'thrm{min}}$ following a scheme like $$ q(t) = \m'
                       + r'ax(q_{0}e^{-t/\tau_{q}}, q_{\mathrm{min}}). $$ T'
                       + r'he default_value of $q(0)$ is usually set betwee'
                       + r'n $10^{-6}$ and $10^{-2}$. It is recommended for'
                       + r' the user to do some test for each new system, a'
                       + r'ltering kalman_q0,  kalman_qmin and  kalman_qtau'
                       + r' to obtain the optimal performance for minimizin'
                       + r'g the root mean square error.',
        'format': r'kalman_q0 a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $q(0)$.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_qmin': {
        'description': r'Parameter $q_{\mathrm{min}}$ for adding artifici'
                       + r'al process noise to the Kalman filter noise matr'
                       + r'ix. See kalman_q0 for a more detailed explanatio'
                       + r'n.',
        'format': r'kalman_qmin a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $q_{\mathrm{min}}$.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_qtau': {
        'description': r'Parameter $	au_q$ for adding artificial process '
                       + r'noise to the Kalman filter noise matrix. See kal'
                       + r'man_q0 for a more detailed explanation.',
        'format': r'kalman_qtau a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'default_value of $	au_q$.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'max_energy': {
        'description': r'Set an upper threshold for the consideration of '
                       + r'a structure during the weight  update. If the to'
                       + r'tal energy is above  max_energy] the data point '
                       + r'will be ignored.',
        'format': r'max_energy a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Maximum energy of a structure to be cons'
                               + r'idered for the weight update (unit: Hart'
                               + r'ree).',
                'type': float,
                'default_value': 10000.0,
            },
        },
        'allow_multiple': False,
    },
    'max_force': {
        'description': r'Set an upper threshold for the consideration of '
                       + r'a structure during the weight  update. If any fo'
                       + r'rce component is above  max_force] the data poin'
                       + r't will be ignored.',
        'format': r'max_force a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Maximum force component of a structure t'
                               + r'o be considered for the weight update (u'
                               + r'nit: Hartree/Bohr).',
                'type': float,
                'default_value': 10000.0,
            },
        },
        'allow_multiple': False,
    },
    'md_mode': {
        'type': bool,
        'description': r'The purpose of this keyword is to reduce the out'
                       + r'put to enable the incorporation of `RuNNer` into'
                       + r' a MD code.',
        'format': r'md_mode',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'mix_all_points': {
        'type': bool,
        'description': r'Randomly reorder the data points in the data set'
                       + r' at the beginning of each new epoch.',
        'format': r'mix_all_points',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'nguyen_widrow_weights_ewald': {
        'type': bool,
        'description': r'Initialize the elecrostatic NN weights according'
                       + r' to the scheme proposed by Nguyen and Widrow. Th'
                       + r'e initial weights and bias default_values in the'
                       + r' hidden layer are chosen such that the input spa'
                       + r'ce is evenly distributed over the nodes. This ma'
                       + r'y speed up the training process.',
        'format': r'nguyen_widrow_weights_ewald',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'nguyen_widrow_weights_short': {
        'type': bool,
        'description': r'Initialize the short-range NN weights according '
                       + r'to the scheme proposed by Nguyen and Widrow. The'
                       + r' initial weights and bias default_values in the '
                       + r'hidden layer are chosen such that the input spac'
                       + r'e is evenly distributed over the nodes. This may'
                       + r' speed up the training process.',
        'format': r'nguyen_widrow_weights_short',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'nn_type_short': {
        'description': r'Specify the NN type of the short-range part.',
        'format': r'nn_type_short i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Set the short-range NN type.',
                'type': int,
                'default_value': None,
                'options': {
                    1: {
                        'description': r'Behler-Parrinello atomic NNs. Th'
                                       + r'e short range energy is construc'
                                       + r'ted as a sum of environment-depe'
                                       + r'ndent atomic energies.',
                    },
                    2: {
                        'description': r'Pair NNs. The short range energy'
                                       + r' is constructed as a sum of envi'
                                       + r'ronment-dependent pair energies.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'nnp_gen': {
        'description': r'This keyword specifies the generation of HDNNP t'
                       + r'hat will be constructed.',
        'format': r'nnp_gen i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Set the short-range and electrostatics N'
                               + r'N type.',
                'type': int,
                'default_value': None,
                'options': {
                    2: {
                        'description': r'2G-HDNNPs only include the short'
                                       + r'-range part (Behler-Parrinello a'
                                       + r'tomic NNs). Users should also sp'
                                       + r'ecify use_short_nn.',
                    },
                    3: {
                        'description': r'3G-HDNNPs include both the short'
                                       + r'-range part and the long-range e'
                                       + r'lectrostatic part. Users are adv'
                                       + r'ised to first construct a repres'
                                       + r'entation for the electrostatic p'
                                       + r'art by specifying use_electrosta'
                                       + r'tics and then switch to the shor'
                                       + r't range part by setting both use'
                                       + r'_short_nn and use_electrostatics'
                                       + r'.',
                    },
                    4: {
                        'description': r'4G-HDNNPs include both the short'
                                       + r'-range part and the long-range e'
                                       + r'lectrostatic part. Users are adv'
                                       + r'ised to first construct a repres'
                                       + r'entation for the electrostatic p'
                                       + r'art by specifying use_electrosta'
                                       + r'tics and then switch to the shor'
                                       + r't range part by setting both use'
                                       + r'_short_nn and use_electrostatics'
                                       + r'.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'noise_charge': {
        'description': r'Introduce artificial noise on the atomic charges'
                       + r' in the training process by setting a lower thre'
                       + r'shold that the absolute charge error of a data p'
                       + r'oint has to  surpass before being considered for'
                       + r' the weight update.   ',
        'format': r'noise_charge a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Noise charge threshold (unit: Hartree pe'
                               + r'r atom). Must be positive.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'noise_energy': {
        'description': r'Introduce artificial noise on the atomic energie'
                       + r's in the training process by setting a lower thr'
                       + r'eshold that the absolute energy error of a data '
                       + r'point has to  surpass before being considered fo'
                       + r'r the weight update.   ',
        'format': r'noise_energy a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Noise energy threshold (unit: electron c'
                               + r'harge). Must be positive.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'noise_force': {
        'description': r'Introduce artificial noise on the atomic forces '
                       + r'in the training process by setting a lower thres'
                       + r'hold that the absolute force error of a data poi'
                       + r'nt has to  surpass before being considered for t'
                       + r'he weight update.   ',
        'format': r'noise_force a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Noise force threshold (unit: Hartree per'
                               + r' Bohr). Must be positive.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'normalize_nodes': {
        'type': bool,
        'description': r'Divide the accumulated sum at each node by the n'
                       + r'umber of nodes in the previous layer before the '
                       + r'activation function is applied. This may help to'
                       + r' activate the activation functions in their non-'
                       + r'linear regions.',
        'format': r'normalize_nodes',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'number_of_elements': {
        'description': r'Specify the number of chemical elements in the s'
                       + r'ystem.',
        'format': r'number_of_elements i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Number of elements.',
                'type': int,
                'default_value': r'None',
            },
        },
        'allow_multiple': False,
    },
    'optmode_charge': {
        'description': r'Specify the optimization algorithm for the atomi'
                       + r'c charges in case of  electrostatic_type 1.',
        'format': r'optmode_charge i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Set the atomic charge optimization algor'
                               + r'ithm.',
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': r'Kalman filter.',
                    },
                    2: {
                        'description': r'Reserved for conjugate gradient,'
                                       + r' not implemented.',
                    },
                    3: {
                        'description': r'Steepest descent. Not recommende'
                                       + r'd.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'optmode_short_energy': {
        'description': r'Specify the optimization algorithm for the short'
                       + r'-range energy contributions.',
        'format': r'optmode_short_energy i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Set the short-range energy optimization '
                               + r'algorithm.',
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': r'Kalman filter.',
                    },
                    2: {
                        'description': r'Reserved for conjugate gradient,'
                                       + r' not implemented.',
                    },
                    3: {
                        'description': r'Steepest descent. Not recommende'
                                       + r'd.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'optmode_short_force': {
        'description': r'Specify the optimization algorithm for the short'
                       + r'-range forces.',
        'format': r'optmode_short_force i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Set the short-range force optimization a'
                               + r'lgorithm.',
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': r'Kalman filter.',
                    },
                    2: {
                        'description': r'Reserved for conjugate gradient,'
                                       + r' not implemented.',
                    },
                    3: {
                        'description': r'Steepest descent. Not recommende'
                                       + r'd.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'parallel_mode': {
        'description': r'This flag controls the parallelization of some s'
                       + r'ubroutines. ',
        'format': r'parallel_mode i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Set the short-range force optimization a'
                               + r'lgorithm.',
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': r'Serial version.',
                    },
                    2: {
                        'description': r'Parallel version.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'points_in_memory': {
        'description': r'This keyword controls memory consumption and IO '
                       + r'and is therefore important to achieve an optimum'
                       + r' performance of `RuNNer`. Has a different meanin'
                       + r'g depending on the current  runner_mode.',
        'format': r'points_in_memory i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'In runner_mode 1 this is the maximum num'
                               + r'ber of structures in memory at a time. I'
                               + r'n runner_mode 3 this is the number of at'
                               + r'oms for which the symmetry functions are'
                               + r' in memory at once. In parallel runs the'
                               + r'se atoms are further split between the p'
                               + r'rocesses.',
                'type': int,
                'default_value': 200,
            },
        },
        'allow_multiple': False,
    },
    'precondition_weights': {
        'type': bool,
        'description': r'Shift the weights of the atomic NNs right after '
                       + r'the initialization so that  the standard deviati'
                       + r'on of the NN energies is the same as the standar'
                       + r'd deviation of the reference energies.',
        'format': r'precondition_weights',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_all_deshortdw': {
        'type': bool,
        'description': r'For debugging only. Prints the derivatives of th'
                       + r'e short range energy with  respect to the short '
                       + r'range NN weight parameters after each update. Th'
                       + r'is derivative array is responsible for the weigh'
                       + r't update. The derivatives (the array `deshortdw`'
                       + r') are written to the file `debug.out`.',
        'format': r'print_all_deshortdw',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_all_dfshortdw': {
        'type': bool,
        'description': r'For debugging only. Prints the derivatives of th'
                       + r'e short range forces with respect to the short r'
                       + r'ange NN weight parameters after each update. Thi'
                       + r's derivative array is responsible for the weight'
                       + r' update. The derivatives (the array `dfshortdw(m'
                       + r'axnum_weightsshort)`) are written to the file `d'
                       + r'ebug.out`.',
        'format': r'print_all_dfshortdw',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_all_electrostatic_weights': {
        'type': bool,
        'description': r'For debugging only. Print the electrostatic NN w'
                       + r'eight parameters after each  update, not only on'
                       + r'ce per epoch to a file. The weights (the array  '
                       + r'`weights_ewald()`) are written to the file `debu'
                       + r'g.out`.',
        'format': r'print_all_electrostatic_weights',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_all_short_weights': {
        'type': bool,
        'description': r'For debugging only. Print the short range NN wei'
                       + r'ght parameters after each  update, not only once'
                       + r' per epoch to a file. The weights (the array `we'
                       + r'ights_short()`) are written to the file `debug.o'
                       + r'ut`. ',
        'format': r'print_all_short_weights',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_convergence_vector': {
        'type': bool,
        'description': r'During training, print a measure for the converg'
                       + r'ence of the weight vector. The output is: ```run'
                       + r'ner-data CONVVEC element epoch C1 C2 wshift wshi'
                       + r'ft2 ``` `C1` and `C2` are two-dimensional coordi'
                       + r'nates of projections of the weight vectors for p'
                       + r'lotting qualitatively the convergence of the wei'
                       + r'ghts. `wshift` is the length (normalized by the '
                       + r'number of weights) of the difference vector of t'
                       + r'he weights between two epochs. `wshift2` is the '
                       + r'length (normalized by the number of weights) of '
                       + r'the difference vector between the current weight'
                       + r's and the weights two epochs ago.',
        'format': r'print_convergence_vector',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_date_and_time': {
        'type': bool,
        'description': r'Print in each training epoch the date and the re'
                       + r'al time in an extra line.',
        'format': r'print_date_and_time',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_force_components': {
        'type': bool,
        'description': r'For debugging only. Prints in  runner_mode 3  th'
                       + r'e contributions of all atomic energies to the fo'
                       + r'rce components of each atom.',
        'format': r'print_force_components',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_mad': {
        'type': bool,
        'description': r'Print a line with the mean absolute deviation as'
                       + r' an additional output line next to the RMSE in  '
                       + r'runner_mode 2.  Usually the MAD is smaller than '
                       + r'the RMSE as outliers do not have such a large  i'
                       + r'mpact.',
        'format': r'print_mad',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'print_sensitivity': {
        'type': bool,
        'description': r'Perform sensitivity analysis on the symmetry fun'
                       + r'ctions of the neural network. The  sensitivity i'
                       + r's a measure of how much the NN output changes wi'
                       + r'th the symmetry functions, i.e. the derivative. '
                       + r'It will be analyzed upon weight initialization a'
                       + r'nd for each training epoch in all short-range, p'
                       + r'air, and electrostatic NNs there are. ',
        'format': r'print_sensitivity',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'random_number_type': {
        'description': r'Specify the type of random number generator used'
                       + r' in `RuNNer`. The seed can be given with the key'
                       + r'word  random_seed.',
        'format': r'random_number_type i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Set the random number generator type.',
                'type': int,
                'default_value': 5,
                'options': {
                    1: {
                        'description': r'Deprecated.',
                    },
                    2: {
                        'description': r'Deprecated.',
                    },
                    3: {
                        'description': r'Deprecated.',
                    },
                    4: {
                        'description': r'Deprecated.',
                    },
                    5: {
                        'description': r'Normal distribution of random nu'
                                       + r'mbers.',
                    },
                    6: {
                        'description': r'Normal distribution of random nu'
                                       + r'mbers with the `xorshift` algori'
                                       + r'thm. **This is the recommended o'
                                       + r'ption.**',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'random_seed': {
        'description': r'Set the integer seed for the random number gener'
                       + r'ator used at many places in  `RuNNer`. In order '
                       + r'to ensure that all results are reproducible, the'
                       + r' same seed will result in exactly the same outpu'
                       + r't at all times (machine and compiler dependence '
                       + r'cannot be excluded). This seed default_value is '
                       + r'used for all random number generator in `RuNNer`'
                       + r', but internally for each purpose a local copy i'
                       + r's made first to avoid interactions between the d'
                       + r'ifferent random number generators.  Please see a'
                       + r'lso the keyword  random_number_type.',
        'format': r'random_seed i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Seed default_value.',
                'type': int,
                'default_value': 200,
            },
        },
        'allow_multiple': False,
    },
    'read_kalman_matrices': {
        'type': bool,
        'description': r'Restart a fit using old Kalman filter matrices f'
                       + r'rom the files `kalman.short.XXX.data` and `kalma'
                       + r'n.elec.XXX.data`. `XXX` is the nuclear charge of'
                       + r' the respective element. Using old Kalman matric'
                       + r'es will reduce the oscillations of the errors wh'
                       + r'en a fit is restarted with the Kalman filter. Th'
                       + r'e Kalman matrices are written to the files using'
                       + r' the keyword save_kalman_matrices',
        'format': r'read_kalman_matrices',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'read_unformatted': {
        'type': bool,
        'description': r'Read old NN weights and/or an old Kalman matrix '
                       + r'from an unformatted input file.',
        'format': r'read_unformatted',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'regularize_fit_param': {
        'description': r'This keyword switches on L2 regularization, main'
                       + r'ly for the electrostatic part in 4G-HDNNPs.',
        'format': r'regularize_fit_param a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Regularization parameter. Recommended se'
                               + r'tting is $10^{-6}$.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'remove_atom_energies': {
        'type': bool,
        'description': r'Remove the energies of the free atoms from the t'
                       + r'otal energies per atom to reduce the absolute de'
                       + r'fault_values of the target energies. This means '
                       + r'that when this keyword is used, `RuNNer` will fi'
                       + r't binding energies instead of total energies. Th'
                       + r'is is expected to facilitate the fitting process'
                       + r' because binding energies are closer to zero. ',
        'format': r'remove_atom_energies',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'remove_vdw_energies': {
        'type': bool,
        'description': r'Subtract van-der-Waals dispersion energy and for'
                       + r'ces from the reference data before fitting a neu'
                       + r'ral network potential. ',
        'format': r'remove_vdw_energies',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'repeated_energy_update': {
        'type': bool,
        'description': r'If this keyword is set, the weights of the short'
                       + r'-range NN are updated a second time after the fo'
                       + r'rce update with respect to the total energies in'
                       + r' the data set.  This usually results in a more a'
                       + r'ccurate potential energy fitting at the cost of '
                       + r'slightly detiorated forces.',
        'format': r'repeated_energy_update',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'reset_kalman': {
        'type': bool,
        'description': r'Re-initialize the correlation matrix of the Kalm'
                       + r'an filter at each new training epoch.',
        'format': r'reset_kalman',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'restrict_weights': {
        'description': r'Restrict the weights of the NN to the interval  '
                       + r'[`-restrictw +1.0`, `restrictw - 1.0`].',
        'format': r'restrict_weights a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Boundary default_value for neural networ'
                               + r'k weights. Must be positive.',
                'type': float,
                'default_value': -100000.0,
            },
        },
        'allow_multiple': False,
    },
    'runner_mode': {
        'description': r'Choose the operating mode of `RuNNer`.',
        'format': r'runner_mode i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'The chosen mode of `RuNNer`.',
                'type': int,
                'default_value': r'None',
                'options': {
                    1: {
                        'description': r'Preparation mode. Generate the s'
                                       + r'ymmetry functions from structure'
                                       + r's in the `input.data` file.',
                    },
                    2: {
                        'description': r'Fitting mode. Determine the NN w'
                                       + r'eight parameters.',
                    },
                    3: {
                        'description': r'Production mode. Application of '
                                       + r'the NN potential, prediction of '
                                       + r'the energy and forces of all str'
                                       + r'uctures in the `input.data` file'
                                       + r'.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'save_kalman_matrices': {
        'type': bool,
        'description': r'Save the Kalman filter matrices to the files `ka'
                       + r'lman.short.XXX.data` and `kalman.elec.XXX.data`.'
                       + r' `XXX` is the nuclear charge of the respective e'
                       + r'lement. The Kalman matrices are read from the fi'
                       + r'les using the keyword  read_kalman_matrices.',
        'format': r'save_kalman_matrices',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'scale_max_elec': {
        'description': r'Rescale the electrostatic symmetry functions to '
                       + r'an interval given by * scale_min_elec and  * sca'
                       + r'le_max_elec  For further details please see  sca'
                       + r'le_symmetry_functions.',
        'format': r'scale_max_elec a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Upper boundary default_value for rescali'
                               + r'ng the electrostatic symmetry functions.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_max_short': {
        'description': r'Rescale the electrostatic symmetry functions to '
                       + r'an interval given by * scale_min_elec and  * sca'
                       + r'le_max_elec  For further details please see  sca'
                       + r'le_symmetry_functions.',
        'format': r'scale_max_short a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Upper boundary default_value for rescali'
                               + r'ng the electrostatic symmetry functions.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_max_short_atomic': {
        'description': r'Rescale the short-range symmetry functions to an'
                       + r' interval given by * scale_min_short_atomic and '
                       + r' * scale_max_short_atomic  For further details p'
                       + r'lease see  scale_symmetry_functions.',
        'format': r'scale_max_short_atomic a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Upper boundary default_value for rescali'
                               + r'ng the short-range symmetry functions.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_max_short_pair': {
        'description': r'Rescale the short-range pairwise symmetry functi'
                       + r'ons to an interval given by * scale_min_short_pa'
                       + r'ir and  * scale_max_short_pair  For further deta'
                       + r'ils please see  scale_symmetry_functions.',
        'format': r'scale_max_short_pair a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Upper boundary default_value for rescali'
                               + r'ng the short-range pairwise symmetry fun'
                               + r'ctions.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_min_elec': {
        'description': r'Rescale the electrostatic symmetry functions to '
                       + r'an interval given by * scale_min_elec and  * sca'
                       + r'le_max_elec  For further details please see  sca'
                       + r'le_symmetry_functions.',
        'format': r'scale_min_elec a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Lower boundary default_value for rescali'
                               + r'ng the electrostatic symmetry functions.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_min_short_atomic': {
        'description': r'Rescale the short-range symmetry functions to an'
                       + r' interval given by * scale_min_short_atomic and '
                       + r' * scale_max_short_atomic  For further details p'
                       + r'lease see  scale_symmetry_functions.',
        'format': r'scale_min_short_atomic a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Lower boundary default_value for rescali'
                               + r'ng the short-range symmetry functions.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_min_short_pair': {
        'description': r'Rescale the short-range pairwise symmetry functi'
                       + r'ons to an interval given by * scale_min_short_pa'
                       + r'ir and  * scale_max_short_pair  For further deta'
                       + r'ils please see  scale_symmetry_functions.',
        'format': r'scale_min_short_pair a0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': r'Lower boundary default_value for rescali'
                               + r'ng the short-range pairwise symmetry fun'
                               + r'ctions.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_symmetry_functions': {
        'type': bool,
        'description': r'Rescale symmetry functions to a certain interval'
                       + r' (the default interval is 0 to  1). This has num'
                       + r'erical advantages if the orders of magnitudes of'
                       + r' different  symmetry functions are very differen'
                       + r't. If the minimum and maximum default_value for '
                       + r'a symmetry function is the same for all structur'
                       + r'es, rescaling is not possible and `RuNNer` will '
                       + r'terminate with an error. The interval can be spe'
                       + r'cified by the  keywords  * scale_min_short_atomi'
                       + r'c, * scale_max_short_atomic,  * scale_min_short_'
                       + r'pair, and * scale_max_short_pair  for the short '
                       + r'range / pairwise NN and by  * scale_min_elec and'
                       + r'  * scale_max_elec  for the electrostatic NN. ',
        'format': r'scale_symmetry_functions',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'separate_bias_ini_short': {
        'type': bool,
        'description': r'Request a separate random initialization of the '
                       + r'bias weights at the beginning of `runner_mode 2`'
                       + r' on an interval between  `biasweights_min` and `'
                       + r'biasweights_max`. ',
        'format': r'separate_bias_ini_short',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'separate_kalman_short': {
        'type': bool,
        'description': r'Use a different Kalman filter correlation matrix'
                       + r' for the energy and force  update. ',
        'format': r'separate_kalman_short',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'short_energy_error_threshold': {
        'description': r'Threshold default_value for the error of the ene'
                       + r'rgies in units of the RMSE of the previous epoch'
                       + r'. A default_value of 0.3 means that only charges'
                       + r' with an error larger than 0.3*RMSE will be used'
                       + r' for the weight update. Large default_values (ab'
                       + r'out 1.0) will speed up the first epochs, because'
                       + r' only a few points will be used. ',
        'format': r'short_energy_error_threshold a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of energy RMSE that a point nee'
                               + r'ds to reach to be used in the weight upd'
                               + r'ate.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'short_energy_fraction': {
        'description': r'Defines the random fraction of energies used for'
                       + r' fitting the short range weights.',
        'format': r'short_energy_fraction a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of energies used for short-rang'
                               + r'e fitting. 100% = 1.0.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'short_energy_group': {
        'description': r'Do not update the short range NN weights after t'
                       + r'he presentation of an  individual atomic charge,'
                       + r' but average the derivatives with respect to the'
                       + r'  weights over the specified number of structure'
                       + r's for each element.',
        'format': r'short_energy_group i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Number of structures per group. The maxi'
                               + r'mum is given by points_in_memory.',
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
    'short_force_error_threshold': {
        'description': r'Threshold default_value for the error of the ato'
                       + r'mic forces in units of the RMSE of the previous '
                       + r'epoch. A default_value of 0.3 means that only fo'
                       + r'rces with an error larger than 0.3*RMSE will be '
                       + r'used for the weight update. Large default_values'
                       + r' (about 1.0) will speed up the first epochs, bec'
                       + r'ause only a few points will be used. ',
        'format': r'short_force_error_threshold a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of force RMSE that a point need'
                               + r's to reach to be used in the weight upda'
                               + r'te.',
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'short_force_fraction': {
        'description': r'Defines the random fraction of forces used for f'
                       + r'itting the short range weights.',
        'format': r'short_force_fraction a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Fraction of force used for short-range f'
                               + r'itting. 100% = 1.0.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'short_force_group': {
        'description': r'Do not update the short range NN weights after t'
                       + r'he presentation of an  individual atomic force, '
                       + r'but average the derivatives with respect to the '
                       + r' weights over the specified number of forces for'
                       + r' each element.',
        'format': r'short_force_group i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'Number of structures per group. The maxi'
                               + r'mum is given by points_in_memory.',
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
    'shuffle_weights_short_atomic': {
        'description': r'Randomly shuffle some weights in the short-range'
                       + r' atomic NN after a defined  number of epochs.',
        'format': r'shuffle_weights_short_atomic i0 a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'The weights will be shuffled every `i0` '
                               + r'epochs.',
                'type': int,
                'default_value': 10,
            },
            'a0': {
                'description': r'Treshold that a random number has to pas'
                               + r's so that the weights at handled will be'
                               + r' shuffled. This indirectly defines the n'
                               + r'umber of weights that will be shuffled.',
                'type': float,
                'default_value': 0.1,
            },
        },
        'allow_multiple': False,
    },
    'steepest_descent_step_charge': {
        'description': r'Step size for steepest descent fitting of the at'
                       + r'omic charges.',
        'format': r'steepest_descent_step_charge a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Charge steepest descent step size.',
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'steepest_descent_step_energy_short': {
        'description': r'Step size for steepest descent fitting of the sh'
                       + r'ort-range energy.',
        'format': r'steepest_descent_step_energy_short a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Short-range energy steepest descent step'
                               + r' size.',
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'steepest_descent_step_force_short': {
        'description': r'Step size for steepest descent fitting of the sh'
                       + r'ort-range forces.',
        'format': r'steepest_descent_step_force_short a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Short-range force steepest descent step '
                               + r'size.',
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'symfunction_correlation': {
        'type': bool,
        'description': r"Determine and print Pearson's correlation of all"
                       + r' pairs of symmetry functions.',
        'format': r'symfunction_correlation',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'symfunction_electrostatic': {
        'description': r'Specification of the symmetry functions for a sp'
                       + r'ecific element with a specific  neighbor element'
                       + r' combination for the electrostatic NN.',
        'format': r'symfunction_electrostatic element type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the center '
                               + r'element.',
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires the pa'
                                       + r'ir `element` and parameters `eta'
                                       + r'` and `rshift`. ```runner-config'
                                       + r' symfunction_electrostatic eleme'
                                       + r'nt 2 element eta rshift cutoff `'
                                       + r'``',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires two p'
                                       + r'air elements and `parameters` `e'
                                       + r'ta`, `lambda`, and `zeta`. ```ru'
                                       + r'nner-config symfunction_electros'
                                       + r'tatic element 3 element element '
                                       + r'eta lambda zeta cutoff ```',
                        'parameters': {
                            'element1': {
                                'type': str,
                                'default_value': None,
                            },
                            'element2': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'lambda': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'zeta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires a pair'
                                       + r' `element` and the parameter `et'
                                       + r'a`. ```runner-config symfunction'
                                       + r'_electrostatic element 4 element'
                                       + r' eta cutoff ```',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig symfunction_electrostatic ele'
                                       + r'ment 5 eta ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config symfunction_electrostat'
                                       + r'ic element 6 cutoff ```',
                    },
                    7: {
                        'description': r'Not implemented.',
                    },
                    8: {
                        'description': r'Angular function. Requires two `'
                                       + r'element`s and `parameters` `eta`'
                                       + r', and `rshift`. ```runner-config'
                                       + r' symfunction_electrostatic eleme'
                                       + r'nt 8 element element eta rshift '
                                       + r'cutoff ```',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires two `'
                                       + r'element`s and `parameters` `eta`'
                                       + r'. ```runner-config symfunction_e'
                                       + r'lectrostatic element 9 element e'
                                       + r'lement eta cutoff ```.',
                        'parameters': {
                            'element1': {
                                'type': str,
                                'default_value': None,
                            },
                            'element2': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'symfunction_short': {
        'description': r'Specification of the symmetry functions for a sp'
                       + r'ecific element with a specific  neighbor element'
                       + r' combination for the short-range NN.',
        'format': r'symfunction_short element type [parameters] cutoff',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': r'The periodic table symbol of the center '
                               + r'element.',
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': r'The type of symmetry function to be used'
                               + r'. Different `parameters` have to be set '
                               + r'depending on the choice of `type`:',
                'type': int,
                'options': {
                    1: {
                        'description': r'Radial function. Requires no fur'
                                       + r'ther `parameters`.',
                    },
                    2: {
                        'description': r'Radial function. Requires the pa'
                                       + r'ir `element` and parameters `eta'
                                       + r'` and `rshift`. ```runner-config'
                                       + r' symfunction_short element 2 ele'
                                       + r'ment eta rshift cutoff ```',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    3: {
                        'description': r'Angular function. Requires two p'
                                       + r'air elements and `parameters` `e'
                                       + r'ta`, `lambda`, and `zeta`. ```ru'
                                       + r'nner-config symfunction_short el'
                                       + r'ement 3 element element eta lamb'
                                       + r'da zeta cutoff ```',
                        'parameters': {
                            'element1': {
                                'type': str,
                                'default_value': None,
                            },
                            'element2': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'lambda': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'zeta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    4: {
                        'description': r'Radial function. Requires a pair'
                                       + r' `element` and the parameter `et'
                                       + r'a`. ```runner-config symfunction'
                                       + r'_short element 4 element eta cut'
                                       + r'off ```',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': r'Cartesian coordinate function. T'
                                       + r'he parameter `eta` will determin'
                                       + r'e the coordinate axis `eta=1.0: '
                                       + r'X, eta=2.0: Y, eta=3.0: Z`. No `'
                                       + r'cutoff` required. ```runner-conf'
                                       + r'ig symfunction_short element 5 e'
                                       + r'ta ```',
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': r'Bond length function. Requires n'
                                       + r'o further `parameters`. ```runne'
                                       + r'r-config symfunction_short eleme'
                                       + r'nt 6 cutoff ```',
                    },
                    7: {
                        'description': r'Not implemented.',
                    },
                    8: {
                        'description': r'Angular function. Requires two `'
                                       + r'element`s and `parameters` `eta`'
                                       + r', and `rshift`. ```runner-config'
                                       + r' symfunction_short element 8 ele'
                                       + r'ment element eta rshift cutoff `'
                                       + r'``',
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                            'rshift': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    9: {
                        'description': r'Angular function. Requires two `'
                                       + r'element`s and `parameters` `eta`'
                                       + r'. ```runner-config symfunction_s'
                                       + r'hort element 9 element element e'
                                       + r'ta cutoff ```.',
                        'parameters': {
                            'element1': {
                                'type': str,
                                'default_value': None,
                            },
                            'element2': {
                                'type': str,
                                'default_value': None,
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                },
            },
            'cutoff': {
                'description': r'The symmetry function cutoff radius (uni'
                               + r't: Bohr).',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'test_fraction': {
        'description': r'Threshold for splitting between training and tes'
                       + r'ting set in  [`runner_mode`](/runner/reference/k'
                       + r'eywords/runner_mode) 1.',
        'format': r'test_fraction a0',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'Splitting ratio. A default_value of e.g.'
                               + r' 0.1 means that 10% of the structures in'
                               + r' the `input.data` file will be used as t'
                               + r'est set and 90% as training set.',
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'update_single_element': {
        'description': r'During training, only the NN weight parameters f'
                       + r'or the NNs of a specified element will be update'
                       + r'd. In this case the printed errors for the  forc'
                       + r'es and the charges will refer only to this eleme'
                       + r'nt. The total energy error  will remain large si'
                       + r'nce some NNs are not optimized. ',
        'format': r'update_single_element i0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': r'The nuclear charge of the element whose '
                               + r'NN should be updated.',
                'type': int,
                'default_value': r'None',
            },
        },
        'allow_multiple': False,
    },
    'update_worst_charges': {
        'description': r'To speed up the fits for each block specified by'
                       + r' points_in_memory first the worst charges are de'
                       + r'termined. Only these charges are then used for t'
                       + r'he weight update for this block of points, no ma'
                       + r'tter if the fit would be reduced during the upda'
                       + r'te.',
        'format': r'update_worst_charges a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The percentage of worst charges to be co'
                               + r'nsidered for the weight update. A defaul'
                               + r't_value of 0.1 here means to identify th'
                               + r'e worst 10%.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'update_worst_short_energies': {
        'description': r'To speed up the fits for each block specified by'
                       + r' points_in_memory first the worst energies are d'
                       + r'etermined. Only these points are then used for t'
                       + r'he weight update for this block of points, no ma'
                       + r'tter if the fit would be reduced during the upda'
                       + r'te.',
        'format': r'update_worst_short_energies a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The percentage of worst energies to be c'
                               + r'onsidered for the weight update. A defau'
                               + r'lt_value of 0.1 here means to identify t'
                               + r'he worst 10%.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'update_worst_short_forces': {
        'description': r'To speed up the fits for each block specified by'
                       + r' points_in_memory first the worst forces are det'
                       + r'ermined. Only these points are then used for the'
                       + r' weight update for this block of points, no matt'
                       + r'er if the fit would be reduced during the update'
                       + r'.',
        'format': r'update_worst_short_forces a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The percentage of worst forces to be con'
                               + r'sidered for the weight update. A default'
                               + r'_value of 0.1 here means to identify the'
                               + r' worst 10%.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'use_atom_charges': {
        'type': bool,
        'description': r'Use atomic charges for fitting. At the moment th'
                       + r'is flag will be switched on automatically by `Ru'
                       + r'NNer` if electrostatic NNs are requested. In fut'
                       + r'ure versions of `RuNNer` this keyword will be us'
                       + r'ed to control different origins of atomic charge'
                       + r's.',
        'format': r'use_atom_charges',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': True,
        'allow_multiple': False,
    },
    'use_atom_energies': {
        'type': bool,
        'description': r'Check if sum of specified atomic energies is equ'
                       + r'al to the total energy of each structure.',
        'format': r'use_atom_energies',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': True,
        'allow_multiple': False,
    },
    'use_damping': {
        'description': r'In order to avoid too large (too positive or too'
                       + r' negative) weight parameters, this damping schem'
                       + r'e can be used. An additional term is added to th'
                       + r'e absolute energy error. * For the short range e'
                       + r'nergy the modification is:    `errore=(1.d0-damp'
                       + r'w)errore + (dampwsumwsquared) /dble(totnum_weigh'
                       + r'tsshort)` * For the short range forces the modif'
                       + r'ication is:   `errorf=(1.d0-dampw)errorf+(dampws'
                       + r'umwsquared) /dble(num_weightsshort(elementindex('
                       + r'zelem(i2))))` * For the short range forces the m'
                       + r'odification is:   `error=(1.d0-dampw)*error + (d'
                       + r'ampwsumwsquared) /dble(num_weightsewald(elementi'
                       + r'ndex(zelem_list(idx(i1),i2))))`',
        'format': r'use_damping a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'The damping parameter.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'use_electrostatics': {
        'type': bool,
        'description': r'Calculate long range electrostatic interactions '
                       + r'explicitly. The type of atomic charges is specif'
                       + r'ied by the keyword  electrostatic_type .',
        'format': r'use_electrostatics',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_fixed_charges': {
        'type': bool,
        'description': r'Do not use a NN to calculate the atomic charges,'
                       + r' but use fixed charges for each element independ'
                       + r'ent of the chemical environment. electrostatic_t'
                       + r'ype.',
        'format': r'use_fixed_charges',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_gausswidth': {
        'type': bool,
        'description': r'Use Gaussian for modeling atomic charges during '
                       + r'the construction of 4G-HDNNPs.',
        'format': r'use_gausswidth',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_noisematrix': {
        'type': bool,
        'description': r'Use noise matrix for fitting the short range wei'
                       + r'ght with the short range NN  weights with Kalman'
                       + r' filter.',
        'format': r'use_noisematrix',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_old_scaling': {
        'type': bool,
        'description': r'Restart a fit with old scaling parameters for th'
                       + r'e short-range and electrostatic  NNs. The symmet'
                       + r'ry function scaling factors are read from `scali'
                       + r'ng.data`.',
        'format': r'use_old_scaling',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_old_weights_charge': {
        'type': bool,
        'description': r'Restart a fit with old weight parameters for the'
                       + r' electrostatic NN. The files `weightse.XXX.data`'
                       + r' must be present.  If the training data set is  '
                       + r'unchanged, the error of epoch 0 must be the same'
                       + r' as the error of the previous fitting cycle.',
        'format': r'use_old_weights_charge',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_old_weights_short': {
        'type': bool,
        'description': r'Restart a fit with old weight parameters for the'
                       + r' short-range NN. This keyword is active only in '
                       + r'mode 2. The files `weights.XXX.data` must be pre'
                       + r'sent.  If the training data set is unchanged, th'
                       + r'e error of epoch 0 must be the same as the error'
                       + r' of the previous fitting cycle. However, if the '
                       + r'training data is  different, the file `scaling.d'
                       + r'ata` changes and either one of the keywords scal'
                       + r'e_symmetry_functions or center_symmetry_function'
                       + r's is used, the RMSE will be different.',
        'format': r'use_old_weights_short',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_omp_mkl': {
        'type': bool,
        'description': r'Make use of the OpenMP threading in Intels MKL l'
                       + r'ibrary. ',
        'format': r'use_omp_mkl',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_short_forces': {
        'type': bool,
        'description': r'Use forces for fitting the short range NN weight'
                       + r's. In  runner_mode 1, the  files `trainforces.da'
                       + r'ta`, `testforces.data`, `trainforcese.data` and '
                       + r'`testforcese.data` are written.  In  runner_mode'
                       + r' 2,  these files are needed to use the short ran'
                       + r'ge forces for optimizing the  short range weight'
                       + r's. However, if the training data is different, t'
                       + r'he file  `scaling.data` changes and either one o'
                       + r'f the keywords scale_symmetry_functions or cente'
                       + r'r_symmetry_functions is used, the RMSE will be d'
                       + r'ifferent.',
        'format': r'use_short_forces',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_short_nn': {
        'type': bool,
        'description': r'Use the a short range NN. Whether an atomic or p'
                       + r'air-based energy expression is used is determine'
                       + r'd via the keyword nn_type_short.',
        'format': r'use_short_nn',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_systematic_weights_electrostatic': {
        'type': bool,
        'description': r'Overwrite the randomly initialized weights of th'
                       + r'e electrostatic NNs with  systematically calcula'
                       + r'ted weights. The weights are evenly distributed '
                       + r'over the interval between the minimum and maximu'
                       + r'm of the weights after the random initialization'
                       + r'.',
        'format': r'use_systematic_weights_electrostatic',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_systematic_weights_short': {
        'type': bool,
        'description': r'Overwrite the randomly initialized weights of th'
                       + r'e short-range and pairwise NNs with systematical'
                       + r'ly calculated weights. The weights are evenly di'
                       + r'stributed over the interval between the minimum '
                       + r'and maximum of the weights after the random init'
                       + r'ialization.',
        'format': r'use_systematic_weights_short',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'use_vdw': {
        'type': bool,
        'description': r'Turn on dispersion corrections.',
        'format': r'use_vdw',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'vdw_screening': {
        'description': r'Specification of the shape parameters of the Fer'
                       + r'mi-screening function in the employed DFT-D2 dis'
                       + r'persion correction expression.',
        'format': r'vdw_screening s_6 d s_R',
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'arguments': {
            's_6': {
                'description': r'The global scaling parameter $s_6$ in th'
                               + r'e screening function.',
                'type': float,
                'default_value': None,
            },
            'd': {
                'description': r'The exchange-correlation functional depe'
                               + r'ndent damping parameter. More informatio'
                               + r'n can be found in the theory section.',
                'type': float,
                'default_value': None,
            },
            's_R': {
                'description': r'Range separation parameter.',
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'vdw_type': {
        'description': r'Specification of the type of dispersion correcti'
                       + r'on to be employed.',
        'format': r'vdw_type i0',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': r'Type of vdW correction scheme.',
                'type': int,
                'default_value': r'default',
                'options': {
                    1: {
                        'description': r'Simple, environment-dependent di'
                                       + r'spersion correction inspired by '
                                       + r'the Tkatchenko-Scheffler scheme.',
                    },
                    2: {
                        'description': r'Grimme DFT-D2 correction.',
                    },
                    3: {
                        'description': r'Grimme DFT-D3 correction.',
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'weight_analysis': {
        'type': bool,
        'description': r'Print analysis of weights in  runner_mode 2.',
        'format': r'weight_analysis',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'weights_max': {
        'description': r'Set an upper limit for the random initialization'
                       + r' of the short-range NN weights. ',
        'format': r'weights_max a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'This number defines the maximum default_'
                               + r'value for initial random short range wei'
                               + r'ghts.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'weights_min': {
        'description': r'Set a lower limit for the random initialization '
                       + r'of the short-range NN weights. ',
        'format': r'weights_min a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'This number defines the minimum default_'
                               + r'value for initial random short range wei'
                               + r'ghts.',
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'weightse_max': {
        'description': r'Set an upper limit for the random initialization'
                       + r' of the electrostatic NN weights. ',
        'format': r'weightse_max a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'This number defines the maximum default_'
                               + r'value for initial random electrostatic w'
                               + r'eights. This keyword is active only if a'
                               + r'n electrostatic NN is used, i.e. for `el'
                               + r'ectrostatic_type` 1.',
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'weightse_min': {
        'description': r'Set a lower limit for the random initialization '
                       + r'of the electrostatic NN weights. ',
        'format': r'weightse_min a0',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': r'This number defines the minimum default_'
                               + r'value for initial random electrostatic w'
                               + r'eights. This keyword is active only if a'
                               + r'n electrostatic NN is used, i.e. for `el'
                               + r'ectrostatic_type` 1.',
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'write_fit_statistics': {
        'type': bool,
        'description': r'',
        'format': r'write_fit_statistics',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_temporary_weights': {
        'type': bool,
        'description': r'Write temporary weights after each data block de'
                       + r'fined by `points_in_memory` to files `tmpweights'
                       + r'.XXX.out` and `tmpweightse.XXX.out`. XXX is the '
                       + r'nuclear charge. This option is active only in `r'
                       + r'unner_mode` 2 and meant to store  intermediate w'
                       + r'eights in very long epochs.',
        'format': r'write_temporary_weights',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_traincharges': {
        'type': bool,
        'description': r'In runner_mode 2  write the files `traincharges.'
                       + r'YYYYYY.out` and `testcharges.YYYYYY.out` for eac'
                       + r'h epoch. `YYYYYY` is the number of the epoch.  T'
                       + r'he files are written only if the electrostatic N'
                       + r'N is used in case of  electrostatic_type 1.  Thi'
                       + r's can generate many large files and is intended '
                       + r'for a detailed analysis of  the fits.',
        'format': r'write_traincharges',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_trainforces': {
        'type': bool,
        'description': r'In runner_mode 2  write the files  `trainforces.'
                       + r'YYYYYY.out`  and  `testforces.YYYYYY.out`  for e'
                       + r'ach epoch. `YYYYYY` is the number of the epoch. '
                       + r'The files are written only if the short range NN'
                       + r' is used and if the forces are used for training'
                       + r' (keyword  use_short_forces.  This can generate '
                       + r' many large files and is intended for a detailed'
                       + r' analysis of the fits.',
        'format': r'write_trainforces',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_symfunctions': {
        'type': bool,
        'description': r'In runner_mode  3 write the file `symfunctions.o'
                       + r'ut` containing  the unscaled and non-centered sy'
                       + r'mmetry functions default_values of all atoms in '
                       + r'the  predicted structure. The format is the same'
                       + r' as for the files  function.data and  testing.da'
                       + r'ta with the exception that no energies are given'
                       + r'. The purpose of this file is code debugging.',
        'format': r'write_symfunctions',
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_binding_energy_only': {
        'type': bool,
        'description': r'In  runner_mode 2  write only the binding energy'
                       + r' instead of total energies in  the files  trainp'
                       + r'oints.XXXXXX.out and testpoints.XXXXXX.out for e'
                       + r'ach epoch.',
        'format': r'write_binding_energy_only',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_trainpoints': {
        'type': bool,
        'description': r'In  runner_mode 2  write the files  trainpoints.'
                       + r'XXXXXX.out and testpoints.XXXXXX.out for each ep'
                       + r'och. `XXXXXX` is the number of the epoch. The fi'
                       + r'les are written only if the short range NN is us'
                       + r'ed. This can generate many large files and is in'
                       + r'tended for a detailed analysis of the fits.',
        'format': r'write_trainpoints',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_unformatted': {
        'type': bool,
        'description': r'Write output without any formatting.',
        'format': r'write_unformatted',
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_weights_epoch': {
        'description': r'Determine in which epochs the files `YYYYYY.shor'
                       + r't.XXX.out` and  `YYYYYY.elec.XXX.out` are writte'
                       + r'n. ',
        'format': r'write_weights_epoch i1',
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i1': {
                'description': r'Frequency of weight output.',
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
}
