# ---------- The full monty - All RuNNer options with description etc. --------#
RUNNERCONFIG_DEFAULTS: dict = {
    'analyze_composition': {
        'type': bool,
        'description': " Print detailed information about the element composition of the data set in  `input.data`.",
        'format': "analyze_composition",
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
        'description': " Print detailed information about the training error.",
        'format': "analyze_error",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'analyze_error_charge_step': {
        'description': " When a detailed analysis of the training error with  analyze_error is performed, this keyword allows for the definition of the interval in which  atoms with the same charge error are grouped together.",
        'format': "analyze_error_charge_step a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The interval in which atoms with the same charge error are grouped together (unit: electron charge).",
                'type': float,
                'default_value': 0.001,
            },
        },
        'allow_multiple': False,
    },
    'analyze_error_energy_step': {
        'description': " When a detailed analysis of the training error with analyze_error is performed, this keyword allows for the definition of the interval in which  atoms with the same energy error are grouped together.",
        'format': "analyze_error_energy_step a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The interval in which atoms with the same energy error are grouped together (unit: Hartree).",
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'analyze_error_force_step': {
        'description': " When a detailed analysis of the training error with analyze_error is performed, this keyword allows for the definition of the interval in which  atoms with the same total force error are grouped together.",
        'format': "analyze_error_force_step a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The interval in which atoms with the same force error are grouped together (unit: Hartree per Bohr).",
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'atom_energy': {
        'description': " Specification of the energies of the free atoms. This keyword must be used for each element if the keyword  remove_atom_energies is used.  In runner_mode 1 the atomic energies are removed from the total energies, in  runner_mode 3 the atomic energies are added to the fitted energy to yield the correct total energy. Internally, `RuNNer` always works with binding energies, if  remove_atom_energies is specified.",
        'format': "atom_energy element energy",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " Element symbol.",
                'type': str,
                'default_value': None,
            },
            'energy': {
                'description': " Atomic reference energy in Hartree.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': True,
    },
    'biasweights_max': {
        'description': " `RuNNer` allows for a separate random initialization of the bias weights at the beginning of  runner_mode 2 through separate_bias_ini_short. In that case the bias weights are randomly initialized on an interval between  biasweights_max  and biasweights_min. ",
        'format': "biasweights_max a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The maximum default_value that is assigned to bias weights during initialization of the weights.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'biasweights_min': {
        'description': " `RuNNer` allows for a separate random initialization of the bias weights at the beginning of  runner_mode 2 through separate_bias_ini_short. In that case the bias weights are randomly initialized on an interval between  biasweights_max  and biasweights_min.",
        'format': "biasweights_min a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The maximum default_value that is assigned to bias weights during initialization of the weights.",
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'bond_threshold': {
        'description': " Threshold for the shortest bond in the structure in Bohr units. If a shorter bond occurs `RuNNer` will stop with an error message in  runner_mode 2 and 3. In  runner_mode 1 the  structure will be eliminated from the data set.",
        'format': "bond_threshold a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " The minimum bond length between any two atoms in the structure (unit: Bohr).",
                'type': float,
                'default_value': 0.5,
            },
        },
        'allow_multiple': False,
    },
    'calculate_final_force': {
        'type': bool,
        'description': " Print detailed information about the forces in the training and testing set at  the end of the NNP training process. ",
        'format': "calculate_final_force",
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
        'description': " Calculate the atomic forces in  runner_mode 3 and write them to the files  runner.out nnforces.out",
        'format': "calculate_forces",
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
        'description': " Calculate the Hessian matrix in  runner_mode 3.  <!-- The implementation is currently in progress and the keyword is not yet ready for use. -->",
        'format': "calculate_hessian",
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
        'description': " Calculate the stress tensor (only for periodic systems) in  runner_mode 3 and  write it to the files runner.out nnstress.out This is at the moment only implemented for the short range part and for the contributions to the stress tensor through vdW interactions.",
        'format': "calculate_stress",
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
        'description': " Shift the symmetry function default_values individually for each symmetry function such that the average is moved to zero. This may have numerical advantages, because  zero is the center of the non-linear regions of most activation functions.",
        'format': "center_symmetry_functions",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'charge_error_threshold': {
        'description': " Threshold default_value for the error of the charges in units of the RMSE of the previous epoch. A default_value of 0.3 means that only charges with an error larger than 0.3RMSE will be used for the weight update. Large default_values (about 1.0) will speed up the first epochs, because only a few points will be used. ",
        'format': "charge_error_threshold a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of charge RMSE that a charge needs to reach to be used in the weight update.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'charge_fraction': {
        'description': " Defines the random fraction of atomic charges used for fitting the electrostatic weights in  runner_mode 2.",
        'format': "charge_fraction a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of atomic charges used for fitting of the electrostatic weights. 100% = 1.0.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'charge_group': {
        'description': " Do not update the electrostatic NN weights after the presentation of an  individual atomic charge, but average the derivatives with respect to the  weights over the specified number of charges for each element.",
        'format': "charge_group i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Number of atomic charges per group. The maximum is given by points_in_memory.",
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
    'check_forces': {
        'type': bool,
        'description': " This keyword allows to check if the sum of all NN force vectors is zero, It is for debugging purposes only, but does not cost much CPU time.",
        'format': "check_forces",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'check_input_forces': {
        'description': " Check, if the sum of all forces of the training structures is sufficiently close to the zero vector.",
        'format': "check_input_forces a0",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Threshold for the absolute default_value of the sum of all force vectors per atom.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'cutoff_type': {
        'description': " This keyword determines the cutoff function to be used for the symmetry functions.",
        'format': "cutoff_type i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Threshold for the absolute default_value of the sum of all force vectors per atom.",
                'type': int,
                'default_value': 1,
                'options': {
                    0: {
                        'description': " Hard function: $1$",
                    },
                    1: {
                        'description': " Cosine function: $\frac{1}{2}[\cos(\pi x)+ 1]$",
                    },
                    2: {
                        'description': " Hyperbolic tangent function 1: $\tanh^{3} (1-\frac{R_{ij}}{R_{\mathrm{c}}})$",
                    },
                    3: {
                        'description': " Hyperbolic tangent function 2: $(\frac{e+1}{e-1})^3 \tanh^{3}(1-\frac{R_{ij}}{R_{\mathrm{c}}})$",
                    },
                    4: {
                        'description': " Exponential function: $\exp(1-\frac{1}{1-x^2})$",
                    },
                    5: {
                        'description': " Polynomial function 1: $(2x -3)x^2+1$",
                    },
                    6: {
                        'description': " Polynomial function 2: $((15-6x)x-10)x^3+1$",
                    },
                    7: {
                        'description': " Polynomial function 3: $(x(x(20x-70)+84)-35)x^4+1$",
                    },
                    8: {
                        'description': " Polynomial function 4: $(x(x(x(315-70x)-540)+420)-126)x^5+1$",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'data_clustering': {
        'description': " Performs an analysis of all symmetry function vectors of all atoms and groups the atomic environments to clusters with a maximum distance of  `a0` between the symmetry function vectors. If  `a1` is larger than 1.0 the assignment of each atom will be  printed.",
        'format': "data_clustering a0 a1",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Maximum distance between the symmetry function vectors of two clusters of atomic environments.",
                'type': float,
                'default_value': 1.0,
            },
            'a1': {
                'description': " If `a1 > 1.0`, print the group that each atom has been assigned to.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'debug_mode': {
        'type': bool,
        'description': " If switched on, this option can produce a lot of output and is meant for debugging new developments only!!!",
        'format': "debug_mode",
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
        'description': " Write detailed timing information for the individual parts of `RuNNer` at the end of the run. This feature has to be used with some care because often the implementation of the time measurement lacks behind in code development.",
        'format': "detailed_timing",
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
        'description': " Write detailed timing information in each epoch in  runner_mode 2.  This feature has to be used with some care because often the implementation of the time measurement lacks behind in code development.",
        'format': "detailed_timing_epoch",
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
        'description': " For each training epoch, checks whether the default_value of a node in any hidden layer  exceeds  saturation_threshold and prints a warning.",
        'format': "detect_saturation",
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
        'description': " Do not update the short-range NN weights after the presentation of an  individual atomic force vector, but average the derivatives with respect to the  weights over the number of force vectors for each element specified by short_force_group",
        'format': "dynamic_force_grouping",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'electrostatic_type': {
        'description': " This keyword determines the cutoff function to be used for the symmetry functions.",
        'format': "electrostatic_type i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Threshold for the absolute default_value of the sum of all force vectors per atom.",
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': " There is a separate set of atomic NNs to fit the atomic charges as a function of the chemical environment.",
                    },
                    2: {
                        'description': " The atomic charges are obtained as a second output node of the short range atomic NNs. **This is not yet implemented.**",
                    },
                    3: {
                        'description': " Element-specific fixed charges are used that are specified in the input.nn file by the keyword fixed_charge.",
                    },
                    4: {
                        'description': " The charges are fixed but can be different for each atom in the system. They are specified in the file `charges.in`.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'element_activation_electrostatic': {
        'description': " Set the activation function for a specified node of a specified element in the electrostatic NN. The default is set by the keyword global_activation_electrostatic.",
        'format': "element_activation_electrostatic element layer node type",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element whose atomic NN the activation function shall be applied to.",
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': " The number of the layer of the target node.",
                'type': int,
                'default_value': None,
            },
            'node': {
                'description': " The number of the target node in layer `layer`.",
                'type': int,
                'default_value': None,
            },
            'type': {
                'description': " The kind of activation function. Options are listed under global_activation_short.",
                'type': str,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_activation_pair': {
        'description': " Set the activation function for a specified node of a specified element pair in  the pairwise NN. The default is set by the keyword global_activation_pair.",
        'format': "element_activation_pair element element layer node type",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the second element in the pair whose short-range pair NN the activation function shall be applied to.",
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': " The number of the layer of the target node.",
                'type': int,
                'default_value': None,
            },
            'node': {
                'description': " The number of the target node in layer `layer`.",
                'type': int,
                'default_value': None,
            },
            'type': {
                'description': " The kind of activation function. Options are listed under global_activation_short.",
                'type': str,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_activation_short': {
        'description': " Set the activation function for a specified node of a specified element in the short range NN. The default is set by the keyword global_activation_short.",
        'format': "element_activation_short element layer node type",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element whose atomic NN the activation function shall be applied to.",
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': " The number of the layer of the target node.",
                'type': int,
                'default_value': None,
            },
            'node': {
                'description': " The number of the target node in layer `layer`.",
                'type': int,
                'default_value': None,
            },
            'type': {
                'description': " The kind of activation function. Options are listed under global_activation_short.",
                'type': str,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_decoupled_forces_v2': {
        'type': bool,
        'description': " This is a more sophisticated version of the element decoupled Kalman filter for force fitting (switched on by the keyword element_decoupled_kalman.",
        'format': "element_decoupled_forces_v2",
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
        'description': " Use the element decoupled Kalman filter for the short range energy and force update (if use_short_forces is switched on). This is implemented only for the atomic energy case. A more sophisticated algorithm for the force fitting can be activated by using additionally the keyword  element_decoupled_forces_v2.  One important parameter for force fitting is  force_update_scaling,  which determines the magnitude of the force update compared to the energy update. Usually 1.0 is a good default default_value.",
        'format': "element_decoupled_kalman",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'element_hidden_layers_electrostatic': {
        'description': " Overwrite the global default number of hidden layers given by global_hidden_layers_electrostatic for a specific element. Just a reduction of the number of hidden layers is possible. .",
        'format': "element_hidden_layers_electrostatic element layers",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element whose atomic NN hidden layer number will be set.",
                'type': str,
                'default_value': None,
            },
            'layers': {
                'description': " The number of hidden layers for this element.",
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_hidden_layers_pair': {
        'description': " Overwrite the global default number of hidden layers given by global_hidden_layers_pair for a specific element. Just a reduction of the number of hidden layers is possible. .",
        'format': "element_hidden_layers_pair element element layers",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element1': {
                'description': " The periodic table symbol of the first pair element whose short-range pair NN hidden layer number will be set.",
                'type': str,
                'default_value': None,
            },
            'element2': {
                'description': " The periodic table symbol of the second pair element whose short-range pair NN hidden layer number will be set.",
                'type': str,
                'default_value': None,
            },
            'layers': {
                'description': " The number of hidden layers for this element pair.",
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_hidden_layers_short': {
        'description': " Overwrite the global default number of hidden layers given by global_hidden_layers_short for a specific element. Just a reduction of the number of hidden layers is possible.",
        'format': "element_hidden_layers_short element layers",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element whose atomic NN hidden layer number will be set.",
                'type': str,
                'default_value': None,
            },
            'layers': {
                'description': " The number of hidden layers for this element.",
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_nodes_electrostatic': {
        'description': " Overwrite the global default number of nodes in the specified hidden layer of an elecrostatic NN given by  global_nodes_electrostatic  for a specific element.  Just a reduction of the number of nodes is possible. ",
        'format': "element_nodes_electrostatic element layer i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element.",
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': " The number of the hidden layer for which the number of nodes is set.",
                'type': int,
                'default_value': None,
            },
            'i0': {
                'description': " The number of nodes to be set.",
                'type': int,
                'default_value': "global_nodes_electrostatic",
            },
        },
        'allow_multiple': True,
    },
    'element_nodes_pair': {
        'description': " Overwrite the global default number of nodes in the specified hidden layer of a pair NN given by  global_nodes_pair  for a specific element.  Just a reduction of the number of nodes is possible. ",
        'format': "element_nodes_pair element element layer i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the second pair element.",
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': " The number of the hidden layer for which the number of nodes is set.",
                'type': int,
                'default_value': None,
            },
            'i0': {
                'description': " The number of nodes to be set.",
                'type': int,
                'default_value': "global_nodes_pair",
            },
        },
        'allow_multiple': True,
    },
    'element_nodes_short': {
        'description': " Overwrite the global default number of nodes in the specified hidden layer of an short-range atomic NN given by  global_nodes_short  for a specific element.  Just a reduction of the number of nodes is possible. ",
        'format': "element_nodes_short element layer i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element.",
                'type': str,
                'default_value': None,
            },
            'layer': {
                'description': " The number of the hidden layer for which the number of nodes is set.",
                'type': int,
                'default_value': None,
            },
            'i0': {
                'description': " The number of nodes to be set.",
                'type': int,
                'default_value': "global_nodes_short",
            },
        },
        'allow_multiple': True,
    },
    'element_pairsymfunction_short': {
        'description': " Set the symmetry functions for one element pair for the short-range pair NN.",
        'format': "element_pairsymfunction_short element element type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element1': {
                'description': " The periodic table symbol of the first pair element.",
                'type': str,
                'default_value': None,
            },
            'element2': {
                'description': " The periodic table symbol of the second pair element.",
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires parameters `eta` and `rshift`. ```runner-config element_pairsymfunction_short element element 2 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`, `lambda`, and `zeta`. ```runner-config element_pairsymfunction_short element element 3 eta lambda zeta cutoff ```",
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
                        'description': " Radial function. Requires parameter `eta`. ```runner-config element_pairsymfunction_short element element 4 eta cutoff ```",
                        'parameters': {
                            'element': {
                                'type': str,
                                'default_value': None
                            },
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config element_pairsymfunction_short element element 5 eta ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config element_pairsymfunction_short element element 6 cutoff ```",                
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires `parameters` `eta`, and `rshift`. ```runner-config element_pairsymfunction_short element element 8 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`. ```runner-config element_pairsymfunction_short element element 9 eta cutoff ```.",
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
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'elements': {
        'description': " The element symbols of all elements in the system in arbitrary order.  The number of specified elements must fit to the default_value of the keyword number_of_elements.",
        'format': "elements element [element...]",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element [element...]': {
                'description': " The periodic table symbols of all the element in the system.",
                'type': str,
                'default_value': [None],
                'allow_multiple': True
            }
        },
        'allow_multiple': False,
    },
    'element_symfunction_electrostatic': {
        'description': " Set the symmetry functions for one element with all possible neighbor element combinations for the electrostatics NN. The variables are the same as for the  keyword  global_symfunction_electrostatic and are explained in more detail there.",
        'format': "element_symfunction_electrostatic element type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element.",
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires parameters `eta` and `rshift`. ```runner-config element_symfunction_electrostatic element 2 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`, `lambda`, and `zeta`. ```runner-config element_symfunction_electrostatic element 3 eta lambda zeta cutoff ```",
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
                        'description': " Radial function. Requires parameter `eta`. ```runner-config element_symfunction_electrostatic element 4 eta cutoff ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config element_symfunction_electrostatic element 5 eta ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config element_pairsymfunction_short element 6 cutoff ```",             
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires `parameters` `eta`, and `rshift`. ```runner-config element_pairsymfunction_short element 8 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`. ```runner-config element_pairsymfunction_short element 9 eta cutoff ```.",
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
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'element_symfunction_short': {
        'description': " Set the symmetry functions for one element with all possible neighbor element combinations for the short-range NN. The variables are the same as for the  keyword  global_symfunction_short and are explained in more detail there.",
        'format': "element_symfunction_short element type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element.",
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires parameters `eta` and `rshift`. ```runner-config element_symfunction_short element 2 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`, `lambda`, and `zeta`. ```runner-config element_symfunction_short element 3 eta lambda zeta cutoff ```",
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
                        'description': " Radial function. Requires parameter `eta`. ```runner-config element_symfunction_short element 4 eta cutoff ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config element_symfunction_short element 5 eta ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config element_pairsymfunction_short element 6 cutoff ```",             
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires `parameters` `eta`, and `rshift`. ```runner-config element_pairsymfunction_short element 8 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`. ```runner-config element_pairsymfunction_short element 9 eta cutoff ```.",
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
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True
    },
    'enable_on_the_fly_input': {
        'type': bool,
        'description': " Read modified input.nn the fitting procedure from a file labeled `input.otf`.",
        'format': "enable_on_the_fly_input",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'energy_threshold': {
        'description': " Set an energy threshold for fitting data. This keyword is only used in  runner_mode 1 for the decision if a point should be used in the training or  test set or if it should be eliminated because of its high energy.",
        'format': "energy_threshold a0",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Threshold for the total energy of a structure (unit: Hartree per atom).",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'enforce_max_num_neighbors_atomic': {
        'description': " Set an upper threshold for the number of neighbors an atom can have.",
        'format': "enforce_max_num_neighbors_atomic i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Maximum number of neighbors for one atom.",
                'type': int,
                'default_value': "None",
            },
        },
        'allow_multiple': False,
    },
    'enforce_totcharge': {
        'description': " Rescale the NN atomic charges to get a neutral system. An overall neutral system is required for a correct calculation of the Ewald sum for periodic systems.  The additional error introduced by rescaling the NN charges is typically much smaller than the fitting error, but this should be checked.",
        'format': "enforce_totcharge i0",
        'modes': {
            'mode1': False,
            'mode2': False,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Switch charge rescaling on (`1`) or off (`0`).",
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'environment_analysis': {
        'type': bool,
        'description': " Print a detailed analysis of the atomic environments in  `trainstruct.data` and `teststruct.data`.",
        'format': "environment_analysis",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'epochs': {
        'description': " The number of epochs for fitting. If `0` is specified, `RuNNer` will calculate the error and terminate without adjusting weights.",
        'format': "epochs i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Number of epochs.",
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_alpha': {
        'description': " Parameter $\alpha$ for the Ewald summation. Determines the accuracy of the  electrostatic energy and force evaluation for periodic systems together with  ewald_kmax and  ewald_cutoff. Recommended settings are  (ewald_alpha = 0.2 and  ewald_kmax or (ewald_alpha = 0.5 and  ewald_kmax and a  sufficiently large ewald_cutoff.",
        'format': "ewald_alpha a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " The default_value of the parameter $\alpha$ for the Ewald summation.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_cutoff': {
        'description': " Parameter for the Ewald summation. Determines the accuracy of the  electrostatic energy and force evaluation for periodic systems together with  ewald_kmax and  ewald_alpha. Must be chosen sufficiently large because it determines the number of neighbors taken into account in the real space part of the Ewald summation (e.g. 15.0 Bohr)",
        'format': "ewald_cutoff a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " The cutoff radius if the Ewald summation (unit: Bohr).",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_kmax': {
        'description': " Parameter for the reciprocal space part of the Ewald summation. Determines the accuracy of the electrostatic energy and force evaluation for  periodic systems together with  ewald_alpha and  ewald_cutoff. Recommended settings are  (ewald_alpha = 0.2 and  ewald_kmax or (ewald_alpha = 0.5 and  ewald_kmax and a  sufficiently large ewald_cutoff.",
        'format': "ewald_kmax i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " k-space cutoff for the Ewald summation.",
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'ewald_prec': {
        'description': " This parameter determines the error tolerance in electrostatic energy and force  evaluation for periodic systems when Ewald Summation is used. `RuNNer` will  automatically choose the optimized  ewald_alpha, ewald_kmax, and  ewald_cutoff.",
        'format': "ewald_prec a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " The desired precision of the Ewald summation. Recommended default_values are $10^{-5}$ to $10^{-6}$.",
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'find_contradictions': {
        'description': " This keyword can be used in  runner_mode 2 to test if the symmetry functions are able to distinguish different atomic environments sufficiently. If two atomic environments of a given element are very similar, they will result in very similar symmetry function vectors. Therefore, the length of the difference vector $$ \Delta G = \sqrt{\sum_{i=1}^N (G_{i,1}-G_{i,2})^2} \,,\notag $$ ($N$ runs over all individual symmetry functions) will be close to zero. If the environments are really similar, the absolute forces acting on the atom should be similar as well, which is measured by $$ \begin{align} \Delta F &= |\sqrt{F_{1,x}^2+F_{1,y}^2+F_{1,z}^2}            -\sqrt{F_{2,x}^2+F_{2,y}^2+F_{2,z}^2}|\,,\notag\\          &= |F_1-F_2| \notag\,. \end{align} $$ If the forces are different ($\Delta F >$ `a1`) but the symmetry functions similar ($\Delta G <$ `a0`) for an atom pair, a message will be printed in the output file. The optimal choices for `a0` and `a1` are system dependent and should be selected such that only the most contradictory data is found. It is not recommended to keep this keyword switched on routinely, because it requires substantial CPU time.",
        'format': "find_contradictions a0 a1",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Symmetry function threshold $\Delta G$.",
                'type': float,
                'default_value': 0.0,
            },
            'a1': {
                'description': " Force threshold $\Delta F$.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'fitting_unit': {
        'description': "Set the energy unit that is printed to the output files during training in runner_mode 2.",
        'format': "fitting_unit i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Switch for different energy units.",
                'type': str,
                'default_value': 'eV',
                'options': {
                    'eV': {
                        'description': " Unit: `eV`. The energy RMSE and MAD in the output file are given in eV/atom, the force error is given in eV/Bohr.",
                    },
                    'Ha': {
                        'description': " Unit: `Ha`. The energy RMSE and MAD in the output file are given in Ha/atom, the force error is given in Ha/Bohr.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'fix_weights': {
        'type': bool,
        'description': " Do not optimize all weights, but freeze some weights, which are specified by the keywords  weight_constraint and  weighte_constraint.",
        'format': "fix_weights",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'fixed_charge': {
        'description': " Use a fixed charge for all atoms of the specified element independent of the chemical environment.",
        'format': "fixed_charge element a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element.",
                'type': str,
                'default_value': None,
            },
            'a0': {
                'description': " The fixed charge of all atoms of this element (unit: electron charge).",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'fixed_gausswidth': {
        'description': " This keyword specifies the Gaussian width for calculating the charges and  electrostatic energy and forces in 4G-HDNNPs. ",
        'format': "fixed_gausswidth element a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the element.",
                'type': str,
                'default_value': None,
            },
            'a0': {
                'description': " The Gaussian width for all atoms of this element (unit: Bohr).",
                'type': float,
                'default_value': -99.0,
            },
        },
        'allow_multiple': True,
    },
    'fixed_short_energy_error_threshold': {
        'description': " Only consider points in the weight update during  runner_mode 2 for which the  absolute error of the total energy is higher than  fixed_short_energy_error_threshold.",
        'format': "fixed_short_energy_error_threshold a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The lower threshold for the absolute error of the total energy.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'fixed_short_force_error_threshold': {
        'description': " Only consider points in the weight update during  runner_mode 2 for which the  absolute error of the total force is higher than  fixed_short_force_error_threshold.",
        'format': "fixed_short_force_error_threshold a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The lower threshold for the absolute error of the total force.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'force_grouping_by_structure': {
        'type': bool,
        'description': " Do not update the short-range NN weights after the presentation of an  individual atomic force vector, but average the derivatives with respect to the  weights over the number of force vectors per structure.",
        'format': "force_grouping_by_structure",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'force_threshold': {
        'description': " Set a force threshold for fitting data. If any force component of a structure in the reference set is larger than `a0` then the point is not used and eliminated from the data set. ",
        'format': "force_threshold a0",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The upper threshold for force components (unit: Ha/Bohr)",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'force_update_scaling': {
        'description': " Since for a given structure the number of forces is much larger than the number of energies, the force updates can have a dominant influence on the fits. This can result in poor energy errors. Using this option the relative strength of the energy and the forces can be  adjusted. A default_value of 0.1 means that the influence of the energy is 10 times stronger than of a single force. A negative default_value will automatically balance the strength of the energy and of the forces by taking into account the actual number of atoms of each structures.",
        'format': "force_update_scaling a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The relative strength of the energy and forces for a weight update.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'global_activation_electrostatic': {
        'description': " Set the default activation function for each hidden layer and the output layer in the electrostatic NNs of all elements. ",
        'format': "global_activation_electrostatic type [type...]",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [type...]': {
                'description': " The kind of activation function. One `type` has to be given for each layer in the NN. Options are listed under global_activation_short.",
                'type': str,
                'default_value': [None],
                'allow_multiple': True
            }
        },
        'allow_multiple': False,
    },
    'global_activation_pair': {
        'description': " Set the default activation function for each hidden layer and the output layer in the NNs of all element pairs. ",
        'format': "global_activation_pair type [type...]",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [type...]': {
                'description': " The kind of activation function. One `type` has to be given for each layer in the NN. Options are listed under global_activation_short.",
                'type': str,
                'default_value': [None],
                'allow_multiple': True
            }
        },
        'allow_multiple': False,
    },
    'global_activation_short': {
        'description': " Set the activation function for each hidden layer and the output layer in the short range NNs of all elements. ",
        'format': "global_activation_short type [type...]",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [type...]': {
                'description': " The kind of activation function. One `type` has to be given for each layer in the NN.",
                'type': str,
                'default_value': [None],
                'options': {
                    'c': {
                        'description': " Cosine function: $\cos(x)$",
                    },
                    'e': {
                        'description': " Exponential function: $\exp(-x)$",
                    },
                    'g': {
                        'description': " Gaussian function: $\exp(-\alpha x^2)$",
                    },
                    'h': {
                        'description': " Harmonic function: $x^2$.",
                    },
                    'l': {
                        'description': " Linear function: $x$",
                    },
                    'p': {
                        'description': " Softplus function: $\ln(1+\exp(x))$",
                    },
                    's': {
                        'description': " Sigmoid function v1: $(1-\exp(-x))^{-1}$",
                    },
                    'S': {
                        'description': " Sigmoid function v2: $1-(1-\exp(-x))^{-1}$",
                    },
                    't': {
                        'description': " Hyperbolic tangent function: $\tanh(x)$",
                    },
                },
                'allow_multiple': True
            },
        },
        'allow_multiple': False,
    },
    'global_hidden_layers_electrostatic': {
        'description': " Set the default number of hidden layers in the electrostatic NNs of all  elements. Internally 1 is added to `maxnum_layers_elec`, which also includes  the output layer.",
        'format': "global_hidden_layers_electrostatic layers",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'layers': {
                'description': " The number of hidden layers.",
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_hidden_layers_pair': {
        'description': " Set the default number of hidden layers in the NNs of all element pairs.  Internally 1 is added to `maxnum_layers_short_pair`, which also includes  the output layer.",
        'format': "global_hidden_layers_pair layers",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'layers': {
                'description': " The number of hidden layers.",
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_hidden_layers_short': {
        'description': " Set the default number of hidden layers in the short-range NNs of all  elements. Internally 1 is added to `maxnum_layers_short`, which also includes  the output layer.",
        'format': "global_hidden_layers_short layers",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'layers': {
                'description': " The number of hidden layers.",
                'type': int,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_nodes_electrostatic': {
        'description': " Set the default number of nodes in the hidden layers of the electrostatic NNs in case of  electrostatic_type 1. In the array, the entries `1 - (maxnum_layerseelec - 1)` refer to the hidden  layers. The first entry (0) refers to the nodes in the input layer and is  determined automatically from the symmetry functions.",
        'format': "global_nodes_electrostatic i0 [i1...]",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0 [i1...]': {
                'description': " The number of nodes to be set in each layer.",
                'type': int,
                'default_value': [None],
                'allow_multiple': True
            }
        },
        'allow_multiple': False,
    },
    'global_nodes_pair': {
        'description': " Set the default number of nodes in the hidden layers of the pairwise NNs in case of  nn_type_short 2.",
        'format': "global_nodes_pair i0 [i1...]",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0 [i1...]': {
                'description': " The number of nodes to be set in each layer.",
                'type': int,
                'default_value': [None],
                'allow_multiple': True
            }
        },
        'allow_multiple': False,
    },
    'global_nodes_short': {
        'description': " Set the default number of nodes in the hidden layers of the short-range NNs in case of  nn_type_short 1. In the array, the entries `1 - maxnum_layersshort - 1` refer to the hidden  layers. The first entry (0) refers to the nodes in the input layer and is  determined automatically from the symmetry functions.",
        'format': "global_nodes_short i0 [i1...]",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0 [i1...]': {
                'description': " The number of nodes to be set in each layer.",
                'type': int,
                'default_value': [None],
                'allow_multiple': True
            }
        },
        'allow_multiple': False,
    },
    'global_pairsymfunction_short': {
        'description': " Specification of the global symmetry functions for all element pairs in the  pairwise NN.",
        'format': "global_pairsymfunction_short type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires parameters `eta` and `rshift`. ```runner-config global_pairsymfunction_short 2 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`, `lambda`, and `zeta`. ```runner-config global_pairsymfunction_short 3 eta lambda zeta cutoff ```",
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
                        'description': " Radial function. Requires parameter `eta`. ```runner-config global_pairsymfunction_short 4 eta cutoff ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    5: {
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config global_pairsymfunction_short 5 eta ```",
                        'parameters': {
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config global_pairsymfunction_short 6 cutoff ```",                  
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires `parameters` `eta`, and `rshift`. ```runner-config global_pairsymfunction_short 8 eta rshift cutoff ```",
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
                        'description': " Angular function. Requires `parameters` `eta`. ```runner-config global_pairsymfunction_short 9 eta cutoff ```.",
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
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_symfunction_electrostatic': {
        'description': " Specification of global symmetry functions for all elements and all element  combinations for the electrostatic NN.",
        'format': "global_symfunction_electrostatic type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires parameters `eta` and `rshift`. ```runner-config global_symfunction_electrostatic 2 eta rshift cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0
                        },
                    },
                    3: {
                        'description': " Angular function. Requires `parameters` `eta`, `lambda`, and `zeta`. ```runner-config global_symfunction_electrostatic 3 eta lambda zeta cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                            'lambda': 0.0,
                            'zeta': 0.0
                        },
                    },
                    4: {
                        'description': " Radial function. Requires parameter `eta`. ```runner-config global_symfunction_electrostatic 2 eta cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    5: {
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config global_symfunction_electrostatic 5 eta ```",
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config global_symfunction_electrostatic 6 cutoff ```",
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires `parameters` `eta`, and `rshift`. ```runner-config global_symfunction_electrostatic 8 eta rshift cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0
                        },
                    },
                    9: {
                        'description': " Angular function. Requires `parameters` `eta`. ```runner-config global_symfunction_electrostatic element 9 eta cutoff ```.",
                        'parameters': {
                            'eta': 0.0
                        },
                    },
                },
            },
            'cutoff': {
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'global_symfunction_short': {
        'description': " Specification of global symmetry functions for all elements and all element  combinations for the short-range atomic NN.",
        'format': "global_symfunction_short type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`.",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires parameters `eta` and `rshift`. ```runner-config global_symfunction_short 2 eta rshift cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0
                        },
                    },
                    3: {
                        'description': " Angular function. Requires `parameters` `eta`, `lambda`, and `zeta`. ```runner-config global_symfunction_short 3 eta lambda zeta cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                            'lambda': 0.0,
                            'zeta': 0.0
                        },
                    },
                    4: {
                        'description': " Radial function. Requires parameter `eta`. ```runner-config global_symfunction_short 2 eta cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    5: {
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config global_symfunction_short 5 eta ```",
                        'parameters': {
                            'eta': 0.0,
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config global_symfunction_short 6 cutoff ```",
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires `parameters` `eta`, and `rshift`. ```runner-config global_symfunction_short 8 eta rshift cutoff ```",
                        'parameters': {
                            'eta': 0.0,
                            'rshift': 0.0
                        },
                    },
                    9: {
                        'description': " Angular function. Requires `parameters` `eta`. ```runner-config global_symfunction_short 9 eta cutoff ```.",
                        'parameters': {
                            'eta': 0.0
                        },
                    },
                },
            },
            'cutoff': {
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'growth_mode': {
        'description': " If this keyword is used, not the full training set will be used in each epoch.  First, only a few points will be used, and after a specified number of epochs further points will be included and so on. ",
        'format': "growth_mode i0 i1",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Number of points that will be added to the training set every `i1` steps.",
                'type': int,
                'default_value': 0,
            },
            'i1': {
                'description': " Number of steps to wait before increasing the number of training points.",
                'type': int,
                'default_value': 0,
            },
        },
        'allow_multiple': False,
    },
    'initialization_only': {
        'type': bool,
        'description': " With this keyword, which is active only in  runner_mode 2, `RuNNer` will stop  after the initialization of the run before epoch 0, i.e. no fit will be done.  This is meant as an automatic stop of the program in case only the analysis carried out in the initialization of  runner_mode 2  is of interest.",
        'format': "initialization_only",
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
        'description': " If this keyword is set, for structures with a nonzero net charge only the forces will be used for fitting, the energies will be omitted. This keyword is  currently implemented only for the atomic short range part.",
        'format': "ion_forces_only",
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
        'description': " This is an experimental keyword not fully tested. For each atom only one weight update is done for an averaged set of gradients calculated from the energy and all forces (not yet working well).",
        'format': "joint_energy_force_update",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'kalman_damp_charge': {
        'description': " Reduce the effective RMSE on the charges for the Kalman filter update of the weights in the electrostatic NN.",
        'format': "kalman_damp_charge a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of charge RMSE that is considered for the weight update. 100% = 1.0.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_damp_force': {
        'description': " Reduce the effective RMSE on the forces for the Kalman filter update of the weights in the short-range NN.",
        'format': "kalman_damp_force a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of force RMSE that is considered for the weight update. 100% = 1.0.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_damp_short': {
        'description': " Reduce the effective RMSE on the energies for the Kalman filter update of the weights in the short-range NN.",
        'format': "kalman_damp_short a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of energy RMSE that is considered for the weight update. 100% = 1.0.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_epsilon': {
        'description': " Set the initialization parameter for the correlation matrix of the Kalman filter according to $$ P(0)=\epsilon^{-1} \mathcal{I}. $$ $\epsilon$ is often set to the order of $10^{-3}$ to $10^{-2}$.",
        'format': "kalman_epsilon a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $\epsilon$.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_lambda_charge': {
        'description': " Kalman filter parameter $\lambda$ for the electrostatic NN weight updates.",
        'format': "kalman_lambda_charge a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $\lambda$.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_lambda_short': {
        'description': " Kalman filter parameter $\lambda$ for the short range NN weight updates.",
        'format': "kalman_lambda_short a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $\lambda$.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_nue_charge': {
        'description': " Kalman filter parameter $\lambda_0$ for the electrostatic NN weight updates.",
        'format': "kalman_nue_charge a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $\lambda_0$.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'kalman_nue_short': {
        'description': " Kalman filter parameter $\lambda_0$ for the short range weight updates.",
        'format': "kalman_nue_short a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $\lambda_0$.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'kalman_q0': {
        'description': " It is possible to add artificial process noise for the Kalman filter in the form of  $$ Q(t) =q(t)\mathcal{I}, $$  with either a fixed $q(t)=q(0)$ or annealing from a higher $q(0)$ to  $q_{\mathrm{min}}$ following a scheme like $$ q(t) = \max(q_{0}e^{-t/\tau_{q}}, q_{\mathrm{min}}). $$ The default_value of $q(0)$ is usually set between $10^{-6}$ and $10^{-2}$. It is recommended for the user to do some test for each new system, altering kalman_q0,  kalman_qmin and  kalman_qtau to obtain the optimal performance for minimizing the root mean square error.",
        'format': "kalman_q0 a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $q(0)$.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_qmin': {
        'description': " Parameter $q_{\mathrm{min}}$ for adding artificial process noise to the Kalman filter noise matrix. See kalman_q0 for a more detailed explanation.",
        'format': "kalman_qmin a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $q_{\mathrm{min}}$.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'kalman_qtau': {
        'description': " Parameter $\tau_q$ for adding artificial process noise to the Kalman filter noise matrix. See kalman_q0 for a more detailed explanation.",
        'format': "kalman_qtau a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " default_value of $\tau_q$.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'max_energy': {
        'description': " Set an upper threshold for the consideration of a structure during the weight  update. If the total energy is above  max_energy] the data point will be ignored.",
        'format': "max_energy a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Maximum energy of a structure to be considered for the weight update (unit: Hartree).",
                'type': float,
                'default_value': 10000.0,
            },
        },
        'allow_multiple': False,
    },
    'max_force': {
        'description': " Set an upper threshold for the consideration of a structure during the weight  update. If any force component is above  max_force] the data point will be ignored.",
        'format': "max_force a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Maximum force component of a structure to be considered for the weight update (unit: Hartree/Bohr).",
                'type': float,
                'default_value': 10000.0,
            },
        },
        'allow_multiple': False,
    },
    'md_mode': {
        'type': bool,
        'description': " The purpose of this keyword is to reduce the output to enable the incorporation of `RuNNer` into a MD code.",
        'format': "md_mode",
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
        'description': " Randomly reorder the data points in the data set at the beginning of each new epoch.",
        'format': "mix_all_points",
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
        'description': " Initialize the elecrostatic NN weights according to the scheme proposed by Nguyen and Widrow. The initial weights and bias default_values in the hidden layer are chosen such that the input space is evenly distributed over the nodes. This may speed up the training process.",
        'format': "nguyen_widrow_weights_ewald",
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
        'description': " Initialize the short-range NN weights according to the scheme proposed by Nguyen and Widrow. The initial weights and bias default_values in the hidden layer are chosen such that the input space is evenly distributed over the nodes. This may speed up the training process.",
        'format': "nguyen_widrow_weights_short",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'nn_type_short': {
        'description': " Specify the NN type of the short-range part.",
        'format': "nn_type_short i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Set the short-range NN type.",
                'type': int,
                'default_value': None,
                'options': {
                    1: {
                        'description': " Behler-Parrinello atomic NNs. The short range energy is constructed as a sum of environment-dependent atomic energies.",
                    },
                    2: {
                        'description': " Pair NNs. The short range energy is constructed as a sum of environment-dependent pair energies.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'nnp_gen': {
        'description': " This keyword specifies the generation of HDNNP that will be constructed.",
        'format': "nnp_gen i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Set the short-range and electrostatics NN type.",
                'type': int,
                'default_value': None,
                'options': {
                    2: {
                        'description': " 2G-HDNNPs only include the short-range part (Behler-Parrinello atomic NNs). Users should also specify use_short_nn.",
                    },
                    3: {
                        'description': " 3G-HDNNPs include both the short-range part and the long-range electrostatic part. Users are advised to first construct a representation for the electrostatic part by specifying use_electrostatics and then switch to the short range part by setting both use_short_nn and use_electrostatics.",
                    },
                    4: {
                        'description': " 4G-HDNNPs include both the short-range part and the long-range electrostatic part. Users are advised to first construct a representation for the electrostatic part by specifying use_electrostatics and then switch to the short range part by setting both use_short_nn and use_electrostatics.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'noise_charge': {
        'description': " Introduce artificial noise on the atomic charges in the training process by setting a lower threshold that the absolute charge error of a data point has to  surpass before being considered for the weight update.   ",
        'format': "noise_charge a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Noise charge threshold (unit: Hartree per atom). Must be positive.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'noise_energy': {
        'description': " Introduce artificial noise on the atomic energies in the training process by setting a lower threshold that the absolute energy error of a data point has to  surpass before being considered for the weight update.   ",
        'format': "noise_energy a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Noise energy threshold (unit: electron charge). Must be positive.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'noise_force': {
        'description': " Introduce artificial noise on the atomic forces in the training process by setting a lower threshold that the absolute force error of a data point has to  surpass before being considered for the weight update.   ",
        'format': "noise_force a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Noise force threshold (unit: Hartree per Bohr). Must be positive.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'normalize_nodes': {
        'type': bool,
        'description': " Divide the accumulated sum at each node by the number of nodes in the previous layer before the activation function is applied. This may help to activate the activation functions in their non-linear regions.",
        'format': "normalize_nodes",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'number_of_elements': {
        'description': " Specify the number of chemical elements in the system.",
        'format': "number_of_elements i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Number of elements.",
                'type': int,
                'default_value': "None",
            },
        },
        'allow_multiple': False,
    },
    'optmode_charge': {
        'description': " Specify the optimization algorithm for the atomic charges in case of  electrostatic_type 1.",
        'format': "optmode_charge i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Set the atomic charge optimization algorithm.",
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': " Kalman filter.",
                    },
                    2: {
                        'description': " Reserved for conjugate gradient, not implemented.",
                    },
                    3: {
                        'description': " Steepest descent. Not recommended.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'optmode_short_energy': {
        'description': " Specify the optimization algorithm for the short-range energy contributions.",
        'format': "optmode_short_energy i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Set the short-range energy optimization algorithm.",
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': " Kalman filter.",
                    },
                    2: {
                        'description': " Reserved for conjugate gradient, not implemented.",
                    },
                    3: {
                        'description': " Steepest descent. Not recommended.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'optmode_short_force': {
        'description': " Specify the optimization algorithm for the short-range forces.",
        'format': "optmode_short_force i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Set the short-range force optimization algorithm.",
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': " Kalman filter.",
                    },
                    2: {
                        'description': " Reserved for conjugate gradient, not implemented.",
                    },
                    3: {
                        'description': " Steepest descent. Not recommended.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'parallel_mode': {
        'description': " This flag controls the parallelization of some subroutines. ",
        'format': "parallel_mode i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Set the short-range force optimization algorithm.",
                'type': int,
                'default_value': 1,
                'options': {
                    1: {
                        'description': " Serial version.",
                    },
                    2: {
                        'description': " Parallel version.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'points_in_memory': {
        'description': " This keyword controls memory consumption and IO and is therefore important to achieve an optimum performance of `RuNNer`. Has a different meaning depending on the current  runner_mode.",
        'format': "points_in_memory i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " In runner_mode 1 this is the maximum number of structures in memory at a time. In runner_mode 3 this is the number of atoms for which the symmetry functions are in memory at once. In parallel runs these atoms are further split between the processes.",
                'type': int,
                'default_value': 200,
            },
        },
        'allow_multiple': False,
    },
    'precondition_weights': {
        'type': bool,
        'description': " Shift the weights of the atomic NNs right after the initialization so that  the standard deviation of the NN energies is the same as the standard deviation of the reference energies.",
        'format': "precondition_weights",
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
        'description': " For debugging only. Prints the derivatives of the short range energy with  respect to the short range NN weight parameters after each update. This derivative array is responsible for the weight update. The derivatives (the array `deshortdw`) are written to the file `debug.out`.",
        'format': "print_all_deshortdw",
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
        'description': " For debugging only. Prints the derivatives of the short range forces with respect to the short range NN weight parameters after each update. This derivative array is responsible for the weight update. The derivatives (the array `dfshortdw(maxnum_weightsshort)`) are written to the file `debug.out`.",
        'format': "print_all_dfshortdw",
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
        'description': " For debugging only. Print the electrostatic NN weight parameters after each  update, not only once per epoch to a file. The weights (the array  `weights_ewald()`) are written to the file `debug.out`.",
        'format': "print_all_electrostatic_weights",
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
        'description': " For debugging only. Print the short range NN weight parameters after each  update, not only once per epoch to a file. The weights (the array `weights_short()`) are written to the file `debug.out`. ",
        'format': "print_all_short_weights",
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
        'description': " During training, print a measure for the convergence of the weight vector. The output is: ```runner-data CONVVEC element epoch C1 C2 wshift wshift2 ``` `C1` and `C2` are two-dimensional coordinates of projections of the weight vectors for plotting qualitatively the convergence of the weights. `wshift` is the length (normalized by the number of weights) of the difference vector of the weights between two epochs. `wshift2` is the length (normalized by the number of weights) of the difference vector between the current weights and the weights two epochs ago.",
        'format': "print_convergence_vector",
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
        'description': " Print in each training epoch the date and the real time in an extra line.",
        'format': "print_date_and_time",
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
        'description': " For debugging only. Prints in  runner_mode 3  the contributions of all atomic energies to the force components of each atom.",
        'format': "print_force_components",
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
        'description': " Print a line with the mean absolute deviation as an additional output line next to the RMSE in  runner_mode 2.  Usually the MAD is smaller than the RMSE as outliers do not have such a large  impact.",
        'format': "print_mad",
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
        'description': " Perform sensitivity analysis on the symmetry functions of the neural network. The  sensitivity is a measure of how much the NN output changes with the symmetry functions, i.e. the derivative. It will be analyzed upon weight initialization and for each training epoch in all short-range, pair, and electrostatic NNs there are. ",
        'format': "print_sensitivity",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'random_number_type': {
        'description': " Specify the type of random number generator used in `RuNNer`. The seed can be given with the keyword  random_seed.",
        'format': "random_number_type i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Set the random number generator type.",
                'type': int,
                'default_value': 5,
                'options': {
                    1: {
                        'description': " Deprecated.",
                    },
                    2: {
                        'description': " Deprecated.",
                    },
                    3: {
                        'description': " Deprecated.",
                    },
                    4: {
                        'description': " Deprecated.",
                    },
                    5: {
                        'description': " Normal distribution of random numbers.",
                    },
                    6: {
                        'description': " Normal distribution of random numbers with the `xorshift` algorithm. **This is the recommended option.**",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'random_seed': {
        'description': " Set the integer seed for the random number generator used at many places in  `RuNNer`. In order to ensure that all results are reproducible, the same seed will result in exactly the same output at all times (machine and compiler dependence cannot be excluded). This seed default_value is used for all random number generator in `RuNNer`, but internally for each purpose a local copy is made first to avoid interactions between the different random number generators.  Please see also the keyword  random_number_type.",
        'format': "random_seed i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Seed default_value.",
                'type': int,
                'default_value': 200,
            },
        },
        'allow_multiple': False,
    },
    'read_kalman_matrices': {
        'type': bool,
        'description': " Restart a fit using old Kalman filter matrices from the files `kalman.short.XXX.data` and `kalman.elec.XXX.data`. `XXX` is the nuclear charge of the respective element. Using old Kalman matrices will reduce the oscillations of the errors when a fit is restarted with the Kalman filter. The Kalman matrices are written to the files using the keyword save_kalman_matrices",
        'format': "read_kalman_matrices",
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
        'description': " Read old NN weights and/or an old Kalman matrix from an unformatted input file.",
        'format': "read_unformatted",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'regularize_fit_param': {
        'description': " This keyword switches on L2 regularization, mainly for the electrostatic part in 4G-HDNNPs.",
        'format': "regularize_fit_param a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Regularization parameter. Recommended setting is $10^{-6}$.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'remove_atom_energies': {
        'type': bool,
        'description': " Remove the energies of the free atoms from the total energies per atom to reduce the absolute default_values of the target energies. This means that when this keyword is used, `RuNNer` will fit binding energies instead of total energies. This is expected to facilitate the fitting process because binding energies are closer to zero. ",
        'format': "remove_atom_energies",
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
        'description': " Subtract van-der-Waals dispersion energy and forces from the reference data before fitting a neural network potential. ",
        'format': "remove_vdw_energies",
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
        'description': " If this keyword is set, the weights of the short-range NN are updated a second time after the force update with respect to the total energies in the data set.  This usually results in a more accurate potential energy fitting at the cost of slightly detiorated forces.",
        'format': "repeated_energy_update",
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
        'description': " Re-initialize the correlation matrix of the Kalman filter at each new training epoch.",
        'format': "reset_kalman",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'restrict_weights': {
        'description': " Restrict the weights of the NN to the interval  [`-restrictw +1.0`, `restrictw - 1.0`].",
        'format': "restrict_weights a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Boundary default_value for neural network weights. Must be positive.",
                'type': float,
                'default_value': -100000.0,
            },
        },
        'allow_multiple': False,
    },
    'runner_mode': {
        'description': " Choose the operating mode of `RuNNer`.",
        'format': "runner_mode i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " The chosen mode of `RuNNer`.",
                'type': int,
                'default_value': "None",
                'options': {
                    1: {
                        'description': " Preparation mode. Generate the symmetry functions from structures in the `input.data` file.",
                    },
                    2: {
                        'description': " Fitting mode. Determine the NN weight parameters.",
                    },
                    3: {
                        'description': " Production mode. Application of the NN potential, prediction of the energy and forces of all structures in the `input.data` file.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'save_kalman_matrices': {
        'type': bool,
        'description': " Save the Kalman filter matrices to the files `kalman.short.XXX.data` and `kalman.elec.XXX.data`. `XXX` is the nuclear charge of the respective element. The Kalman matrices are read from the files using the keyword  read_kalman_matrices.",
        'format': "save_kalman_matrices",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'scale_max_elec': {
        'description': " Rescale the electrostatic symmetry functions to an interval given by * scale_min_elec and  * scale_max_elec  For further details please see  scale_symmetry_functions.",
        'format': "scale_max_elec a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Upper boundary default_value for rescaling the electrostatic symmetry functions.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_max_short': {
        'description': " Rescale the electrostatic symmetry functions to an interval given by * scale_min_elec and  * scale_max_elec  For further details please see  scale_symmetry_functions.",
        'format': "scale_max_short a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Upper boundary default_value for rescaling the electrostatic symmetry functions.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_max_short_atomic': {
        'description': " Rescale the short-range symmetry functions to an interval given by * scale_min_short_atomic and  * scale_max_short_atomic  For further details please see  scale_symmetry_functions.",
        'format': "scale_max_short_atomic a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Upper boundary default_value for rescaling the short-range symmetry functions.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_max_short_pair': {
        'description': " Rescale the short-range pairwise symmetry functions to an interval given by * scale_min_short_pair and  * scale_max_short_pair  For further details please see  scale_symmetry_functions.",
        'format': "scale_max_short_pair a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Upper boundary default_value for rescaling the short-range pairwise symmetry functions.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_min_elec': {
        'description': " Rescale the electrostatic symmetry functions to an interval given by * scale_min_elec and  * scale_max_elec  For further details please see  scale_symmetry_functions.",
        'format': "scale_min_elec a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Lower boundary default_value for rescaling the electrostatic symmetry functions.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_min_short_atomic': {
        'description': " Rescale the short-range symmetry functions to an interval given by * scale_min_short_atomic and  * scale_max_short_atomic  For further details please see  scale_symmetry_functions.",
        'format': "scale_min_short_atomic a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Lower boundary default_value for rescaling the short-range symmetry functions.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_min_short_pair': {
        'description': " Rescale the short-range pairwise symmetry functions to an interval given by * scale_min_short_pair and  * scale_max_short_pair  For further details please see  scale_symmetry_functions.",
        'format': "scale_min_short_pair a0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'a0': {
                'description': " Lower boundary default_value for rescaling the short-range pairwise symmetry functions.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'scale_symmetry_functions': {
        'type': bool,
        'description': " Rescale symmetry functions to a certain interval (the default interval is 0 to  1). This has numerical advantages if the orders of magnitudes of different  symmetry functions are very different. If the minimum and maximum default_value for a symmetry function is the same for all structures, rescaling is not possible and `RuNNer` will terminate with an error. The interval can be specified by the  keywords  * scale_min_short_atomic, * scale_max_short_atomic,  * scale_min_short_pair, and * scale_max_short_pair  for the short range / pairwise NN and by  * scale_min_elec and  * scale_max_elec  for the electrostatic NN. ",
        'format': "scale_symmetry_functions",
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
        'description': " Request a separate random initialization of the bias weights at the beginning of `runner_mode 2` on an interval between  `biasweights_min` and `biasweights_max`. ",
        'format': "separate_bias_ini_short",
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
        'description': " Use a different Kalman filter correlation matrix for the energy and force  update. ",
        'format': "separate_kalman_short",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'short_energy_error_threshold': {
        'description': " Threshold default_value for the error of the energies in units of the RMSE of the previous epoch. A default_value of 0.3 means that only charges with an error larger than 0.3*RMSE will be used for the weight update. Large default_values (about 1.0) will speed up the first epochs, because only a few points will be used. ",
        'format': "short_energy_error_threshold a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of energy RMSE that a point needs to reach to be used in the weight update.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'short_energy_fraction': {
        'description': " Defines the random fraction of energies used for fitting the short range weights.",
        'format': "short_energy_fraction a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of energies used for short-range fitting. 100% = 1.0.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'short_energy_group': {
        'description': " Do not update the short range NN weights after the presentation of an  individual atomic charge, but average the derivatives with respect to the  weights over the specified number of structures for each element.",
        'format': "short_energy_group i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Number of structures per group. The maximum is given by points_in_memory.",
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
    'short_force_error_threshold': {
        'description': " Threshold default_value for the error of the atomic forces in units of the RMSE of the previous epoch. A default_value of 0.3 means that only forces with an error larger than 0.3*RMSE will be used for the weight update. Large default_values (about 1.0) will speed up the first epochs, because only a few points will be used. ",
        'format': "short_force_error_threshold a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of force RMSE that a point needs to reach to be used in the weight update.",
                'type': float,
                'default_value': 0.0,
            },
        },
        'allow_multiple': False,
    },
    'short_force_fraction': {
        'description': " Defines the random fraction of forces used for fitting the short range weights.",
        'format': "short_force_fraction a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Fraction of force used for short-range fitting. 100% = 1.0.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'short_force_group': {
        'description': " Do not update the short range NN weights after the presentation of an  individual atomic force, but average the derivatives with respect to the  weights over the specified number of forces for each element.",
        'format': "short_force_group i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " Number of structures per group. The maximum is given by points_in_memory.",
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
    'shuffle_weights_short_atomic': {
        'description': " Randomly shuffle some weights in the short-range atomic NN after a defined  number of epochs.",
        'format': "shuffle_weights_short_atomic i0 a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " The weights will be shuffled every `i0` epochs.",
                'type': int,
                'default_value': 10,
            },
            'a0': {
                'description': " Treshold that a random number has to pass so that the weights at handled will be shuffled. This indirectly defines the number of weights that will be shuffled.",
                'type': float,
                'default_value': 0.1,
            },
        },
        'allow_multiple': False,
    },
    'steepest_descent_step_charge': {
        'description': " Step size for steepest descent fitting of the atomic charges.",
        'format': "steepest_descent_step_charge a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Charge steepest descent step size.",
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'steepest_descent_step_energy_short': {
        'description': " Step size for steepest descent fitting of the short-range energy.",
        'format': "steepest_descent_step_energy_short a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Short-range energy steepest descent step size.",
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'steepest_descent_step_force_short': {
        'description': " Step size for steepest descent fitting of the short-range forces.",
        'format': "steepest_descent_step_force_short a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Short-range force steepest descent step size.",
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'symfunction_correlation': {
        'type': bool,
        'description': " Determine and print Pearson's correlation of all pairs of symmetry functions.",
        'format': "symfunction_correlation",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'symfunction_electrostatic': {
        'description': " Specification of the symmetry functions for a specific element with a specific  neighbor element combination for the electrostatic NN.",
        'format': "symfunction_electrostatic element type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the center element.",
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires the pair `element` and parameters `eta` and `rshift`. ```runner-config symfunction_electrostatic element 2 element eta rshift cutoff ```",
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
                        'description': " Angular function. Requires two pair elements and `parameters` `eta`, `lambda`, and `zeta`. ```runner-config symfunction_electrostatic element 3 element element eta lambda zeta cutoff ```",
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
                        'description': " Radial function. Requires a pair `element` and the parameter `eta`. ```runner-config symfunction_electrostatic element 4 element eta cutoff ```",
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
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config symfunction_electrostatic element 5 eta ```",
                        'parameters': {                          
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config symfunction_electrostatic element 6 cutoff ```",             
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires two `element`s and `parameters` `eta`, and `rshift`. ```runner-config symfunction_electrostatic element 8 element element eta rshift cutoff ```",
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
                        'description': " Angular function. Requires two `element`s and `parameters` `eta`. ```runner-config symfunction_electrostatic element 9 element element eta cutoff ```.",
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
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'symfunction_short': {
        'description': " Specification of the symmetry functions for a specific element with a specific  neighbor element combination for the short-range NN.",
        'format': "symfunction_short element type [parameters] cutoff",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'element': {
                'description': " The periodic table symbol of the center element.",
                'type': str,
                'default_value': None,
            },
            'type [parameters]': {
                'description': " The type of symmetry function to be used. Different `parameters` have to be set depending on the choice of `type`:",
                'type': int,
                'options': {
                    1: {
                        'description': " Radial function. Requires no further `parameters`.",
                    },
                    2: {
                        'description': " Radial function. Requires the pair `element` and parameters `eta` and `rshift`. ```runner-config symfunction_short element 2 element eta rshift cutoff ```",
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
                        'description': " Angular function. Requires two pair elements and `parameters` `eta`, `lambda`, and `zeta`. ```runner-config symfunction_short element 3 element element eta lambda zeta cutoff ```",
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
                        'description': " Radial function. Requires a pair `element` and the parameter `eta`. ```runner-config symfunction_short element 4 element eta cutoff ```",
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
                        'description': " Cartesian coordinate function. The parameter `eta` will determine the coordinate axis `eta=1.0: X, eta=2.0: Y, eta=3.0: Z`. No `cutoff` required. ```runner-config symfunction_short element 5 eta ```",
                        'parameters': {                          
                            'eta': {
                                'type': float,
                                'default_value': 0.0,
                            },
                        },
                    },
                    6: {
                        'description': " Bond length function. Requires no further `parameters`. ```runner-config symfunction_short element 6 cutoff ```",             
                    },
                    7: {
                        'description': " Not implemented.",
                    },
                    8: {
                        'description': " Angular function. Requires two `element`s and `parameters` `eta`, and `rshift`. ```runner-config symfunction_short element 8 element element eta rshift cutoff ```",
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
                        'description': " Angular function. Requires two `element`s and `parameters` `eta`. ```runner-config symfunction_short element 9 element element eta cutoff ```.",
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
                'description': " The symmetry function cutoff radius (unit: Bohr).",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': True,
    },
    'test_fraction': {
        'description': " Threshold for splitting between training and testing set in  [`runner_mode`](/runner/reference/keywords/runner_mode) 1.",
        'format': "test_fraction a0",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " Splitting ratio. A default_value of e.g. 0.1 means that 10% of the structures in the `input.data` file will be used as test set and 90% as training set.",
                'type': float,
                'default_value': 0.01,
            },
        },
        'allow_multiple': False,
    },
    'update_single_element': {
        'description': " During training, only the NN weight parameters for the NNs of a specified element will be updated. In this case the printed errors for the  forces and the charges will refer only to this element. The total energy error  will remain large since some NNs are not optimized. ",
        'format': "update_single_element i0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i0': {
                'description': " The nuclear charge of the element whose NN should be updated.",
                'type': int,
                'default_value': "None",
            },
        },
        'allow_multiple': False,
    },
    'update_worst_charges': {
        'description': " To speed up the fits for each block specified by points_in_memory first the worst charges are determined. Only these charges are then used for the weight update for this block of points, no matter if the fit would be reduced during the update.",
        'format': "update_worst_charges a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The percentage of worst charges to be considered for the weight update. A default_value of 0.1 here means to identify the worst 10%.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'update_worst_short_energies': {
        'description': " To speed up the fits for each block specified by points_in_memory first the worst energies are determined. Only these points are then used for the weight update for this block of points, no matter if the fit would be reduced during the update.",
        'format': "update_worst_short_energies a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The percentage of worst energies to be considered for the weight update. A default_value of 0.1 here means to identify the worst 10%.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'update_worst_short_forces': {
        'description': " To speed up the fits for each block specified by points_in_memory first the worst forces are determined. Only these points are then used for the weight update for this block of points, no matter if the fit would be reduced during the update.",
        'format': "update_worst_short_forces a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The percentage of worst forces to be considered for the weight update. A default_value of 0.1 here means to identify the worst 10%.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'use_atom_charges': {
        'type': bool,
        'description': " Use atomic charges for fitting. At the moment this flag will be switched on automatically by `RuNNer` if electrostatic NNs are requested. In future versions of `RuNNer` this keyword will be used to control different origins of atomic charges.",
        'format': "use_atom_charges",
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
        'description': " Check if sum of specified atomic energies is equal to the total energy of each structure.",
        'format': "use_atom_energies",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': True,
        },
        'default_value': True,
        'allow_multiple': False,
    },
    'use_damping': {
        'description': " In order to avoid too large (too positive or too negative) weight parameters, this damping scheme can be used. An additional term is added to the absolute energy error. * For the short range energy the modification is:    `errore=(1.d0-dampw)errore + (dampwsumwsquared) /dble(totnum_weightsshort)` * For the short range forces the modification is:   `errorf=(1.d0-dampw)errorf+(dampwsumwsquared) /dble(num_weightsshort(elementindex(zelem(i2))))` * For the short range forces the modification is:   `error=(1.d0-dampw)*error + (dampwsumwsquared) /dble(num_weightsewald(elementindex(zelem_list(idx(i1),i2))))`",
        'format': "use_damping a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " The damping parameter.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'use_electrostatics': {
        'type': bool,
        'description': " Calculate long range electrostatic interactions explicitly. The type of atomic charges is specified by the keyword  electrostatic_type .",
        'format': "use_electrostatics",
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
        'description': " Do not use a NN to calculate the atomic charges, but use fixed charges for each element independent of the chemical environment. electrostatic_type.",
        'format': "use_fixed_charges",
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
        'description': " Use Gaussian for modeling atomic charges during the construction of 4G-HDNNPs.",
        'format': "use_gausswidth",
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
        'description': " Use noise matrix for fitting the short range weight with the short range NN  weights with Kalman filter.",
        'format': "use_noisematrix",
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
        'description': " Restart a fit with old scaling parameters for the short-range and electrostatic  NNs. The symmetry function scaling factors are read from `scaling.data`.",
        'format': "use_old_scaling",
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
        'description': " Restart a fit with old weight parameters for the electrostatic NN. The files `weightse.XXX.data` must be present.  If the training data set is  unchanged, the error of epoch 0 must be the same as the error of the previous fitting cycle.",
        'format': "use_old_weights_charge",
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
        'description': " Restart a fit with old weight parameters for the short-range NN. This keyword is active only in mode 2. The files `weights.XXX.data` must be present.  If the training data set is unchanged, the error of epoch 0 must be the same as the error of the previous fitting cycle. However, if the training data is  different, the file `scaling.data` changes and either one of the keywords scale_symmetry_functions or center_symmetry_functions is used, the RMSE will be different.",
        'format': "use_old_weights_short",
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
        'description': " Make use of the OpenMP threading in Intels MKL library. ",
        'format': "use_omp_mkl",
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
        'description': " Use forces for fitting the short range NN weights. In  runner_mode 1, the  files `trainforces.data`, `testforces.data`, `trainforcese.data` and `testforcese.data` are written.  In  runner_mode 2,  these files are needed to use the short range forces for optimizing the  short range weights. However, if the training data is different, the file  `scaling.data` changes and either one of the keywords scale_symmetry_functions or center_symmetry_functions is used, the RMSE will be different.",
        'format': "use_short_forces",
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
        'description': " Use the a short range NN. Whether an atomic or pair-based energy expression is used is determined via the keyword nn_type_short.",
        'format': "use_short_nn",
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
        'description': " Overwrite the randomly initialized weights of the electrostatic NNs with  systematically calculated weights. The weights are evenly distributed over the interval between the minimum and maximum of the weights after the random initialization.",
        'format': "use_systematic_weights_electrostatic",
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
        'description': " Overwrite the randomly initialized weights of the short-range and pairwise NNs with systematically calculated weights. The weights are evenly distributed over the interval between the minimum and maximum of the weights after the random initialization.",
        'format': "use_systematic_weights_short",
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
        'description': "Turn on dispersion corrections.",
        'format': "use_vdw",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },    
    'vdw_screening': {
        'description': " Specification of the shape parameters of the Fermi-screening function in the employed DFT-D2 dispersion correction expression.",
        'format': "vdw_screening s_6 d s_R",
        'modes': {
            'mode1': True,
            'mode2': False,
            'mode3': True,
        },
        'arguments': {
            's_6': {
                'description': " The global scaling parameter $s_6$ in the screening function.",
                'type': float,
                'default_value': None,
            },
            'd': {
                'description': " The exchange-correlation functional dependent damping parameter. More information can be found in the theory section.",
                'type': float,
                'default_value': None,
            },
            's_R': {
                'description': " Range separation parameter.",
                'type': float,
                'default_value': None,
            },
        },
        'allow_multiple': False,
    },
    'vdw_type': {
        'description': " Specification of the type of dispersion correction to be employed.",
        'format': "vdw_type i0",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'arguments': {
            'i0': {
                'description': " Type of vdW correction scheme.",
                'type': int,
                'default_value': "default",
                'options': {
                    1: {
                        'description': " Simple, environment-dependent dispersion correction inspired by the Tkatchenko-Scheffler scheme.",
                    },
                    2: {
                        'description': " Grimme DFT-D2 correction.",
                    },
                    3: {
                        'description': " Grimme DFT-D3 correction.",
                    },
                },
            },
        },
        'allow_multiple': False,
    },
    'weight_analysis': {
        'type': bool,
        'description': " Print analysis of weights in  runner_mode 2.",
        'format': "weight_analysis",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'weights_max': {
        'description': " Set an upper limit for the random initialization of the short-range NN weights. ",
        'format': "weights_max a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " This number defines the maximum default_value for initial random short range weights.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'weights_min': {
        'description': " Set a lower limit for the random initialization of the short-range NN weights. ",
        'format': "weights_min a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " This number defines the minimum default_value for initial random short range weights.",
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'weightse_max': {
        'description': " Set an upper limit for the random initialization of the electrostatic NN weights. ",
        'format': "weightse_max a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " This number defines the maximum default_value for initial random electrostatic weights. This keyword is active only if an electrostatic NN is used, i.e. for `electrostatic_type` 1.",
                'type': float,
                'default_value': 1.0,
            },
        },
        'allow_multiple': False,
    },
    'weightse_min': {
        'description': " Set a lower limit for the random initialization of the electrostatic NN weights. ",
        'format': "weightse_min a0",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'a0': {
                'description': " This number defines the minimum default_value for initial random electrostatic weights. This keyword is active only if an electrostatic NN is used, i.e. for `electrostatic_type` 1.",
                'type': float,
                'default_value': -1.0,
            },
        },
        'allow_multiple': False,
    },
    'write_fit_statistics': {
        'type': bool,
        'description': "",
        'format': "write_fit_statistics",
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
        'description': " Write temporary weights after each data block defined by `points_in_memory` to files `tmpweights.XXX.out` and `tmpweightse.XXX.out`. XXX is the nuclear charge. This option is active only in `runner_mode` 2 and meant to store  intermediate weights in very long epochs.",
        'format': "write_temporary_weights",
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
        'description': " In runner_mode 2  write the files `traincharges.YYYYYY.out` and `testcharges.YYYYYY.out` for each epoch. `YYYYYY` is the number of the epoch.  The files are written only if the electrostatic NN is used in case of  electrostatic_type 1.  This can generate many large files and is intended for a detailed analysis of  the fits.",
        'format': "write_traincharges",
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
        'description': " In runner_mode 2  write the files  `trainforces.YYYYYY.out`  and  `testforces.YYYYYY.out`  for each epoch. `YYYYYY` is the number of the epoch. The files are written only if the short range NN is used and if the forces are used for training (keyword  use_short_forces.  This can generate  many large files and is intended for a detailed analysis of the fits.",
        'format': "write_trainforces",
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
        'description': " In runner_mode  3 write the file `symfunctions.out` containing  the unscaled and non-centered symmetry functions default_values of all atoms in the  predicted structure. The format is the same as for the files  function.data and  testing.data with the exception that no energies are given. The purpose of this file is code debugging.",
        'format': "write_symfunctions",
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
        'description': " In  runner_mode 2  write only the binding energy instead of total energies in  the files  trainpoints.XXXXXX.out and testpoints.XXXXXX.out for each epoch.",
        'format': "write_binding_energy_only",
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
        'description': " In  runner_mode 2  write the files  trainpoints.XXXXXX.out and testpoints.XXXXXX.out for each epoch. `XXXXXX` is the number of the epoch. The files are written only if the short range NN is used. This can generate many large files and is intended for a detailed analysis of the fits.",
        'format': "write_trainpoints",
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
        'description': " Write output without any formatting.",
        'format': "write_unformatted",
        'modes': {
            'mode1': True,
            'mode2': True,
            'mode3': True,
        },
        'default_value': False,
        'allow_multiple': False,
    },
    'write_weights_epoch': {
        'description': " Determine in which epochs the files `YYYYYY.short.XXX.out` and  `YYYYYY.elec.XXX.out` are written. ",
        'format': "write_weights_epoch i1",
        'modes': {
            'mode1': False,
            'mode2': True,
            'mode3': False,
        },
        'arguments': {
            'i1': {
                'description': " Frequency of weight output.",
                'type': int,
                'default_value': 1,
            },
        },
        'allow_multiple': False,
    },
}

# ---------- RuNNer ASE general defaults --------------------------------------#
# These dictionaries contains two types of keywords:
#   1. those which have to be changed frequently / depend on the system at hand,
#   2. those where the default value is different from the RuNNer default.
# They are not ordered alphabetically but by category instead.
# Here, only the keywords independent of the NNP type are given.

RUNNERASE_PARAMS: dict = {
    # General for all modes.
    'runner_mode': 1,                   # Default should be starting a new fit.
    'elements': None,                   # Will be set by ASE when attaching an atoms object.
    'number_of_elements': None,         # Will be set by ASE when attaching an atoms object.
    'bond_threshold': 0.5,              # Default ok but this has to be changed for every system.
    'nn_type_short': 1,                 # Most people use atomic NNs.                   
    'nnp_gen': 2,                       # 2Gs remain the most common use case.
    'use_short_nn': True,               # Short-range fitting is the default.
    'optmode_charge': 1,                # Default ok but should be a visible options.
    'optmode_short_energy': 1,          # Default ok but should be a visible options.
    'optmode_short_force': 1,           # Default ok but should be a visible options.
    'points_in_memory': 1000,           # Default is very small, modern PCs can handle more.
    'scale_symmetry_functions': True,   # Scaling is used by almost everyone.
    'cutoff_type': 1,                   # Default ok, but important.
    # Mode 1.
    'test_fraction': 0.1,               # The RuNNer default is only 1%, more is standard procedure.                
    # Mode 1 and 2.
    'random_seed': 0,                   # The seed will be initialized by np.rand later on, but it is an important information.     
    'use_short_forces': True,           # Force fitting is standard procedure.
    # Mode 1 and 3.
    'remove_atom_energies': True,       # Everyone only fits binding energies.
    'atom_energy': [],                  # Interdependent with `remove_atom_energies`.
    # Mode 2.
    'epochs': 30,                       # Default was 0 and most train for 30 epochs.
    'kalman_lambda_short': 0.98000,     # Very typical default value.
    'kalman_nue_short': 0.99870,        # Very typical default value.
    'mix_all_points': True,             # This is a standard option for most.
    'nguyen_widrow_weights_short': True,# Typically improves the fit.
    'repeated_energy_update': True,     # Typically improves the fit with force fitting.
    'short_energy_error_threshold': 0.1,# Only energies with 0.1*RMSE are used.
    'short_energy_fraction': 1.0,       # All energies are used.
    'short_force_error_threshold': 1.0, # All forces are used.
    'short_force_fraction': 0.1,        # 10% of the forces are used.   
    'use_old_weights_charge': False,    # Might be very important for fit restarting.
    'use_old_weights_short': False,     # Might be very important for fit restarting.
    'write_weights_epoch': 5,           # Default is 1, very verbose.
    # Mode 2 and 3.
    'center_symmetry_functions': True,  # This is standard procedure.
    'precondition_weights': True,       # This is standard procedure.
    # Mode 3.
    'calculate_forces': False,          # Will be set by ASE automatically, but should be visible.
    'calculate_stress': False,          # Will be set by ASE automatically, but should be visible.
    ### 2G-specific keywords.
    # All modes.
    'symfunction_short': [],
    # Mode 2 and 3.
    'global_activation_short': [None],
    'global_hidden_layers_short': None,
    'global_nodes_short': [None],
    ### PairNN-specific keywords.
    # All modes.
    'element_pairsymfunction_short': [None, None, None],
    'global_pairsymfunction_short': [None, None, None],
    # Mode 2 and 3.
    'element_activation_pair': [None, None, None, None],
    'element_hidden_layers_pair': [None, None],
    'element_nodes_pair': [None, None, 'global_nodes_pair'],
    'global_activation_pair': None,
    'global_hidden_layers_pair': None,
    'global_nodes_pair': None,
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
    # 'charge_fraction': 1.0,                    # Default is 0, which is useless.
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

RUNNERDATA_KEYWORDS = [
    'begin',               
    'comment',
    'lattice',
    'atom',
    'charge',
    'energy',
    'end'
]


if __name__ == "__main__":

    print(RUNNERCONFIG_DEFAULTS)
