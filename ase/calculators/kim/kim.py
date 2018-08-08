#!/usr/bin/env python

"""
Knowledgebase of Interatomic Models (KIM) Calculator for ASE written by:

Ellad B. Tadmor
University of Minnesota

This calculator selects an appropriate calculator for a KIM model depending on
whether it supports the KIM application programming interface (API) or is a
KIM Simulator Model.  For more information on KIM, visit https://openkim.org.
"""
from __future__ import print_function
import re
import os
import kimsm
from ase.calculators.lammpslib import LAMMPSlib
# from kimlammpsrun import kimLAMMPSrun
# from asap3 import EMT, EMTMetalGlassParameters, EMTRasmussenParameters, \
#                   OpenKIMcalculator
from .kimmodel import KIMModelCalculator
from .exceptions import KIMCalculatorError
from ase.data import atomic_numbers

__version__ = '1.4.0'
__author__ = 'Ellad Tadmor'

###############################################################################


def _get_kim_model_id_and_type(extended_kim_id):
    '''
    Determine whether "extended_kim_id" corresponds to either a KIM Model
    or KIM Simulator Model and extract the short KIM ID
    '''
    # Determine whether this is a KIM Model or SM
    if kimsm.is_simulator_model(extended_kim_id):
        this_is_a_KIM_MO = False
        pref = 'SM'
    else:
        this_is_a_KIM_MO = True
        pref = 'MO'
    # Try to parse model name assuming it has an extended KIM ID format
    # to obtain short KIM_ID. This is used to name the directory
    # containing the SM files.
    extended_kim_id_regex = pref + '_[0-9]{12}_[0-9]{3}'
    try:
        kim_id = re.search(extended_kim_id_regex, extended_kim_id).group(0)
    except AttributeError:
        kim_id = extended_kim_id  # Model name does not contain a short KIM ID,
                                  # so use full model name for the file directory.

    return kim_id, this_is_a_KIM_MO

###############################################################################


def _get_params_for_LAMMPS_calculator(model_defn):
    '''
    Extract parameters for LAMMPS calculator from model definition lines.
    Returns a dictionary with entries for "pair_style" and "pair_coeff".
    Expects there to be only one "pair_style" line. There can be multiple
    "pair_coeff" lines (result is returned as a list).
    '''
    parameters = {}
    parameters['pair_style'] = ''
    parameters['pair_coeff'] = []
    parameters['model_post'] = []
    found_pair_style = False
    found_pair_coeff = False
    for i in range(0, len(model_defn)):
        c = model_defn[i]
        if c.lower().startswith('pair_style'):
            if found_pair_style:
                raise KIMCalculatorError(
                    'ERROR: More than one pair_style in metadata file.')
            found_pair_style = True
            parameters['pair_style'] = c.split(" ", 1)[1]
        elif c.lower().startswith('pair_coeff'):
            found_pair_coeff = True
            parameters['pair_coeff'].append(c.split(" ", 1)[1])
        else:
            parameters['model_post'].append(c)
    if not found_pair_style:
        raise KIMCalculatorError(
            'ERROR: pair_style not found in metadata file.')
    if not found_pair_coeff:
        raise KIMCalculatorError(
            'ERROR: pair_coeff not found in metadata file.')
    return parameters

###############################################################################


def _add_init_lines_to_parameters(parameters, model_init):
    '''
    Add Simulator Model initialization lines to the parameter list for LAMMPS
    if there are any.
    '''
    parameters['model_init'] = []
    for i in range(0, len(model_init)):
        parameters['model_init'].append(model_init[i])

###############################################################################


def KIM(extended_kim_id, debug=False, kim_mo_simulator='kimpy',
                  lammps_calculator='lammpslib', lammps_lib_suffix=None):
    '''
    Wrapper routine that selects KIMCalculator for KIM Models or an appropriate
    ASE Calculator for KIM Simulator Models.

    extended_kim_id: string
       Extended KIM ID of the model to be calculated
    debug: boolean
       If True, temporary files are kept
    kim_mo_simulator: string
       Name of simulator to be used for running KIM API compliant KIM Models
       Available options: kimpy (default), asap, lammps
    '''
    # Determine whether this is a standard KIM Model or
    # a KIM Simulator Model
    kim_id, this_is_a_KIM_MO = _get_kim_model_id_and_type(extended_kim_id)

    # If this is a KIM Model (supports KIM API) return support through
    # a KIM-compliant simulator
    if this_is_a_KIM_MO:

        kim_mo_simulator = kim_mo_simulator.lower().strip()

        if kim_mo_simulator == 'kimpy':
            return KIMModelCalculator(extended_kim_id, debug=debug)

        elif kim_mo_simulator == 'asap':
            if debug:
                return(OpenKIMcalculator(extended_kim_id, verbose=True))  # ASAP
            else:
                return(OpenKIMcalculator(extended_kim_id))  # ASAP

        elif kim_mo_simulator == 'lammps':
            param_filenames = []  # no parameter files to pass
            parameters = {}
            parameters['pair_style'] = 'kim KIMvirial ' + \
                extended_kim_id.strip() + os.linesep
            parameters['pair_coeff'] = ['* * @<atom-type-list>@' + os.linesep]
            parameters['model_init'] = []
            parameters['model_post'] = []
            # Return LAMMPS calculator
            if debug:
                return kimLAMMPSrun(parameters=parameters, files=param_filenames,
                                    keep_tmp_files=True)
            else:
                return kimLAMMPSrun(parameters=parameters, files=param_filenames)

        else:
            raise KIMCalculatorError(
                'ERROR: Unsupported simulator "%s" requested to run KIM API compliant KIM Models.' % kim_mo_simulator)

    ### If we get to here, the model is a KIM Simulator Model ###

    # Initialize KIM SM object
    ksm = kimsm.ksm_object(extended_kim_id=extended_kim_id)
    param_filenames = ksm.get_model_param_filenames()

    # Double check that the extended KIM ID of the Simulator Model
    # matches the expected value. (If not, the KIM SM is corrupted.)
    SM_extended_kim_id = ksm.get_model_extended_kim_id()
    if extended_kim_id != SM_extended_kim_id:
        raise KIMCalculatorError('ERROR: SM extended KIM ID ("%s") does not match expected value ("%s").' % (
            SM_extended_kim_id, extended_kim_id))

    # Get simulator name
    simulator_name = ksm.get_model_simulator_name().lower()

    #  Get model definition from SM metadata
    model_defn = ksm.get_model_defn_lines()
    if len(model_defn) == 0:
        raise KIMCalculatorError(
            'ERROR: model-defn is an empty list in metadata file of Simulator Model "%s".' % extended_kim_id)
    if "" in model_defn:
        raise KIMCalculatorError(
            'ERROR: model-defn contains one or more empty strings in metadata file of Simulator Model "%s".' % extended_kim_id)

    ############################################################
    #  ASAP
    ############################################################

    if simulator_name == "asap":

        # Verify units (ASAP models are expected to work with "ase" units)
        supported_units = ksm.get_model_units().lower().strip()
        if supported_units != "ase":
            raise KIMCalculatorError(
                'ERROR: KIM Simulator Model units are "%s", but expected to be "ase" for ASAP.' % supported_units)

        # There should be only one model_defn line
        if len(model_defn) != 1:
            raise KIMCalculatorError(
                'ERROR: model-defn contains %d lines, but should only contain one line for an ASAP model.' % len(model_defn))

        # Return calculator
        unknown_potential = False
        if (model_defn[0].lower().strip().startswith("emt")):
            # pull out potential parameters
            pp = ''
            mobj = re.search(r"\(([A-Za-z0-9_\(\)]+)\)", model_defn[0])
            if not mobj == None:
                pp = mobj.group(1).strip().lower()
            if pp == '':
                calc = EMT()
            elif pp.startswith('emtrasmussenparameters'):
                calc = EMT(EMTRasmussenParameters())
            elif pp.startswith('emtmetalglassparameters'):
                calc = EMT(EMTMetalGlassParameters())
            else:
                unknown_potential = True

        if unknown_potential:
            raise KIMCalculatorError(
                'ERROR: Unknown model "%s" for simulator ASAP.' % model_defn[0])
        else:
            calc.set_subtractE0(False) # Use undocumented feature for the EMT
                                       # calculators to take the energy of an
                                       # isolated atoms as zero. (Otherwise it
                                       # is taken to be that of perfect FCC.)
            return calc

    ############################################################
    #  LAMMPS
    ############################################################

    elif simulator_name == "lammps":
        if lammps_calculator == 'lammpslib':
            supported_species = ksm.get_model_supported_species()
            atom_type_sym_list_string = ' '.join(supported_species)
            atom_type_num_list_string = ' '.join([str(atomic_numbers[s]) for s in supported_species])
        else:
            atom_type_sym_list_string = ''
            atom_type_num_list_string = ''

        # Process KIM templates (parameter file names, atom types, and queries)
        for i in range(0, len(model_defn)):
            model_defn[i] = kimsm.template_substitution(
                model_defn[i], param_filenames, ksm.sm_dirname,
                atom_type_sym_list_string, atom_type_num_list_string)

        # Get model init lines
        model_init = ksm.get_model_init_lines()

        #  Process KIM templates (parameter file names, atom types, and queries)
        for i in range(0, len(model_init)):
            model_init[i] = kimsm.template_substitution(
                model_init[i], param_filenames, ksm.sm_dirname,
                atom_type_sym_list_string, atom_type_num_list_string)

        # Get model supported units
        supported_units = ksm.get_model_units().lower().strip()

        if (lammps_calculator == 'kimlammpsrun'):

            # add cross-platform line separation to model definition lines
            model_defn = [s + os.linesep for s in model_defn]

            # Extract parameters for LAMMPS calculator from model definition lines
            parameters = _get_params_for_LAMMPS_calculator(model_defn)

            # Add units to parameters
            parameters["units"] = supported_units

            # add cross-platform line separation to model definition lines
            model_init = [s + os.linesep for s in model_init]

            # Add init lines to parameter list
            _add_init_lines_to_parameters(parameters, model_init)

            # Return LAMMPS calculator
            if debug:
                return kimLAMMPSrun(parameters=parameters, files=param_filenames,
                                    keep_tmp_files=True)
            else:
                return kimLAMMPSrun(parameters=parameters, files=param_filenames)

        elif (lammps_calculator == 'lammpslib'):

            # Setup LAMMPS header commands
            # lookup table
            model_init.insert(0, 'atom_modify map array sort 0 0')
            if not any("atom_style" in s.lower() for s in model_init): # atom style (if needed)
                model_init.insert(0, 'atom_style atomic')
            model_init.insert(
                0, 'units ' + supported_units.strip())     # units

            atom_types = {}
            for i_s, s in enumerate(supported_species):
                atom_types[s] = i_s+1
            # Return LAMMPSlib calculator
            return LAMMPSlib(lammps_header=model_init,
                                lammps_name=lammps_lib_suffix,
                                lmpcmds=model_defn,
                                atom_types = atom_types,
                                log_file='lammps.log',
                                keep_alive=True)

        else:
            raise KIMCalculatorError(
                'ERROR: Unknown LAMMPS calculator: "%s"' % lammps_calculator)

    ############################################################
    #  UNSUPPORTED
    ############################################################

    else:
        raise KIMCalculatorError(
            'ERROR: Unsupported simulator; simulator_name = "%s".' % simulator_name)

###############################################################################


def KIM_get_supported_species_list(extended_kim_id, kim_mo_simulator='kimpy'):
    '''
    Returns a list of the atomic species (element names) supported by the
    specified KIM Model or KIM Supported Model.

    extended_kim_id: string
       Extended KIM ID of the model to be calculated

    kim_mo_simulator: string
       Name of simulator to be used for obtaining the list of model species
       Available options: kimpy (default), asap
    '''
    # Determine whether this is a standard KIM Model or
    # a KIM Simulator Model
    kim_id, this_is_a_KIM_MO = _get_kim_model_id_and_type(extended_kim_id)

    # If this is a KIM Model, get supported species list
    if this_is_a_KIM_MO:

        if kim_mo_simulator == 'kimpy':

            calc = KIMModelCalculator(extended_kim_id)
            speclist = list(calc.get_kim_model_supported_species())

        elif kim_mo_simulator == 'asap':

            calc = OpenKIMcalculator(extended_kim_id)
            speclist = list(calc.get_kim_model_supported_species())

        else:
            raise KIMCalculatorError(
                'ERROR: Unsupported simulator "%s" requested to obtain KIM Model species list.' % kim_mo_simulator)

    # Otherwise this is an SM and we'll get the supported species list from metadata
    else:

        # Initialize KIM SM object
        ksm = kimsm.ksm_object(extended_kim_id=extended_kim_id)
        speclist = ksm.get_model_supported_species()

    # Return list of supported species
    return speclist


# do nothing if called directly
if __name__ == '__main__':
    pass
