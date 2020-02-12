""" This module defines io functions for MuST calculator"""

from ase import Atoms
import numpy as np
from ase.units import Bohr, Rydberg
from ase.io.must.default_params import defaults

magmoms = {'Fe': 2.1,
           'Co': 1.4,
           'Ni': 0.6}


def write_positions_input(atoms):
    with open('position.dat', 'w') as filehandle:
        filehandle.write(str(1.0) + '\n\n')

        for i in range(3):
            filehandle.write('%s\n' % str(atoms.get_cell()[i] / Bohr)[1:-1])
        filehandle.write('\n')

        for index in range(len(atoms)):
            filehandle.write('%s %s\n' % (atoms[index].symbol, str(atoms[index].position / Bohr)[1:-1]))


def write_atomic_pot_input(symbol, nspins=1, moment=0., xc=1, niter=40, mp=0.1):
    title = str(symbol) + ' Atomic Potential'
    output_file = str(symbol) + '_a_out'
    pot_file = str(symbol) + '_a_pot'
    z = int(Atoms(symbol).get_atomic_numbers())

    if moment == 0. and nspins == 2 and symbol in ['Fe', 'Co', 'Ni']:
        moment = magmoms[symbol]

    space = '                    '
    contents = ['Title:' + str(title),
                str(output_file) + space + 'Output file name. If blank, data will show on screen',
                str(z) + space + 'Atomic number', str(moment) + space + 'Magnetic moment',
                str(nspins) + space + 'Number of spins',
                str(xc) + space + 'Exchange-correlation type (1=vb-hedin,2=vosko)',
                str(niter) + space + 'Number of Iterations',
                str(mp) + space + 'Mixing parameter', str(pot_file) + space + 'Output potential file']

    with open(str(symbol) + '_a_in', 'w') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)


def write_single_site_pot_input(symbol, crystal_type, a, nspins=1, moment=0., xc=1, lmax=3, print_level=1, ncomp=1,
                                conc=1., mt_radius=0., ws_radius=0, egrid=(10, -0.4, 0.3), ef=0.7, niter=50, mp=0.1):
    title = str(symbol) + ' Single Site Potential'
    output_file = str(symbol) + '_ss_out'
    input_file = str(symbol) + '_a_pot'
    pot_file = str(symbol) + '_ss_pot'
    keep_file = str(symbol) + '_ss_k'

    z = int(Atoms(symbol).get_atomic_numbers())
    a = a / Bohr

    if moment == 0.:
        if nspins == 2:
            if symbol in ['Fe', 'Co', 'Ni']:
                moment = magmoms[symbol]
            else:
                moment = 0.
        else:
            moment = 0.
    space = '                    '
    contents = [str(title), str(output_file) + space + 'Output file name. If blank, data will show on screen',
                str(print_level) + space + 'Print level', str(crystal_type) + space + 'Crystal type (1=FCC,2=BCC)',
                str(lmax) + space + 'lmax', str(a) + space + 'Lattice constant',
                str(nspins) + space + 'Number of spins',
                str(xc) + space + 'Exchange Correlation type (1=vb-hedin,2=vosko)',
                str(ncomp) + space + 'Number of components', str(z) + '  ' +
                str(moment) + space + 'Atomic number, Magnetic moment', str(conc) + space + 'Concentrations',
                str(mt_radius / Bohr) + '  ' + str(ws_radius / Bohr) + space + 'mt radius, ws radius',
                str(input_file) + space + 'Input potential file', str(pot_file) + space + 'Output potential file',
                str(keep_file) + space + 'Keep file', str(egrid[0]) + ' ' + str(egrid[1]) + ' ' + str(
            egrid[2]) + space + 'e-grid: ndiv(=#div/0.1Ryd), bott, eimag',
                str(ef / Rydberg) + ' ' + str(ef / Rydberg) + space + 'Fermi energy (estimate)',
                str(niter) + ' ' + str(mp) + space + 'Number of scf iterations, Mixing parameter']

    with open(str(symbol) + '_ss_in', 'w') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)


def write_input_parameters_file(atoms, parameters):

    energy_params = ['etol', 'ptol', 'ftol',
                     'offset_energy_pt', 'em_switch']    # Parameters with units of energy
    distance_params = ['liz_cutoff', 'max_core_radius',
                       'max_mt_radius', 'core_radius', 'mt_radius']   # Parameters with units of length
    vector_params = ['uniform_grid', 'grid_origin', 'grid_1', 'grid_2', 'grid_3', 'grid_pts', 'kpts',
                     'moment_direction', 'constrain_field', 'liz_shell_lmax', 'em_mix_param']    # vector parameters
    # Header
    hline = 80 * '='
    separator = 18 * ' ' + 3 * ('* * *' + 14 * ' ')
    header = [hline, '{:^80s}'.format('Input Parameter Data File'),
              hline, separator, hline, '{:^80}'.format('System Related Parameters'), hline,
              'No. Atoms in System (> 0)  ::  ' + str(len(atoms)), hline, separator, hline]

    with open('i_new', 'w') as filehandle:
        for entry in header:
            filehandle.write('%s\n' % entry)

    species = np.unique(atoms.get_chemical_symbols())
    pot_files = ''
    for element in species:
        pot_files += str(element) + '_ss_pot '

    contents = ['Default Potential Input File Name  ::  ' + pot_files]

    # Rest of the parameters:
    for key in parameters.keys():
        if key in energy_params:
            parameters[key] = parameters[key] / Rydberg

        if key in distance_params:
            parameters[key] = parameters[key] / Bohr

        if key in vector_params:
            parameters[key] = str(parameters[key])[1:-1]

        contents.append(defaults[key] + '  ::  ' + str(parameters[key]))

    with open('i_new', 'a') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)
