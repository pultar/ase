from ase import Atoms
import numpy as np
from ase.units import Bohr, Rydberg

magmoms = {'Fe': 2.1,
           'Co': 1.4,
           'Ni': 0.6}


def write_positions_input(atoms):
    with open('position.dat', 'w') as filehandle:
        filehandle.write(str(1.0) + '\n\n')

        for i in range(3):
            filehandle.write('%s\n' % str(atoms.get_cell()[i])[1:-1])
        filehandle.write('\n')

        for index in range(len(atoms)):
            filehandle.write('%s %s\n' % (atoms[index].symbol, str(atoms[index].position)[1:-1]))


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
                str(mt_radius) + '  ' + str(ws_radius) + space + 'mt radius, ws radius',
                str(input_file) + space + 'Input potential file', str(pot_file) + space + 'Output potential file',
                str(keep_file) + space + 'Keep file', str(egrid[0]) + ' ' + str(egrid[1]) + ' ' + str(
            egrid[2]) + space + 'e-grid: ndiv(=#div/0.1Ryd), bott, eimag',
                str(ef) + ' ' + str(ef) + space + 'Fermi energy (estimate)',
                str(niter) + ' ' + str(mp) + space + 'Number of scf iterations, Mixing parameter']

    with open(str(symbol) + '_ss_in', 'w') as filehandle:
        for entry in contents:
            filehandle.write('%s\n' % entry)


def write_input_parameters_file(atoms, pot_in_form, pot_out_form,
                                stop_rout_name, nscf, method, out_to_scr,
                                temperature, val_e_rel, core_e_rel,
                                potential_type, xc, uniform_grid,
                                read_mesh, n_egrids, erbot, real_axis_method, real_axis_points,
                                spin, lmax_T, mt_radius, ndivin, liz_cutoff, k_scheme, kpts, bzsym,
                                mix_algo, mix_quantity, mix_param, etol, ptol,
                                em_iter, em_scheme, em_mix_param, em_eswitch, em_tm_tol):
    hline = 80 * '='
    separator = 18 * ' ' + 3 * ('* * *' + 14 * ' ')
    header = [hline, '{:^80s}'.format('Input Parameter Data File'),
              hline, separator, hline, '{:^80}'.format('System Related Parameters'), hline,
              'No. Atoms in System (> 0)  ::  ' + str(len(atoms)), hline, separator, hline]

    with open('i_new', 'w') as filehandle:
        for entry in header:
            filehandle.write('%s\n' % entry)

    def position_and_potential():
        species = np.unique(atoms.get_chemical_symbols())
        pot_files = ''
        for element in species:
            pot_files += str(element) + '_ss_pot '

        contents = ['{:^80s}'.format('Position and Potential Data Files'), hline,
                    'Atomic Position File Name  :: position.dat',
                    'Default Potential Input File Name  ::  ' + pot_files,
                    'Default Potential Input File Form  ::  ' + str(pot_in_form),
                    '   = 0: ASCII Format       ---------- \n' +
                    '   = 1: XDR Format         ----------\n' +
                    '   = 2: HDF Format         ----------\n' +
                    '   = 3: Machine Dependent Binary ----',
                    'Default Potential Output File Name ::  OUTPUT_Pot',
                    'Default Potential Output File Form ::  ' + str(pot_out_form),
                    '   = 0: ASCII Format       ----------\n' +
                    '   = 1: XDR Format         ----------\n' +
                    '   = 2: HDF Format         ----------\n' +
                    '   = 3: Machine Dependent Binary ----',
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def scf_params():
        contents = ['{:^80s}'.format('SCF-related Parameters'), hline,
                    'Stop-at Routine Name       :: ' + stop_rout_name,
                    'No. Iterations (> 0)       ::  ' + str(nscf),
                    'Method of SCF Calculation  ::  ' + str(method),
                    '   -2. Single Site         -----------\n' +
                    '   -1. ScreenKKR-LSMS      -----------(To be implemented)\n' +
                    '    0. Screen-KKR          -----------(To be implemented)\n' +
                    '    1. LSMS                -----------\n' +
                    '    2. KKR                 -----------\n' +
                    '    3. KKR-CPA             -----------',
                    'Output to Screen (y/n)     ::  ' + out_to_scr,
                    'Temperature Parameter (K)  ::  ' + str(temperature),
                    'Val. Electron Rel (>= 0)   ::  ' + str(val_e_rel),
                    '     0. Non-relativitic    ---------\n' +
                    '     1. Scalar-relativitic ---------\n' +
                    '     2. Full-relativitic   ---------',
                    'Core Electron Rel( >= 0)::' + str(core_e_rel),
                    '     0. Non-relativitic    ---------\n' +
                    '     1. Full-relativitic   ---------',
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def lda_pot_params():
        contents = ['{:^80s}'.format('LDA Potential-related Parameters'), hline,
                    'Potential Type (>= 0)      ::  ' + str(potential_type),
                    '     0. Muffin-tin         ----------\n' +
                    '     1. ASA                ----------\n' +
                    '     2. Muffin-tin ASA     ----------\n' +
                    '     3. Full               ----------\n' +
                    '     4. Muffin-Tin Test    ----------\n' +
                    '     5. Empty Lattice      ----------\n' +
                    '     6. Mathieu Potential  ----------',
                    'Exch-Corr. LDA Type (>= 0) ::  ' + str(xc),
                    '   Note: The input can be either one of the following numbers or,\n' +
                    '         e.g., LDA_X+LDA_C_HL for Hedin-Lundqvist LDA functional, or\n' +
                    '         GGA_X_PBE+GGA_C_PBE for PBE GGA function, etc.. The naming convention here\n' +
                    '         follows the definition given in LibXC.\n' +
                    '     0. Barth-Hedin        ---------\n' +
                    '     1. Vosk-Wilk-Nusair   ---------\n' +
                    '     2. Perdew-Zunger      ---------\n' +
                    '     3. Perdew-Wang GGA    ---------\n' +
                    '     4. PBE                ---------\n',
                    'Uniform Grid Parameters    ::  ' + str(uniform_grid)[1:-1],
                    '     = 2^n with n =1, 2, ... Three intergers used to define the grid numbers along\n' +
                    '       three Bravais lattice vector directions\n' +
                    'Note: Uniform grid is used for calculating the non-spherical electrostatic\n' +
                    '      potential, so it is only used for the full-potential calculation.',
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def energy_contour_params():
        contents = ['{:^80s}'.format('Energy (Ryd.) Contour Parameters'), hline,
                    'Read E-mesh from emeshs.inp :: ' + str(int(read_mesh)),
                    '     0. No                 ---------\n'
                    '     1. Yes. In this case, the following data have no effect',
                    'No. Energy Grids           ::  ' + str(n_egrids),
                    'Real Axis Bottom, erbot    ::  ' + str(erbot),
                    'SS Real Axis Int. Method   ::  ' + str(real_axis_method),
                    '     0. Uniform\n' + \
                    '     1. Adaptive',
                    'SS Real Axis Int. Points   ::  ' + str(real_axis_points),
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def magnetism_params():
        contents = ['{:^80s}'.format('Magnetism-related Parameters'), hline,
                    'Spin Index Param (>= 1)    ::  ' + str(spin),
                    '     1. No Spin            ---------\n'
                    '     2. Spin-polarized     ---------\n'
                    '     3. Spin-canted        ---------',
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def scattering_theory_params():
        contents = ['{:^80s}'.format('Scattering Theory-related Parameters'), hline,
                    'Default Lmax-T matrix      ::  ' + str(lmax_T),
                    'Default Muffin-tin Radius  ::  ' + str(mt_radius),
                    '   = 0: Using the inscribed sphere radius\n' +
                    '   = 1: Using the internal muffin-tin radius defined in ChemElementModule\n'+
                    '   = A specific real value (> 0.0, in atomic units)',
                    'Default No. Rad Points ndivin  :: ' + str(ndivin),
                    '   = 0: Not specified ---------------\n'
                    '   > 0: Speciflied. Note:  0  < r(j)=exp(j*hin) <= rmt, j=1,2,...,ndivin',
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def rk_space_params():
        contents = ['{:^80s}'.format('R-space or K-space Related Parameters'), hline,
                    'Default LIZ Cutoff Radius  :: ' + str(np.round(liz_cutoff / Bohr, 2)),
                    'Scheme to Generate K (>=0) ::  ' + str(k_scheme),
                    '     0. Special K-points ---------\n'
                    '     1. Tetrahedron      ---------\n'
                    '     2. Direction        ---------\n',
                    'Kx, Ky, Kz Division (> 0)  ::   ' + str(kpts)[1:-1],
                    'Symmetrize BZ Integration  ::   ' + str(bzsym),
                    '     0. No                 ---------\n'
                    '     1. Yes                ---------\n'
                    '    -2. Yes(Equiv. points) ---------',
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def mix_tol_params():
        contents = ['{:^80s}'.format('Mixing and Tolerance Parameters'), hline,
                    'Mixing algorithm           ::  ' + str(mix_algo),
                    '     0. Simple Mixing      ---------\n'
                    '     1. D.G.A. Mixing      ---------\n'
                    '     2. Broyden Mixing     ---------',
                    'Mixing quantity type       ::  ' + str(mix_quantity),
                    '     0. Charge mixing      ---------\n'
                    '     1. Potential mixing   ---------',
                    'Default Mixing Parameter   ::  ' + str(mix_param),
                    'Energy (Ryd) Tol (> 0)     ::  ' + str(etol / Rydberg),
                    'Potential Tol (> 0)        ::  ' + str(ptol / Rydberg),
                    hline, separator, hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    def mix_tol_em_params():
        contents = ['{:^80s}'.format('Mixing and Tolerance Parameters for Effective Medium'), hline,
                    'Maximum Effective Medium Iterations   :: ' + str(em_iter),
                    'Effective Medium Mixing Scheme        :: ' + str(em_scheme),
                    '     = 0: Simple mixing; = 1: Anderson mixing; = 2: Broyden mixing; = 3: Anderson Mixing by Messina group',
                    'Effective Medium Mixing Parameters    :: ' + str(em_mix_param)[1:-1],
                    '     Note: The first mixing value is for the energy points in standard mixing mode; the second mixing value\n'
                    '           is for the energy points in conservative mixing mode',
                    'Effective Medium Mixing eSwitch Value :: ' + str(np.round(em_eswitch / Rydberg, 3)),
                    '     Note: If Re[E] > 0 and Im[E] < eSwitch, the effective medium iteration is switched to the conservative mode',
                    'Effective Medium T-matrix Tol (> 0)   ::  ' + str(em_tm_tol), hline]

        with open('i_new', 'a') as filehandle:
            for entry in contents:
                filehandle.write('%s\n' % entry)

    position_and_potential()
    scf_params()
    lda_pot_params()
    energy_contour_params()
    magnetism_params()
    scattering_theory_params()
    rk_space_params()
    mix_tol_params()
    mix_tol_em_params()