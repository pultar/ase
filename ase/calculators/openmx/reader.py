"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2018 JaeHwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.
"""
#  from ase.calculators import SinglePointDFTCalculator
import os
import struct
import numpy as np
from ase.units import Ha, Bohr, Debye
from ase.calculators.calculator import ReadError


def read_openmx(filename=None, debug=False):
    from ase.calculators.openmx import OpenMX
    from ase import Atoms
    """
    Read results from typical OpenMX output files and returns the atom object
    In default mode, it reads every implementd properties we could get from
    the files. Unlike previous version, we read the information based on file.
    previous results will be eraised unless previous results are written in the
    next calculation results.

    Read the 'LABEL.log' file seems redundant. Because all the
    information should already be written in '.out' file. However, in the
    version 3.8.3, stress tensor are not written in the '.out' file. It only
    contained in the '.log' file. So... I implented reading '.log' file method
    """
    log_data = read_file(get_file_name('.log', filename))
    restart_data = read_file(get_file_name('.dat#', filename))
    dat_data = read_file(get_file_name('.dat', filename))
    out_data = read_file(get_file_name('.out', filename))
    scfout_data = read_scfout_file(get_file_name('.scfout', filename))
    dos_data = read_dos_file(get_file_name('.Dos.val', filename))
    """
    First, we get every data we could get from the all results files. And then,
    reform the data to fit to data structure of Atom object. While doing this,
    Fix the unit to ASE format.
    """
    parameters = get_parameters(out_data=out_data, log_data=log_data,
                                restart_data=restart_data, dat_data=dat_data,
                                scfout_data=scfout_data)
    atomic_formula = get_atomic_formula(out_data=out_data, log_data=log_data,
                                        restart_data=restart_data,
                                        scfout_data=scfout_data,
                                        dat_data=dat_data)
    results = get_results(out_data=out_data, log_data=log_data,
                          restart_data=restart_data, scfout_data=scfout_data,
                          dat_data=dat_data)

    atoms = Atoms(**atomic_formula)
    atoms.set_calculator(OpenMX(**parameters))
    atoms.calc.results = results
    return atoms


def read_file(filename, debug=False):
    """
    Read the 'LABEL.out' file. Using 'parameters.py', we read every 'allowed_
    dat' dictionory. while reading a file, if one find the key matcheds That
    'patters', which indicates the property we want is written, it will returns
    the pair value of that key. For example,
            example will be written later
    """
    from ase.calculators.openmx import parameters as param
    if os.path.isfile(filename) is not True:
        return {}
    patterns = {
      'Stress tensor': ('stress', read_stress_tensor),
      'Dipole moment': ('dipole', read_dipole),
      'Fractional coordinates of': ('scaled_positions', read_scaled_positions),
      'Utot.': ('energy', read_energy),
      'Chemical Potential': ('chemical_potential', read_chemical_potential),
      'coordinates.forces': ('forces', read_forces),
      'Eigenvalues': ('eigenvalues', read_eigenvalues)}
    special_patterns = {
      'Total spin moment': (('magmoms', 'total_magmom'),
                            read_magmoms_and_total_magmom),
                        }
    out_data = {}
    line = '\n'
    if(debug):
        print('Read results from %s' % filename)
    with open(filename, 'r') as f:
        while line != '':
            line = f.readline()
            for key in param.integer_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_integer(line)
            for key in param.float_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_float(line)
            for key in param.string_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_string(line)
            for key in param.bool_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_bool(line)
            for key in param.list_int_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_list_int(line)
            for key in param.list_float_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_list_float(line)
            for key in param.list_bool_keys:
                if key in line:
                    out_data[get_standard_key(key)] = read_list_bool(line)
            for key in param.matrix_keys:
                if '<'+key in line:
                    out_data[get_standard_key(key)] = read_matrix(line, key, f)
            for key in patterns.keys():
                if key in line:
                    out_data[patterns[key][0]] = patterns[key][1](line, f)
            for key in special_patterns.keys():
                if key in line:
                    a, b = special_patterns[key][1](line, f)
                    out_data[special_patterns[key][0][0]] = a
                    out_data[special_patterns[key][0][1]] = b
    return out_data


def read_scfout_file(filename=None):
    """
    Read the Developer output '.scfout' files. It Behaves like read_scfout.c,
    OpenMX module, but written in python. Note that some array are begin with
    1, not 0

    atomnum: the number of total atoms
    Catomnum: the number of atoms in the central region
    Latomnum: the number of atoms in the left lead
    Ratomnum: the number of atoms in the left lead
    SpinP_switch:
                 0: non-spin polarized
                 1: spin polarized
    TCpyCell: the total number of periodic cells
    Solver: method for solving eigenvalue problem
    ChemP: chemical potential
    Valence_Electrons: total number of valence electrons
    Total_SpinS: total value of Spin (2*Total_SpinS = muB)
    E_Temp: electronic temperature
    Total_NumOrbs: the number of atomic orbitals in each atom
    size: Total_NumOrbs[atomnum+1]
    FNAN: the number of first neighboring atoms of each atom
    size: FNAN[atomnum+1]
    natn: global index of neighboring atoms of an atom ct_AN
    size: natn[atomnum+1][FNAN[ct_AN]+1]
    ncn: global index for cell of neighboring atoms of an atom ct_AN
    size: ncn[atomnum+1][FNAN[ct_AN]+1]
    atv: x,y,and z-components of translation vector of periodically copied cell
    size: atv[TCpyCell+1][4]:
    atv_ijk: i,j,and j number of periodically copied cells
    size: atv_ijk[TCpyCell+1][4]:
    tv[4][4]: unit cell vectors in Bohr
    rtv[4][4]: reciprocal unit cell vectors in Bohr^{-1}
         note:
         tv_i \dot rtv_j = 2PI * Kronecker's delta_{ij}
         Gxyz[atomnum+1][60]: atomic coordinates in Bohr
         Hks: Kohn-Sham matrix elements of basis orbitals
    size: Hks[SpinP_switch+1]
             [atomnum+1]
             [FNAN[ct_AN]+1]
             [Total_NumOrbs[ct_AN]]
             [Total_NumOrbs[h_AN]]
    iHks:
         imaginary Kohn-Sham matrix elements of basis orbitals
         for alpha-alpha, beta-beta, and alpha-beta spin matrices
         of which contributions come from spin-orbit coupling
         and Hubbard U effective potential.
    size: iHks[3]
              [atomnum+1]
              [FNAN[ct_AN]+1]
              [Total_NumOrbs[ct_AN]]
              [Total_NumOrbs[h_AN]]
    OLP: overlap matrix
    size: OLP[atomnum+1]
             [FNAN[ct_AN]+1]
             [Total_NumOrbs[ct_AN]]
             [Total_NumOrbs[h_AN]]
    OLPpox: overlap matrix with position operator x
    size: OLPpox[atomnum+1]
                [FNAN[ct_AN]+1]
                [Total_NumOrbs[ct_AN]]
                [Total_NumOrbs[h_AN]]
    OLPpoy: overlap matrix with position operator y
    size: OLPpoy[atomnum+1]
                [FNAN[ct_AN]+1]
                [Total_NumOrbs[ct_AN]]
                [Total_NumOrbs[h_AN]]
    OLPpoz: overlap matrix with position operator z
    size: OLPpoz[atomnum+1]
                [FNAN[ct_AN]+1]
                [Total_NumOrbs[ct_AN]]
                [Total_NumOrbs[h_AN]]
    DM: overlap matrix
    size: DM[SpinP_switch+1]
            [atomnum+1]
            [FNAN[ct_AN]+1]
            [Total_NumOrbs[ct_AN]]
            [Total_NumOrbs[h_AN]]
    dipole_moment_core[4]:
    dipole_moment_background[4]:
    """
    from numpy import insert as ins
    from numpy import cumsum as cum
    from numpy import split as spl
    from numpy import sum, zeros
    if os.path.isfile(filename) is not True:
        return {}

    def easyReader(byte, data_type, shape):
        data_size = {'d': 8, 'i': 4}
        data_struct = {'d': float, 'i': int}
        dt = data_type
        ds = data_size[data_type]
        unpack = struct.unpack
        if len(byte) == ds:
            if dt == 'i':
                return data_struct[dt].from_bytes(byte, byteorder='little')
            elif dt == 'd':
                return np.array(unpack(dt*(len(byte)//ds), byte))[0]
        elif shape is not None:
            return np.array(unpack(dt*(len(byte)//ds), byte)).reshape(shape)
        else:
            return np.array(unpack(dt*(len(byte)//ds), byte))

    def inte(byte, shape=None):
        return easyReader(byte, 'i', shape)

    def floa(byte, shape=None):
        return easyReader(byte, 'd', shape)

    def readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f):
            myOLP = []
            myOLP.append([])
            for ct_AN in range(1, atomnum + 1):
                myOLP.append([])
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN] + 1):
                    myOLP[ct_AN].append([])
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        myOLP[ct_AN][h_AN].append(floa(f.read(8*TNO2)))
            return myOLP

    def readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, f):
        Hks = []
        for spin in range(SpinP_switch + 1):
            Hks.append([])
            Hks[spin].append([np.zeros(FNAN[0] + 1)])
            for ct_AN in range(1, atomnum + 1):
                Hks[spin].append([])
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN] + 1):
                    Hks[spin][ct_AN].append([])
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        Hks[spin][ct_AN][h_AN].append(floa(f.read(8*TNO2)))
        return Hks

    try:
        f = open(filename, mode='rb')
        atomnum, SpinP_switch = inte(f.read(8))
        Catomnum, Latomnum, Ratomnum, TCpyCell = inte(f.read(16))
        atv = floa(f.read(8*4*(TCpyCell+1)), shape=(TCpyCell+1, 4))
        atv_ijk = inte(f.read(4*4*(TCpyCell+1)), shape=(TCpyCell+1, 4))
        Total_NumOrbs = np.insert(inte(f.read(4*(atomnum))), 0, 1, axis=0)
        FNAN = np.insert(inte(f.read(4*(atomnum))), 0, 0, axis=0)
        natn = ins(spl(inte(f.read(4*sum(FNAN[1:] + 1))), cum(FNAN[1:] + 1)),
                   0, zeros(FNAN[0] + 1), axis=0)[:-1]
        ncn = ins(spl(inte(f.read(4*np.sum(FNAN[1:] + 1))), cum(FNAN[1:] + 1)),
                  0, np.zeros(FNAN[0] + 1), axis=0)[:-1]
        tv = ins(floa(f.read(8*3*4), shape=(3, 4)), 0, [0, 0, 0, 0], axis=0)
        rtv = ins(floa(f.read(8*3*4), shape=(3, 4)), 0, [0, 0, 0, 0], axis=0)
        Gxyz = ins(floa(f.read(8*(atomnum)*4), shape=(atomnum, 4)), 0,
                   [0., 0., 0., 0.], axis=0)
        Hks = readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, f)
        iHks = []
        if SpinP_switch == 3:
            iHks = readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, f)
        OLP = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        OLPpox = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        OLPpoy = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        OLPpoz = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        DM = readHam(SpinP_switch, FNAN, atomnum, Total_NumOrbs, natn, f)
        Solver = inte(f.read(4))
        ChemP, E_Temp = floa(f.read(8*2))
        dipole_moment_core = floa(f.read(8*3))
        dipole_moment_background = floa(f.read(8*3))
        Valence_Electrons, Total_SpinS = floa(f.read(8*2))

    except IOError:
        raise(IOError('Can not find %s' % filename))
    f.close()
    scf_out = {'atomnum': atomnum, 'SpinP_switch': SpinP_switch,
               'Catomnum': Catomnum, 'Latomnum': Latomnum, 'Hks': Hks,
               'Ratomnum': Ratomnum, 'TCpyCell': TCpyCell, 'atv': atv,
               'Total_NumOrbs': Total_NumOrbs, 'FNAN': FNAN, 'natn': natn,
               'ncn': ncn, 'tv': tv, 'rtv': rtv, 'Gxyz': Gxyz, 'OLP': OLP,
               'OLPpox': OLPpox, 'OLPpoy': OLPpoy, 'OLPpoz': OLPpoz,
               'Solver': Solver, 'ChemP': ChemP, 'E_Temp': E_Temp,
               'dipole_moment_core': dipole_moment_core, 'iHks': iHks,
               'dipole_moment_background': dipole_moment_background,
               'Valence_Electrons': Valence_Electrons, 'atv_ijk': atv_ijk,
               'Total_SpinS': Total_SpinS, 'DM': DM
               }
    return scf_out


def read_electron_valency(filename='H_CA13'):
    array = []
    with open(os.path.join(os.environ['OPENMX_DFT_DATA_PATH'],
                           'VPS/' + filename + '.vps'), 'r') as f:
        array = f.readlines()
        f.close()
    required_line = ''
    for line in array:
        if 'valence.electron' in line:
            required_line = line
    return rn(required_line)


def rn(line='\n', n=1):
    """
    Read n'th to last value.
    For example:
        ...
        scf.XcType          LDA
        scf.Kgrid         4 4 4
        ...
    In Python,
        >>> str(rn(line, 1))
        LDA
        >>> line = f.readline()
        >>> int(rn(line, 3))
        4
    """
    if line == '':
        return 0
    i = 0
    for j in range(n):
        i -= 1
        while line.split(' ')[i] == '' or line.split(' ')[i] == '\n':
            i -= 1
    return line.split(' ')[i].split('\n')[0]


def read_tuple(line):
    return (float(rn(line, 3)), float(rn(line, 2)), float(rn(line, 1)))


def read_tuple_integer(line):
    return (int(rn(line, 3)), int(rn(line, 2)), int(rn(line, 1)))


def read_tuple_float(line):
    return (float(rn(line, 3)), float(rn(line, 2)), float(rn(line, 1)))


def read_integer(line):
    return int(rn(line))


def read_float(line):
    return float(rn(line))


def read_string(line):
    return str(rn(line))


def read_bool(line):
    bool = str(rn(line)).lower()
    if bool == 'on':
        return True
    elif bool == 'off':
        return False
    else:
        print('Waning! boolean is %s. Return string' % bool)
        return bool


def read_list_int(line):
    print('read_list_int Not Implemented Yet')


def read_list_float(line):
    return float(rn(line))


def read_list_bool(line):
    print('read_list_bool Not Implemented Yet')


def read_matrix(line, key, f):
    matrix = []
    line = f.readline()
    while key not in line:
        matrix.append(line.split())
        line = f.readline()
    return matrix


def read_stress_tensor(line, f):
    f.readline()  # passing empty line
    f.readline()
    line = f.readline()
    xx, xy, xz = read_tuple_float(line)
    line = f.readline()
    yx, yy, yz = read_tuple_float(line)
    line = f.readline()
    zx, zy, zz = read_tuple_float(line)
    stress = [xx, yy, zz, (zy + yz)/2, (zx + xz)/2, (yx + xy)/2]
    return stress


def read_magmoms_and_total_magmom(line, f):
    total_magmom = read_float(line)
    f.readline()  # Skip empty lines
    f.readline()
    line = f.readline()
    magmoms = []
    while not(line == '' or line.isspace()):
        magmoms.append(read_float(line))
        line = f.readline()
    return magmoms, total_magmom


def read_energy(line, f):
    from ase.units import Ha
    return read_float(line) * Ha


def read_eigenvalues(line, f, debug=False):
    """
    Read the Eigenvalues in the `.out` file and returns the eigenvalue
    First, it assumes system have two spins and start reading until it reaches
    the end('*****...').

        eigenvalues[spin][kpoint][nbands]
    """
    if 'Hartree' in line:
        return None
    from ase.units import Ha
    if(debug):
        print("Read eigenvalue output")
    eigenvalues = []
    eigenvalues.append([])
    eigenvalues.append([])  # Assume two spins
    i = 0
    while '******' not in line:
        line = f.readline()
        if 'kloop' in line:
            line = f.readline()
            line = f.readline()
            line = f.readline()
            eigenvalues[0].append([])
            eigenvalues[1].append([])
            line = f.readline()
            while not (line == '' or line.isspace()):
                eigenvalues[0][i].append(float(rn(line, 2)))
                eigenvalues[1][i].append(float(rn(line, 1)))
                line = f.readline()
            i += 1
    return np.asarray(eigenvalues) * Ha


def read_forces(line, f):
    forces = []
    f.readline()  # Skip Empty line
    line = f.readline()
    while 'coordinates.forces>' not in line:
        forces.append(read_tuple_float(line))
        line = f.readline()
    return forces


def read_dipole(line, f):
    dipole = []
    while 'Total' not in line:
        line = f.readline()
    dipole.append(read_tuple_float(line))
    return dipole


def read_scaled_positions(line, f):
    scaled_positions = []
    f.readline()  # Skip Empty lines
    f.readline()
    f.readline()
    line = f.readline()
    while not(line == '' or line.isspace()):  # Detect empty line
        scaled_positions.append(read_tuple_float(line))
        line = f.readline()
    return scaled_positions


def read_chemical_potential(line, f):
    return read_float(line)


def read_dos_file(filename=None):
    if os.path.isfile(filename) is not True:
        return {}
    dos_data = {}
    with open(filename, 'r') as f:
        line = ''
        while 'irange' not in line:
            line = f.readline()
        number_of_bands = read_integer(line)
        while 'Kgrid' not in line:
            line = f.readline()
            number_of_k_points = int(rn(string, 1)) * int(rn(string, 2)) * int(rn(string, 3))
        while 'Eigenvalues' not in line:
            line = f.readline()
        line = f.readline()
        number_of_spins = get_number_of_spins()
        eigenvalues = \
            np.ndarray((number_of_k_points, number_of_spins, number_of_bands), float)
        for i in range(number_of_k_points):
            for j in range(number_of_spins):
                for k in range(number_of_bands):
                    eigenvalues[i, j, k] = float(rn(line, number_of_bands - k)) * Ha
                line = f.readline()
        dos_data['number_of_bands'] = number_of_bands
    return dos_data


def read_eigen_file(filename=None, debug=False):
    """Read the eigenvalues from '.eigen' file format"""
    try:
        with open(os.path.join(directory, prefix + '.eigen'), 'r') as f:
            string = f.readline()
            results['chemical_potential'] = float(rn(string, 1)) * Ha
            string = f.readline()
            nbands = int(rn(string, 1))
            nspin = 1  # still need to implement spin polarization method
            eigenvalues = []
            string = f.readline()
            while 'WF' not in string:
                eigenvalues.append(np.ndarray((nspin, nbands), float))
                for n in range(nbands):
                    eigenvalues[-1][0, n] = \
                        float(rn(string)) * Ha
                    string = f.readline()
    except ReadError:
        if(debug):
            print("No .eigen file found")
    return eigenvalues


def read_band_file(self, filename=None):
    npoints = 0
    band_kpath = []
    kpts = []
    try:
        fl = float
        if filename is None:
            filename = self.get_file_name('.Band')
        with open(filename, 'r') as f:
            li = f.readline()
            nbands = int(rn(li, 3))
            self.results['chemical_potential'] = float(rn(li)) * Ha
            li = f.readline()
            rec_cell = [[fl(rn(li, 9)), fl(rn(li, 8)), fl(rn(li, 7))],
                        [fl(rn(li, 6)), fl(rn(li, 5)), fl(rn(li, 4))],
                        [fl(rn(li, 3)), fl(rn(li, 2)), fl(rn(li, 1))]]
            li = f.readline()
            number_of_sym_points = int(rn(li))
            for a in range(number_of_sym_points):
                li = f.readline()
                npoints += int(rn(li, 9))
                start_p = (fl(rn(li, 8)), fl(rn(li, 7)), fl(rn(li, 6)))
                end_pts = (fl(rn(li, 5)), fl(rn(li, 4)), fl(rn(li, 3)))
                symbols = (str(rn(li, 2)), str(rn(li, 1)))
                band_kpath.append({'kpts': int(rn(li, 9)),
                                   'start_point': start_p,
                                   'end_point': end_pts,
                                   'path_symbols': symbols})
            self['band_kpath'] = band_kpath
            nspins = self.get_number_of_spins()
            if(self.debug):
                print('Spin, KPTS, bands', nspins, npoints, nbands)
            eigenvalues = np.zeros((nspins, npoints, nbands))
            for kpt in range(npoints):
                li = f.readline()
                kp = (fl(rn(li, 3)), fl(rn(li, 2)), fl(rn(li, 1)))
                kpts.append(kp)
                li = f.readline()
                eigenvalues_at_kp = li.split()
                eigenvalues[0, kpt] = eigenvalues_at_kp[:]
                if nspins == 2:
                    li = f.readline()
                    kp = (fl(rn(li, 3)), fl(rn(li, 2)), fl(rn(li, 1)))
                    li = f.readline()
                    eigenvalues_at_kp = li.split()
                    eigenvalues[1, kpt] = eigenvalues_at_kp[:]

    except ReadError:
        if(self.debug):
            print("No .Band file found")
    return eigenvalues[:] * Ha


def read_out_output(filename=None, debug=False):
    try:
        if debug:
            print("Trying to read .out File")
        fl = float
        eig = []  # eigenvalues
        nkpts = len(get_ibz_k_points())
        nbands = get_number_of_bands()
        nspins = get_number_of_spins()
        eig = np.zeros((nspins, nkpts, nbands))
        if(debug):
            print('Spin, KPTS, bands', nspins, nkpts, nbands)
        k = 0
        if filename is None:
            filename = get_file_name('.out')
        with open(filename, 'r') as f:
            while True:
                line = f.readline()
                if 'kloop' in line:
                    line = f.readline()
                    line = f.readline()
                    line = f.readline()
                    i = 0
                    try:
                        for b in range(nbands):
                            # print(line)
                            eig[0, k, b] = fl(rn(line, 1)) * Ha
                            eig[0, nkpts-1-k, b] = fl(rn(line, 1)) * Ha
                            if(nspins == 2):
                                eig[1, k, b] = fl(rn(line, 2)) * Ha
                                eig[1, nkpts-1-k, b] = fl(rn(line, 2)) * Ha
                            line = f.readline()
                            i += 1
                    except IndexError:
                        nbands = i
                        if debug:
                            print('nbands != orbital number %d' % i)
                        k += 1
                        continue
                    k += 1
                if nkpts == k or not line:
                    break
    except ReadError:
        if debug:
            print("No .out file found")
    return eig


def read_eigenvalue_output(filename=None):
    if self['dos_erange'] is not None:
        self.results['eigenvalues'] = self.read_dos_output()
    elif False:  # proper condtion setting required
        self.results['eigenvalues'] = self.read_eigen_output()
    elif self['band_dispersion'] is True or self['band_kpath'] is not None:
        self.results['eigenvalues'] = self.read_band_output()
    else:
        self.results['eigenvalues'] = self.read_out_output()


def read_mmn_file(filename=None):
    mmn = {}
    try:
        with open(filename, 'r') as f:
            string = f.readline()
            string = f.readline()
            # d_num = int(read_nth_to_last_value(string, 2))
            kpt_num = int(rn(string, 3))
            band_num = int(rn(string, 4))
            self['bloch_overlaps'] = [{} for i in range(kpt_num)]
            string = f.readline()
            while string != '':
                kpoint = int(read_nth_to_last_value(string, 5)) - 1
                nextkpoint = int(read_nth_to_last_value(string, 4)) - 1
                bzone = (int(read_nth_to_last_value(string, 3)),
                         int(read_nth_to_last_value(string, 2)),
                         int(read_nth_to_last_value(string, 1)))
                dirG = self.dirG(nextkpoint - kpoint, bzone)
                self['bloch_overlaps'][kpoint][dirG] = np.matrix(
                    np.zeros((band_num, band_num), complex))
                for m in range(band_num):
                    for n in range(band_num):
                        string = f.readline()
                        self['bloch_overlaps'][kpoint][dirG][m, n] = \
                            complex(
                                 float(read_nth_to_last_value(string, 2)),
                                 float(read_nth_to_last_value(string, 1)))
                string = f.readline()
    except ReadError:
        raise Exception('Please calculate the overlap matrix elements for '
                        'the bloch states')
    return mmn


def read_initial_wannier_projections():
    try:
        with open(os.path.join(self.directory,
                               self.prefix + '.amn'), 'r') as f:
            string = f.readline()
            string = f.readline()
            nbands = int(rn(string, 4))
            kpt_num = int(rn(string, 3))
            nwannier = int(rn(string, 2))
            self['initial_wannier_projections'] = [np.ndarray(
                (nbands, nwannier), complex) for i in range(kpt_num)]
            for k in range(kpt_num):
                for n in range(nwannier):
                    for m in range(nbands):
                        string = f.readline()
                        self['initial_wannier_projections'][k][m, n] = \
                            complex(
                            float(rn(string, 2)),
                            float(rn(string, 1)))
    except ReadError:
        raise Exception('Please calculate the overlap matrix elements for '
                        'the bloch states')


def get_parameters(out_data=None, log_data=None, restart_data=None,
                   scfout_data=None, dat_data=None):
    """
    From the given data sets, construct the dictionary 'parameters'. If data
    is in the paramerters, it will save it.
    """
    from ase.calculators.openmx import parameters as param
    scaned_data = [out_data, log_data, restart_data, scfout_data, dat_data]
    openmx_keywords = [param.tuple_integer_keys, param.tuple_float_keys,
                       param.tuple_bool_keys, param.integer_keys,
                       param.float_keys, param.string_keys, param.bool_keys,
                       param.list_int_keys, param.list_bool_keys,
                       param.list_float_keys, param.matrix_keys]
    parameters = {}
    for scaned_datum in scaned_data:
        for scaned_key in scaned_datum.keys():
            for openmx_keyword in openmx_keywords:
                if scaned_key in get_standard_key(openmx_keyword):
                    parameters[scaned_key] = scaned_datum[scaned_key]
                    continue
    translated_parameters = get_standard_parameters(parameters)
    parameters.update(translated_parameters)
    return {k: v for k, v in parameters.items() if v is not None}


def get_standard_parameters(parameters):
    """
    Translate the OpenMX parameters to standard ASE parameters. For example,

        scf.XcType -> xc
        scf.maxIter -> maxiter
        scf.energycutoff -> energy_cutoff
        scf.Kgrid -> kpts
        scf.EigenvalueSolver -> eigensolver
        scf.SpinPolarization -> spinpol
        scf.criterion -> convergence
        scf.Electric.Field -> external
        scf.Mixing.Type -> mixer
        scf.system.charge -> charge

    We followed GPAW schem.
    """
    from ase.calculators.openmx import parameters as param
    from ase.units import Bohr, Ha, Ry, fs, m, s
    units = param.unit_dat_keywords
    standard_parameters = {}
    standard_units = {'eV': 1, 'Ha': Ha, 'Ry': Ry, 'Bohr': Bohr, 'fs': fs,
                      'K': 1, 'GV / m': 1e9/1.6e-19 / m, 'Ha/Bohr': Ha/Bohr,
                      'm/s': m/s, '_amu': 1, 'Tesla': 1}
    translated_parameters = {
        'scf.XcType': 'xc',
        'scf.maxIter': 'maxiter',
        'scf.energycutoff': 'energy_cutoff',
        'scf.Kgrid': 'kpts',
        'scf.EigenvalueSolver': 'eigensolver',
        'scf.SpinPolarization': 'spinpol',
        'scf.criterion': 'convergence',
        'scf.Electric.Field': 'external',
        'scf.Mixing.Type': 'mixer',
        'scf.system.charge': 'charge'
        }

    for key in parameters.keys():
        for openmx_key in translated_parameters.keys():
            if key == get_standard_key(openmx_key):
                standard_key = translated_parameters[openmx_key]
                unit = standard_units.get(units.get(openmx_key), 1)
                standard_parameters[standard_key] = parameters[key] * unit
    return standard_parameters


def get_atomic_formula(out_data=None, log_data=None, restart_data=None,
                       scfout_data=None, dat_data=None,
                       scaled_positions=False):
    """
    From the given data sets, construct the dictionary 'atomic_formula'.
    OpenMX results gives following information. Since, we should pick one
    between position/scaled_position, scaled_positions are suppressed by
    default. We use input value of position. Not the position after
    calculation. It is temporal.

       Atoms.SpeciesAndCoordinate -> symbols
       Atoms.SpeciesAndCoordinate -> positions
       Atoms.UnitVectors -> cell
       scaled_positions -> scaled_positions, It is off By Default
       magmoms -> magmoms, Single value for each atom or three numbers for each
                           atom for non-collinear calculations.
    """
    atomic_formula = {}
    parameters = {'symbols': list, 'positions': list, 'scaled_positions': list,
                  'magmoms': list, 'cell': list}
    datas = [out_data, log_data, restart_data, scfout_data, dat_data]
    for data in datas:
        if 'atoms_speciesandcoordinates' in data:
            atoms_spncrd = data['atoms_speciesandcoordinates']
        if 'atoms_unitvectors' in data:
            atoms_unitvectors = data['atoms_unitvectors']
        for openmx_keyword in data.keys():
            for standard_keyword in parameters.keys():
                if openmx_keyword == standard_keyword:
                    atomic_formula[standard_keyword] = data[openmx_keyword]
    atomic_formula['symbols'] = [i[1] for i in atoms_spncrd]
    atomic_formula['positions'] = [[i[2], i[3], i[4]] for i in atoms_spncrd]
    atomic_formula['cell'] = atoms_unitvectors
    atomic_formula['pbc'] = True
    if atomic_formula.get('scaled_positions') is not None:
        del atomic_formula['scaled_positions']
    return atomic_formula


def get_results(out_data=None, log_data=None, restart_data=None,
                scfout_data=None, dat_data=None):
    """
    From the gien data sets, construct the dictionary 'results' and return it'
    OpenMX version 3.8 can yeild following properties
       free_energy,              Ha       # Same value with energy
       energy,                   Ha
       forces,                   Ha/Bohr
       stress(after 3.8 only)    Ha/Bohr**3
       dipole                    Debye
       read_chemical_potential   Ha
       magmoms                   muB  ??  set to 1
       magmom                    muB  ??  set to 1
    """
    from numpy import array as arr
    results = {}
    implemented_properties = {'free_energy': Ha, 'energy': Ha,
                              'forces': Ha/Bohr, 'stress': Ha/Bohr**3,
                              'dipole': Debye, 'chemical_potential': Ha,
                              'magmom': 1, 'magmoms': 1, 'eigenvalues': Ha}
    data = [out_data, log_data, restart_data, scfout_data, dat_data]
    for datum in data:
        for key in datum.keys():
            for property in implemented_properties.keys():
                if key == property:
                    results[key] = arr(datum[key])*implemented_properties[key]
    return results


def get_file_name(extension='.out', filename=None):
    directory, prefix = os.path.split(filename)
    abs_dir = os.path.join(os.getcwd(), directory)
    abs_lab = os.path.join(abs_dir, prefix)
    return abs_lab + extension


def get_standard_key(key):
    """
    Standard ASE parameter format is to USE unerbar(_) instead of dot(.). Also,
    It is recommended to use lower case alphabet letter. Not Upper. Thus, we
    change the key to standard key
    For example:
        'scf.XcType' -> 'scf_xctype'
    """
    if type(key) is str:
        return key.lower().replace('.', '_')
    elif type(key) is list:
        return [k.lower().replace('.', '_') for k in key]
