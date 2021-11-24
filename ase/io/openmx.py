"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2021 JaeHwan Shim and JaeJun Yu

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

import os
import re
import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointDFTCalculator

special_keywords = [# "system_currentdirectory",
                    # "data_path",
                    "species_number",
                    "definition_of_atomic_species",
                    "atoms_unitvectors",
                    "atoms_number",
                    "atoms_speciesandcoordinates"]

matrix_keywords = [
    'Definition.of.Atomic.Species',
    'Atoms.SpeciesAndCoordinates',
    'Atoms.UnitVectors',
    'Hubbard.U.values',
    'Atoms.Cont.Orbitals',
    'MD.Fixed.XYZ',
    'MD.TempControl',
    'MD.Init.Velocity',
    'Band.KPath.UnitCell',
    'Band.kpath',
    'MO.kpoint',
    'Wannier.Initial.Projectors'
]

unit_dat_keywords = {
    'Hubbard.U.Values': 'eV',
    'scf.Constraint.NC.Spin.v': 'eV',
    'scf.ElectronicTemperature': 'K',
    'scf.energycutoff': 'Ry',
    'scf.criterion': 'Ha',
    'scf.Electric.Field': 'GV / m',
    'OneDFFT.EnergyCutoff': 'Ry',
    'orbitalOpt.criterion': '(Ha/Borg)**2',
    'MD.Opt.criterion': 'Ha/Bohr',
    'MD.TempControl': 'K',
    'NH.Mass.HeatBath': '_amu',
    'MD.Init.Velocity': 'm/s',
    'Dos.Erange': 'eV',
    'scf.NC.Mag.Field.Spin': 'Tesla',
    'scf.NC.Mag.Field.Orbital': 'Tesla'
}

omx_bl = {True: 'On', False: 'Off'}


def get_up_down_spin(magmom, element, xc, data_path, year):
    magmom = np.linalg.norm(magmom)
    suffix = get_pseudo_potential_suffix(element, xc, year)
    filename = os.path.join(data_path, 'VPS/' + suffix + '.vps')
    valence_electron = float(get_electron_valency(filename))
    return [valence_electron / 2 + magmom / 2, valence_electron / 2 - magmom/2]


def get_spin_direction(magmoms):
    '''
    From atoms.magmom, returns the spin direction of phi and theta
    '''
    if np.array(magmoms).dtype == float or \
       np.array(magmoms).dtype is np.float64:
        return []
    else:
        magmoms = np.array(magmoms)
        return magmoms/np.linalg.norm(magmoms, axis=1)


def get_electron_valency(filename='H_CA13'):
    array = []
    with open(filename, 'r') as fd:
        array = fd.readlines()
        fd.close()
    required_line = ''
    for line in array:
        if 'valence.electron' in line:
            required_line = line
    return required_line.split()[-1]


def get_spinpol(atoms, parameters):
    ''' Judgeds the keyword 'scf.SpinPolarization'
     If the keyword is not None, spinpol gets the keyword by following priority
       1. standard_spinpol
       2. scf_spinpolarization
       3. magnetic moments of atoms
    '''
    standard_spinpol = parameters.get('spinpol', None)
    scf_spinpolarization = parameters.get('scf_spinpolarization', None)
    m = atoms.get_initial_magnetic_moments()
    syn = {True: 'On', False: None, 'on': 'On', 'off': None,
           None: None, 'nc': 'NC'}
    spinpol = np.any(m >= 0.1)
    if scf_spinpolarization is not None:
        spinpol = scf_spinpolarization
    if standard_spinpol is not None:
        spinpol = standard_spinpol
    if isinstance(spinpol, str):
        spinpol = spinpol.lower()
    return syn[spinpol]


def get_cutoff_radius_and_orbital(element=None, orbital=None):
    """
    For a given element, retruns the string specifying cutoff radius and
    orbital using default_settings.py. For example,
       'Si'   ->   'Si.7.0-s2p2d1'
    If one wannts to change the atomic radius for a special purpose, one should
    change the default_settings.py directly.
    """
    from ase.calculators.openmx import default_settings
    orbital = element
    orbital_letters = ['s', 'p', 'd', 'f', 'g', 'h']
    default_dictionary = default_settings.default_dictionary
    orbital_numbers = default_dictionary[element]['orbitals used']
    cutoff_radius = default_dictionary[element]['cutoff radius']
    orbital += "%.1f" % float(cutoff_radius) + '-'
    for i, orbital_number in enumerate(orbital_numbers):
        orbital += orbital_letters[i] + str(orbital_number)
    return orbital


def get_pseudo_potential_suffix(element=None, xc=None, year='13'):
    """
    For a given element, returns the string specifying pseudo potential suffix.
    For example,
        'Si'   ->   'Si_CA13'
    or
        'Si'   ->   'Si_CA19'
    depending on pseudo potential generation year
    """
    from ase.calculators.openmx import default_settings
    default_dictionary = default_settings.default_dictionary
    pseudo_potential_suffix = element
    vps = get_vps(xc)
    suffix = default_dictionary[element]['pseudo-potential suffix']
    pseudo_potential_suffix += '_' + vps + year + suffix
    return pseudo_potential_suffix


def get_vps(xc):
    if xc in ['GGA-PBE']:
        return 'PBE'
    else:
        return 'CA'


def get_dft_data_year(parameters):
    """
    It seems there is no version or potential year checker in openmx, thus we
    implemented one. It parse the pesudo potential path variable such as
    `~/PATH/TO/OPENMX/openmx3.9/DFT_DATA19/` or `.../openmx3.8/DFT_DATA13/`.
    By spliting this string, we harness the number of the year that generated
    such pseudo potential path.
    """
    year = parameters.get('data_path').split('/')[-1][-2:]

    if year in ['13', '19']:
        return year
    else:
        raise ValueError('data_path can not be found. Please specify '
                         '`data_path` as year of pseudo potential relesed')


def get_cutoff_radius_and_orbital(element=None, orbital=None):
    """
    For a given element, retruns the string specifying cutoff radius and
    orbital using default_settings.py. For example,
       'Si'   ->   'Si.7.0-s2p2d1'
    If one wannts to change the atomic radius for a special purpose, one should
    change the default_settings.py directly.
    """
    from ase.calculators.openmx import default_settings
    orbital = element
    orbital_letters = ['s', 'p', 'd', 'f', 'g', 'h']
    default_dictionary = default_settings.default_dictionary
    orbital_numbers = default_dictionary[element]['orbitals used']
    cutoff_radius = default_dictionary[element]['cutoff radius']
    orbital += "%.1f" % float(cutoff_radius) + '-'
    for i, orbital_number in enumerate(orbital_numbers):
        orbital += orbital_letters[i] + str(orbital_number)
    return orbital


def write_openmx_in(dst, atoms, properties=None, parameters=None, **kwargs):
    omx_keywords = {}
    # omx_keywords = parameters2keywords(atoms, parameters, **kwargs)
    # omx_keywords = omx_keywords_beautify(omx_keywords)
    for k in special_keywords:
        parameters[k] = parameters.get(k, None)

    with open(dst, 'w') as fd:
        for keyword, value in parameters.items():
            print(keyword)
            # Check if there exists special writing method for that keyword
            if keyword in special_keywords:
                write_special_keyword = globals().get('write_' + keyword)
                write_special_keyword(fd, atoms, parameters, **kwargs)
            elif keyword in matrix_keywords:
                write_matrix_keyword(fd, keyword, value, **kwargs)
            else:
                write_keyword(fd, keyword, value, **kwargs)


def write_keyword(fd, keyword, value):
    keyword = keyword.replace('_', '.')
    if isinstance(value, bool):
        fd.write(" {0:<30} {1}".format(keyword, omx_bl[value]))
    elif isinstance(value, (int, float, str)):
        fd.write(" {0:<30} {1}".format(keyword, value))
    elif isinstance(value, (list, tuple)):
        N = len(value)
        if N == 0:
            raise NotImplementedError("Empty list key", keyword, value)
        elif isinstance(value[0], bool):
            onoff = [omx_bl[bl] for bl in value]
            value = onoff
            valuestr = ' '.join([f'{{{i}:<5}}' for i in range(N)])
        elif isinstance(value[0], int):
            valuestr = ' '.join([f'{{{i}:<5}}' for i in range(N)])
        elif isinstance(value[0], float):
            valuestr = ' '.join([f'{{{i}:<8}}' for i in range(N)])
        elif isinstance(value[0], str):
            valuestr = ' '.join([f'{{{i}:<8}}' for i in range(N)])
        elif isinstance(value[0], (list, tuple)):
            write_matrix_keyword(fd, keyword, value, **kwargs)
        else:
            raise NotImplementedError("Unknown value type", keyword, value)
        formatstr = " {key:<30} " + valuestr
        fd.write(formatstr.format(*value, key=keyword))
    else:
        raise NotImplementedError("Unknown value type", keyword, value)
    fd.write("\n")


def write_matrix_keyword(fd, keyword, value):
    keyword = keyword.replace('_', '.')
    fd.write('<' + keyword + "\n")
    for val in value:
        N = len(val)
        if N == 0:
            raise NotImplementedError("Empty list inside key", keyword, val)
        fd.write("   ")
        if isinstance(val[0], bool):
            onoff = [omx_bl[bl] for bl in value]
            val = onoff
            valuestr = ' '.join([f'{{{i}:<5}}' for i in range(N)])
        elif isinstance(val[0], int):
            valuestr = ' '.join([f'{{{i}:<5}}' for i in range(N)])
        elif isinstance(val[0], float):
            valuestr = ' '.join([f'{{{i}:<8}}' for i in range(N)])
        elif isinstance(val[0], str):
            valuestr = ' '.join([f'{{{i}:<8}}' for i in range(N)])
        fd.write(valuestr.format(*val) + "\n")
    fd.write(keyword + '>')
    fd.write("\n\n")


def write_atoms_speciesandcoordinates(fd, atoms, parameters, **kwargs):
    """
    The atomic coordinates and the number of spin charge are given by the
    keyword
    'Atoms.SpeciesAndCoordinates' as follows:
    <Atoms.SpeciesAndCoordinates
     1  Mn    0.00000   0.00000   0.00000   8.0  5.0  45.0 0.0 45.0 0.0  1 on
     2  O     1.70000   0.00000   0.00000   3.0  3.0  45.0 0.0 45.0 0.0  1 on
    Atoms.SpeciesAndCoordinates>
    to know more, link <http://www.openmx-square.org/openmx_man3.7/node85.html>
    """
    key = 'atoms_speciesandcoordinates'

    unit = parameters.get(key + '_unit', 'ang').lower()
    if unit == 'ang':
        positions = atoms.get_positions()
    elif unit == 'frac':
        positions = atoms.get_scaled_positions(wrap=False)
    elif unit == 'au':
        positions = atoms.get_positions() / Bohr
    # `atoms.positions` overrides 'atoms_speciesandcoordinates'
    if parameters.get(key) is not None:
        atoms_speciesandcoordinates = parameters[key].copy()
        for i in range(len(atoms)):
            atoms_speciesandcoordinates[i][2] = positions[i, 0]
            atoms_speciesandcoordinates[i][3] = positions[i, 1]
            atoms_speciesandcoordinates[i][4] = positions[i, 2]
    else:
        atoms_speciesandcoordinates = []
        # Appending number and elemental symbol
        elements = atoms.get_chemical_symbols()
        for i, element in enumerate(elements):
            atoms_speciesandcoordinates.append([str(i + 1), element])
        # Positions
        for i, position in enumerate(positions):
            atoms_speciesandcoordinates[i].extend(position)

        xc = parameters.get('scf_xctype')
        year = get_dft_data_year(parameters)
        data_pth = parameters.get('data_path')

        # Mag moment
        magmoms = atoms.get_initial_magnetic_moments()
        for i, magmom in enumerate(magmoms):
            up_down_spin = get_up_down_spin(magmom, elements[i], xc, data_pth, year)
            atoms_speciesandcoordinates[i].extend(up_down_spin)
        # Appending magnetic field Spin magnetic moment theta phi
        spin_directions = get_spin_direction(magmoms)
        for i, spin_direction in enumerate(spin_directions):
            atoms_speciesandcoordinates[i].extend(spin_direction)

    write_matrix_keyword(fd, key, atoms_speciesandcoordinates)


def write_species_number(fd, atoms, parameters, **kwargs):
    key = 'species_number'
    if parameters.get(key) is not None:
        write_keyword(fd, key, parameters[key])
    else:
        write_keyword(fd, key, len(set(atoms.symbols)))


def write_atoms_number(fd, atoms, parameters, **kwargs):
    key = 'atoms_number'
    if parameters.get(key) is not None:
        write_keyword(fd, key, parameters[key])
    else:
        write_keyword(fd, key, len(atoms))


def write_definition_of_atomic_species(fd, atoms, parameters, **kwargs):
    """
    Using atoms and parameters, Returns the list `definition_of_atomic_species`
    where matrix of strings contains the information between keywords.
    For example,
     definition_of_atomic_species =
         [['H','H5.0-s1>1p1>1','H_CA13'],
          ['C','C5.0-s1>1p1>1','C_CA13']]
    Goes to,
      <Definition.of.Atomic.Species
        H   H5.0-s1>1p1>1      H_CA13
        C   C5.0-s1>1p1>1      C_CA13
      Definition.of.Atomic.Species>
    Further more, you can specify the wannier information here.
    A. Define local functions for projectors
      Since the pseudo-atomic orbitals are used for projectors,
      the specification of them is the same as for the basis functions.
      An example setting, for silicon in diamond structure, is as following:
   Species.Number          2
      <Definition.of.Atomic.Species
        Si       Si7.0-s2p2d1    Si_CA13
        proj1    Si5.5-s1p1d1f1  Si_CA13
      Definition.of.Atomic.Species>
    """
    key = 'definition_of_atomic_species'
    if parameters.get(key) is not None:
        write_matrix_keyword(fd, key, parameters[key])
        return None

    # Maybe it should not give at all
    definition_of_atomic_species = []
    year = get_dft_data_year(parameters)
    xc = parameters.get('scf_xctype')

    species = list(set(atoms.get_chemical_symbols()))
    for element in species:
        rad_orb = get_cutoff_radius_and_orbital(element=element)
        suffix = get_pseudo_potential_suffix(element=element, xc=xc, year=year)
        definition_of_atomic_species.append([element, rad_orb, suffix])
    # Put the same orbital and radii with chemical symbol.
    wannier_projectors = parameters.get('definition_of_wannier_projectors', [])
    for i, projector in enumerate(wannier_projectors):
        full_projector = definition_of_atomic_species[i]
        full_projector[0] = projector
        definition_of_atomic_species.append(full_projector)
    return definition_of_atomic_species


def write_atoms_unitvectors(fd, atoms, parameters, **kwargs):
    key = "atoms_unitvectors"
    if parameters.get(key) is not None:
        write_matrix_keyword(fd, key, parameters[key])
        return None

    unit = parameters.get('atoms_unitvectors_unit', 'ang').lower()
    if unit == 'ang':
        write_matrix_keyword(fd, key, atoms.cell)
    elif unit == 'au':
        write_matrix_keyword(fd, key, atoms.cell / Bohr)
    else:
        raise NotImplementedError("Unit %s not implemented" % unit)


def parse_openmx_log_cell(txt, version='3.9.2'):
    pattern = 'lattice vectors \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1)
    fp = match.end(0)
    ep = re.search(r'reciprocal', txt[fp:]).start(0)
    cell = []
    A = re.search(r'^A\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    B = re.search(r'^B\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    C = re.search(r'^C\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    cell.append([float(l) for l in A])
    cell.append([float(l) for l in B])
    cell.append([float(l) for l in C])
    return cell

def parse_openmx_log_pbc(txt, version='3.9.2'):
    return True


def parse_openmx_log_symbols(txt, version='3.9.2'):
    pattern = r'<Band_DFT>  DM,'

    fp = re.search(pattern, txt).end(0)
    ep = re.search(r'Sum of MulP:', txt[fp:]).end(0)
    symbols = []
    for line in txt[fp:fp+ep].split('\n')[1:-1]:
        symbols.append(line.split()[1])
    return symbols


def parse_openmx_log_positions(txt, version='3.9.2'):
    pattern = r'XYZ\((\S+)\) Fxyz\((\S+)\)=(.+)'
    positions = []
    for m in re.finditer(pattern, txt, re.M):
        unit = m.group(1)
        lines = m.group(3).split()[:3]
        if unit == 'ang':
            position = [float(l) for l in lines]
        positions.append(position)
    return positions

def parse_openmx_log_energy(txt, version='3.9.2'):
    pattern = r'Utot  =\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    return float(match.group(1))

def parse_openmx_log_forces(txt, version='3.9.2'):
    pattern = r'Fxyz\((\S+)\)=(.+)'
    forces = []
    for m in re.finditer(pattern, txt, re.M):
        unit = m.group(1)
        lines = m.group(2).split()[-3:]
        if unit == 'a.u.':
            force = [float(l) for l in lines]
        forces.append(force)
    return forces

def parse_openmx_log_stress(txt, version='3.9.2'):
    pattern = r'Stress tensor \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1)
    fp = match.start(0)
    fp += re.search(r'[\r\n]', txt[fp:]).end(0)
    fp += re.search(r'[\r\n]', txt[fp:]).end(0)
    ep = fp + re.search('\*', txt[fp:]).start(0)
    if unit == 'Hartree/bohr^3':
        stress = [float(val) for val in txt[fp:ep].split()]
    return stress

def parse_openmx_log_version(txt, version='3.9.2'):
    pattern = r'Welcome to OpenMX\s+Ver\.\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    version = match.group(1)
    return version

def parse_openmx_log_steps(txt, return_partition=False, version='3.9.2'):
    #pattern = r"SCF history at MD= \d+"
    #pattern = r'SCF calculation at MD = (\S+)'
    #pattern = r'MD or geometry opt\. at MD = (\S+)'
    pattern = r'Allocation of atoms to proccesors at MD_iter=\s+(\d+)'
    regrex = re.compile(pattern, re.M)
    steps = []
    partition = []
    for m in regrex.finditer(txt):
        steps.append(m.group(1))
        partition.append(m.start(0))
    partition.append(len(txt))
    if return_partition:
        return steps, partition
    return steps


def read_openmx_log(filename='openmx.log', index=-1):
    """
    return atoms or list of atoms
    """

    if isinstance(index, int):
        index = [index]
    elif index == ':':
        index = np.s_[:]
    elif isinstance(index, list):
        None
    else:
        raise NotImplemented('')

    with open(filename, 'r') as fd:
        txt = fd.read()

    version = parse_openmx_log_version(txt)
    steps, partition = parse_openmx_log_steps(txt, version=version,
                                              return_partition=True)

    md = False
    if len(steps) != 1:
        md = True

    images =[]
    idx = np.arange(len(steps))[index]

    for i in idx:
        text = txt[partition[i]:partition[i+1]]

        cell = parse_openmx_log_cell(text, version=version)
        symbols = parse_openmx_log_symbols(text, version=version)
        if md:
            positions = parse_openmx_log_md_positions(text, version=version)
        else:
            positions = np.zeros((len(symbols), 3))

        energy = parse_openmx_log_energy(text, version=version)
        forces = parse_openmx_log_forces(text, version=version)
        stress = parse_openmx_log_stress(text, version=version)

        pbc = parse_openmx_log_pbc(text, version=version)

        results = {'energy': energy, 'forces': forces,
                  'stress': stress}

        atoms = Atoms(symbols, positions=positions,
                      cell=cell, pbc=pbc)

        calc = SinglePointDFTCalculator(
            atoms, **results)
        atoms.calc = calc
        atoms.calc.name = 'openmx2'

        print('cell', cell)
        print('symbols', symbols)
        print('positons', positions)
        print('E', energy)
        print('Forces', forces)
        print('Stress', stress)

        images.append(atoms)
    return images

def parse_openmx_out_cell(txt, version='3.9.2'):
    pattern = 'lattice vectors \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1)
    fp = match.end(0)
    ep = re.search(r'reciprocal', txt[fp:]).start(0)
    cell = []
    A = re.search(r'^A\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    B = re.search(r'^B\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    C = re.search(r'^C\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    cell.append([float(l) for l in A])
    cell.append([float(l) for l in B])
    cell.append([float(l) for l in C])
    return cell

def parse_openmx_out_pbc(txt, version='3.9.2'):
    return True


def parse_openmx_out_symbols(txt, version='3.9.2'):
    pattern = r'<Band_DFT>  DM,'

    fp = re.search(pattern, txt).end(0)
    ep = re.search(r'Sum of MulP:', txt[fp:]).end(0)
    symbols = []
    for line in txt[fp:fp+ep].split('\n')[1:-1]:
        symbols.append(line.split()[1])
    return symbols


def parse_openmx_out_positions(txt, version='3.9'):
    pattern = r'xyz-coordinates \((\S+)\)'
    unit = re.finditer(pattern, txt, re.M)
    pattern = r'<coordinates'
    unit = re.finditer(pattern, txt, re.M)
    positions = []
    for m in re.finditer(pattern, txt, re.M):
        unit = m.group(1)
        lines = m.group(3).split()[:3]
        if unit == 'ang':
            position = [float(l) for l in lines]
        positions.append(position)
    return positions

def parse_openmx_out_energy(txt, version='3.9'):
    pattern = r'Total energy \(\S+\) = (\d+)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1)
    return float(match.group(2)) * unit

def parse_openmx_out_forces(txt, version='3.9'):
    pattern = r'and forces \((\S+)\)'
    unit = re.finditer(pattern, txt, re.M)
    pattern = r'<coordinates'
    forces = []
    for m in re.finditer(pattern, txt, re.M):
        unit = m.group(1)
        lines = m.group(3).split()[:3]
        if unit == 'ang':
            forces = [float(l) for l in lines]
        positions.append(position)


def parse_openmx_out_version(txt, version='3.9.2'):
    pattern = r'This calculation was performed by OpenMX\s+Ver\.\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    version = match.group(1)
    return version

def read_openmx_out_results(filename='openmx.out'):
    """
    return atoms with results in it
    """

    with open(filename, 'r') as fd:
        txt = fd.read()

    version = parse_openmx_out_version(txt)

    cell = parse_openmx_out_cell(txt, version=version)
    symbols = parse_openmx_out_symbols(txt, version=version)
    positions = parse_openmx_out_positions(text, version=version)

    energy = parse_openmx_out_energy(text, version=version)
    forces = parse_openmx_out_forces(text, version=version)

    pbc = parse_openmx_out_pbc(text, version=version)

    results = {'energy': energy, 'forces': forces}

    atoms = Atoms(symbols, positions=positions,
                  cell=cell, pbc=pbc)

    calc = SinglePointDFTCalculator(
        atoms, **results)
    atoms.calc = calc
    atoms.calc.name = 'openmx2'

    return atoms

def read_openmx_results(outfile, logfile):
    results = {}
    version =
    parse_openmx_out_energy()
    parse_openmx_out_energy()

def read_openmx_md(filename='openmx.md'):
    None
