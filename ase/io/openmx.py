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
from ase.units import Bohr, Ang, Ha
# from typing import runtime_checkable
from io import IOBase

units = {'bohr': Bohr, 'au': Bohr, 'ang': Ang, 'hartree/bohr^3': Ha / Bohr**3,
         'hartree': Ha, 'hartree/bohr': Ha / Bohr}


special_keywords = ["species_number",
                    "definition_of_atomic_species",
                    "atoms_unitvectors",
                    "atoms_number",
                    "atoms_speciesandcoordinates"]

matrix_keywords = [
    'definition_of_atomic_species',
    'atoms_speciesandcoordinates',
    'atoms_unitvectors',
    'hubbard_u_values',
    'atoms_cont_orbitals',
    'md_fixed_xyz',
    'md_tempcontrol',
    'md_init_velocity',
    'band_kpath_unitcell',
    'band_kpath',
    'mo_kpoint',
    'wannier_initial_projectors'
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

def write_openmx_in(dst, atoms, properties=None, parameters=None, **kwargs):
    for k in special_keywords:
        parameters[k] = parameters.get(k, None)

    # if isinstance(dst, (str, os.PathLike)):
    fd_close_flag = False
    if isinstance(dst, IOBase):
        fd = dst
    else:
        fd_close_flag = True
        fd = open(dst, 'w')

    for keyword, value in parameters.items():
        # Check if there exists special writing method for that keyword
        if keyword in special_keywords:
            write_special_keyword = globals().get('write_' + keyword)
            write_special_keyword(fd, atoms, parameters, **kwargs)
        elif keyword in matrix_keywords:
            write_matrix_keyword(fd, keyword, value, **kwargs)
        else:
            write_keyword(fd, keyword, value, **kwargs)

    if fd_close_flag:
        fd.close()


def write_keyword(fd, keyword, value):
    """Write the f'$(keyword)      $(value)' pair at the `fd`

    >>> import sys
    >>> fd = sys.stdout  # or fd = open('filename')

    None example
    >>> keyword = 'scf_stress_tensor'
    >>> value = None
    >>> write_keyword(fd, keyword, value)

    Boolean example
    >>> keyword = 'scf_stress_tensor'
    >>> value = True
    >>> write_keyword(fd, keyword, value)
     scf.stress.tensor              On
    >>> value = False
    >>> write_keyword(fd, keyword, value)
     scf.stress.tensor              Off

    Integer / Float / String example
    >>> keyword = 'scf_energycutoff'
    >>> value = 300
    >>> write_keyword(fd, keyword, value)
     scf.energycutoff               300
    >>> keyword = 'scf_criterion'
    >>> value = 0.0001
    >>> write_keyword(fd, keyword, value)
     scf.criterion                  0.0001
    >>> keyword = 'scf_xctype'
    >>> value = 'gga-pbe'
    >>> write_keyword(fd, keyword, value)
     scf.xctype                     gga-pbe

    """
    if value is None:
        return None
    keyword = keyword.replace('_', '.')
    if isinstance(value, bool):
        fd.write(" {0:<30} {1}".format(keyword, omx_bl[value]))
    elif isinstance(value, (int, float, str)):
        fd.write(" {0:<30} {1}".format(keyword, value))
    elif isinstance(value, (list, tuple)):
        valuestr = ''
        for v in value:
            if isinstance(v, bool):
                valuestr += omx_bl[v]
            elif isinstance(v, int):
                valuestr += '{0:<5} '.format(v)
            elif isinstance(v, str):
                if len(v) < 6:
                    valuestr += '{0:<6} '.format(v)
                else:
                    valuestr += '{0:<10} '.format(v)
            elif isinstance(v, float):
                valuestr += '{0:<10} '.format(v)
            else:
                raise NotImplementedError("Unknown value type", keyword, value)
        fd.write(" {0:<30} {1}".format(keyword, valuestr))
    else:
        raise NotImplementedError("Unknown value type", keyword, value)
    fd.write("\n")


def write_matrix_keyword(fd, keyword, value):
    keyword = keyword.replace('_', '.')
    fd.write('<' + keyword + "\n")
    for val in value:
        fd.write("   ")
        valuestr = ''
        for v in val:
            if isinstance(v, bool):
                valuestr += omx_bl[v]
            elif isinstance(v, int):
                valuestr += '{0:<5} '.format(v)
            elif isinstance(v, str):
                if len(v) < 6:
                    valuestr += '{0:<6} '.format(v)
                else:
                    valuestr += '{0:<10} '.format(v)
            elif isinstance(v, float):
                valuestr += '{0:<10} '.format(v)
            else:
                valuestr += '{0:<10} '.format(v)
        fd.write(valuestr.strip() + "\n")
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
    # 'atoms_speciesandcoordinates' overrides
    key = 'atoms_speciesandcoordinates'
    if parameters.get(key) is not None:
        atoms_speciesandcoordinates = parameters[key].copy()
        write_matrix_keyword(fd, key, atoms_speciesandcoordinates)
        return None

    unit = parameters.get(key + '_unit', 'ang').lower()
    if unit == 'ang':
        positions = atoms.get_positions()
    elif unit == 'frac':
        positions = atoms.get_scaled_positions(wrap=False)
    elif unit == 'au':
        positions = atoms.get_positions() / Bohr

    atoms_speciesandcoordinates = []
    # Appending number and elemental symbol
    elements = atoms.get_chemical_symbols()
    for i, element in enumerate(elements):
        atoms_speciesandcoordinates.append([str(i + 1), element])
    # Positions
    for i, position in enumerate(positions):
        atoms_speciesandcoordinates[i].extend(position)

    # Valence electron read
    pattern = r'valence\.electron\s+(\S+)'
    vps_path = parameters.get('data_path')
    atomic_species = parameters.get('definition_of_atomic_species')
    valance_electron = {}
    for atomic_vps in atomic_species:
        sym, orb, pp = atomic_vps
        vpsname = vps_path + '/VPS/' + pp + '.vps'
        with open(vpsname, 'r') as fd2:
            vpstxt = fd2.read()
        match = re.search(pattern, vpstxt, re.M)
        valance_electron[sym] = float(match.group(1))

    magmoms = atoms.get_initial_magnetic_moments()
    # Magnetic moments
    for i, magmom in enumerate(magmoms):
        up = valance_electron[atoms[i].symbol] / 2.
        down = up
        up += magmom / 2
        down += magmom / 2
        atoms_speciesandcoordinates[i].extend([up, down])

    write_matrix_keyword(fd, key, atoms_speciesandcoordinates)


def write_species_number(fd, atoms, parameters, **kwargs):
    """Write a number of species to `fd`

    >>> import sys
    >>> from ase.atoms import Atoms
    >>> write_species_number(sys.stdout, Atoms('CH4'), {})
     species.number                 2
    >>> write_species_number(sys.stdout, Atoms('CH4'), {'species_number': 4})
     species.number                 4

    Parameters:
            fd:  (FileIO or StdIO):
                An IO object to write down
            atoms: :class:`~ase.Atoms` object
            parameters: dict
                Dictionary containing OpenMX keywords value pair

    """
    key = 'species_number'
    value = parameters.get(key)
    if value is None:
        value = len(set(atoms.symbols))
    write_keyword(fd, key, value)


def write_atoms_number(fd, atoms, parameters, **kwargs):
    """Write a number of atoms to `fd`

    >>> import sys
    >>> from ase.atoms import Atoms
    >>> write_atoms_number(sys.stdout, Atoms('CH4'), {})
     atoms.number                   5
    >>> write_atoms_number(sys.stdout, Atoms('CH4'), {'atoms_number': 8})
     atoms.number                   8

    Parameters:
            fd:  (FileIO or StdIO):
                An IO object to write down
            atoms: :class:`~ase.Atoms` object
            parameters: dict
                Dictionary containing OpenMX keywords value pair

    """
    key = 'atoms_number'
    value = parameters.get(key)
    if value is None:
        value = len(atoms)
    write_keyword(fd, key, value)


def write_definition_of_atomic_species(fd, atoms, parameters, **kwargs):
    """ Write down `definition_of_atomic_species` to `fd`

    >>> import sys
    >>> from ase.atoms import Atoms
    >>> param = {'definition_of_atomic_species':
    ... [['H','H5.0-s1>1p1>1','H_CA13'],
    ... ['C','C5.0-s1>1p1>1','C_CA13']]}
    >>> write_definition_of_atomic_species(sys.stdout, Atoms(), param)
    <definition.of.atomic.species
       H      H5.0-s1>1p1>1 H_CA13
       C      C5.0-s1>1p1>1 C_CA13
    definition.of.atomic.species>
    <BLANKLINE>

    Further more, you can specify the wannier information here.
    A. Define local functions for projectors
      Since the pseudo-atomic orbitals are used for projectors,
      the specification of them is the same as for the basis functions.
      An example setting, for silicon in diamond structure, is as following:
    >>> param = {'definition_of_atomic_species':
    ... [['Si', 'Si7.0-s2p2d1', 'Si_CA13'],
    ... ['proj1', 'Si5.5-s1p1d1f1', 'Si_CA13']]}
    >>> write_definition_of_atomic_species(sys.stdout, Atoms(), param)
    <definition.of.atomic.species
       Si     Si7.0-s2p2d1 Si_CA13
       proj1  Si5.5-s1p1d1f1 Si_CA13
    definition.of.atomic.species>
    <BLANKLINE>

    Parameters:
            fd:  (FileIO or StdIO):
                An IO object to write down
            atoms: :class:`~ase.Atoms` object
            parameters: dict
                Dictionary containing OpenMX keywords value pair

    """
    key = 'definition_of_atomic_species'
    if parameters[key] is None:
        raise NotImplementedError("Must specify", key)
    write_matrix_keyword(fd, key, parameters[key])


def write_atoms_unitvectors(fd, atoms, parameters, **kwargs):
    """ Write down `atoms_unitvectors` to `fd`

    >>> import sys
    >>> from ase.atoms import Atoms
    >>> atoms = Atoms('Al', cell=[5, 5, 5])
    >>> write_atoms_unitvectors(sys.stdout, atoms, {})
    >>> write_atoms_unitvectors(sys.stdout, atoms,
    ... {'scf_eigenvaluesolver': 'Band'})
    <atoms.unitvectors
       5.0        0.0        0.0
       0.0        5.0        0.0
       0.0        0.0        5.0
    atoms.unitvectors>
    <BLANKLINE>

    Parameters
    ----------
    fd:  (FileIO or StdIO)
        An IO object to write down
    atoms: :class:`~ase.Atoms` object
    parameters: dict
        Dictionary containing OpenMX keywords value pair

    """
    key = "atoms_unitvectors"
    if parameters.get(key) is not None:
        write_matrix_keyword(fd, key, parameters[key])
        return None

    # Cluster case
    scf_eigenvaluesolver = parameters.get('scf_eigenvaluesolver')
    if scf_eigenvaluesolver is None or scf_eigenvaluesolver == 'cluster':
        return None

    unit = parameters.get('atoms_unitvectors_unit', 'ang').lower()
    write_matrix_keyword(fd, key, atoms.cell / units[unit])


def parse_openmx_log_cell(txt, version='3.9.2'):
    """Parse cell info from the `.log` text

    >>> import numpy as np
    >>> txt = '''
    ... lattice vectors (bohr)
    ... A  = 18.897259885789,  0.000000000000,  0.000000000000
    ... B  =  0.000000000000, 18.897259885789,  0.000000000000
    ... C  =  0.000000000000,  0.000000000000, 18.897259885789
    ... reciprocal lattice vectors (bohr^-1)
    ... '''
    >>> np.all(np.isclose(parse_openmx_log_cell(txt), 10. * np.identity(3)))
    True
    """
    pattern = r'lattice vectors \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1).lower()
    fp = match.end(0)
    ep = re.search(r'reciprocal', txt[fp:]).start(0)
    cell = []
    A = re.search(r'^A\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    B = re.search(r'^B\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    C = re.search(r'^C\s+=(.+)', txt[fp:fp+ep], re.M).group(1).split(',')
    cell.append([float(l) * units[unit] for l in A])
    cell.append([float(l) * units[unit] for l in B])
    cell.append([float(l) * units[unit] for l in C])
    return cell


def parse_openmx_log_pbc(txt, version='3.9.2'):
    """ Return whether it is periodic boundary (Band) or not (Cluster)

    >>> txt = '''
    ... <Band_DFT>  DM, time=0.002782
    ...     1    C  MulP   2.0249  2.0249 sum   4.0499
    ...     2    H  MulP   0.4978  0.4978 sum   0.9956
    ...     3    H  MulP   0.4978  0.4978 sum   0.9956
    ...     4    H  MulP   0.4897  0.4897 sum   0.9794
    ...     5    H  MulP   0.4897  0.4897 sum   0.9794
    ...  Sum of MulP: up   =     4.00000 down          =     4.00000
    ... '''
    >>> parse_openmx_log_pbc(txt)
    True
    >>> txt = '''
    ... <Cluster> time=0.002782
    ...     1    C  MulP   2.0249  2.0249 sum   4.0499
    ...     2    H  MulP   0.4978  0.4978 sum   0.9956
    ...     3    H  MulP   0.4978  0.4978 sum   0.9956
    ...     4    H  MulP   0.4897  0.4897 sum   0.9794
    ...     5    H  MulP   0.4897  0.4897 sum   0.9794
    ...  Sum of MulP: up   =     4.00000 down          =     4.00000
    ... '''
    >>> parse_openmx_log_pbc(txt)
    False

    """
    pattern = r'<Band_DFT>  DM,'
    match = re.search(pattern, txt)
    if match is not None:
        return True
    else:
        return False


def parse_openmx_log_symbols(txt, version='3.9.2'):
    """ Returns chemical symbols from `.log` file

    >>> txt = '''
    ... <Band_DFT>  DM, time=0.002782
    ...     1    C  MulP   2.0249  2.0249 sum   4.0499
    ...     2    H  MulP   0.4978  0.4978 sum   0.9956
    ...     3    H  MulP   0.4978  0.4978 sum   0.9956
    ...     4    H  MulP   0.4897  0.4897 sum   0.9794
    ...     5    H  MulP   0.4897  0.4897 sum   0.9794
    ...  Sum of MulP: up   =     4.00000 down          =     4.00000
    ... '''
    >>> parse_openmx_log_symbols(txt)
    ['C', 'H', 'H', 'H', 'H']
    >>> txt = '''
    ... <Cluster> time=0.002782
    ...     1    C  MulP   2.0249  2.0249 sum   4.0499
    ...     2    H  MulP   0.4978  0.4978 sum   0.9956
    ...     3    H  MulP   0.4978  0.4978 sum   0.9956
    ...     4    H  MulP   0.4897  0.4897 sum   0.9794
    ...     5    H  MulP   0.4897  0.4897 sum   0.9794
    ...  Sum of MulP: up   =     4.00000 down          =     4.00000
    ... '''
    >>> parse_openmx_log_symbols(txt)
    ['C', 'H', 'H', 'H', 'H']
    """

    pattern = r'<Band_DFT>  DM,'
    match = re.search(pattern, txt)
    if match is not None:
        fp = match.end(0)
    else:
        pattern = r'<Cluster>'
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
    """ Parse the energy info from the `.log` file

    >>> txt = '''
    ... *******************************************************
    ...                 Total Energy (Hartree) at MD = 1
    ... *******************************************************
    ...
    ...  Uele  =      -3.340914999307
    ...
    ...  Ukin  =       5.935026847220
    ...  UH0   =     -14.382433025217
    ...  UH1   =       0.031375557056
    ...  Una   =      -5.140556701686
    ...  Unl   =      -0.195114690874
    ...  Uxc0  =      -1.588410771623
    ...  Uxc1  =      -1.588410771623
    ...  Ucore =       8.803238883717
    ...  Uhub  =       0.000000000000
    ...  Ucs   =       0.000000000000
    ...  Uzs   =       0.000000000000
    ...  Uzo   =       0.000000000000
    ...  Uef   =       0.000000000000
    ...  UvdW  =       0.000000000000
    ...  Uch   =       0.000000000000
    ...  Utot  =      -8.125284673032
    ...  '''
    >>> import numpy as np
    >>> np.isclose(parse_openmx_log_energy(txt), -221.10025779574835)
    True

    """
    pattern = r'Total Energy \((\S+)\)'
    unit = re.search(pattern, txt, re.M).group(1).lower()
    pattern = r'Utot  =\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    return float(match.group(1)) * units[unit]


def parse_openmx_log_forces(txt, version='3.9.2'):
    """
    >>> import numpy as np
    >>> txt= '''
    ... *******************************************************
    ...          MD or geometry opt. at MD = 4
    ... *******************************************************
    ...
    ... <Steepest_Descent>  SD_scaling= 1.259817324092
    ... <Steepest_Descent>  |Maximum force| (Hartree/Bohr) = 0.009143946261
    ... <Steepest_Descent>  Criterion       (Hartree/Bohr) = 0.000100000000
    ...
    ... atom=  1, XYZ(ang) Fxyz(a.u.)=  0.0  0.0  0.0  0.0000  0.0000   0.0024
    ... atom=  2, XYZ(ang) Fxyz(a.u.)=  0.6  0.6  0.6  0.0059  0.0059   0.0037
    ... atom=  3, XYZ(ang) Fxyz(a.u.)= -0.6 -0.6  0.6 -0.0059 -0.0059   0.0037
    ... atom=  4, XYZ(ang) Fxyz(a.u.)= -0.6  0.6 -0.6 -0.0029  0.0029  -0.0049
    ... atom=  5, XYZ(ang) Fxyz(a.u.)=  0.6 -0.6 -0.6  0.0029 -0.0029  -0.0049
    ... '''
    >>> res = np.array(
    ...     [[0.0, 0.0, 0.12341296101715354],
    ...     [0.3033901958338358, 0.3033901958338358, 0.1902616482347784],
    ...     [-0.3033901958338358, -0.3033901958338358, 0.1902616482347784],
    ...     [-0.14912399456239386, 0.14912399456239386, -0.25196812874335517],
    ...     [0.14912399456239386, -0.14912399456239386, -0.25196812874335517]])
    >>> np.all(np.isclose(parse_openmx_log_forces(txt), res))
    True
    >>> txt = '''
    ... *******************************************************
    ...              MD or geometry opt. at MD = 5
    ... *******************************************************
    ...
    ... <DIIS>  |Maximum force| (Hartree/Bohr) = 0.004207954148
    ... <DIIS>  Criterion       (Hartree/Bohr) = 0.000100000000
    ...
    ...      atom=   1, XYZ(ang) Fxyz(a.u.)=  0.0  0.0  0.0  0.1   0.2   0.3
    ...      atom=   2, XYZ(ang) Fxyz(a.u.)=  0.6  0.6  0.6  0.1   0.2   0.3
    ...      atom=   3, XYZ(ang) Fxyz(a.u.)= -0.6 -0.6  0.6 -0.1  -0.2   0.3
    ...      atom=   4, XYZ(ang) Fxyz(a.u.)= -0.6  0.6 -0.6 -0.1   0.2  -0.3
    ...      atom=   5, XYZ(ang) Fxyz(a.u.)=  0.6 -0.6 -0.6  0.1  -0.2  -0.3
    ... '''
    >>> res = np.array(
    ... [[5.1422067090480645, 10.284413418096129, 15.426620127144194],
    ...  [5.1422067090480645, 10.284413418096129, 15.426620127144194],
    ...  [-5.1422067090480645, -10.284413418096129, 15.426620127144194],
    ...  [-5.1422067090480645, 10.284413418096129, -15.426620127144194],
    ...  [5.1422067090480645, -10.284413418096129, -15.426620127144194]])

    >>> np.all(np.isclose(parse_openmx_log_forces(txt), res))
    True

    """
    pattern = r'Fxyz\((\S+)\)=(.+)'
    forces = []
    for m in re.finditer(pattern, txt, re.M):
        unit = m.group(1).lower()
        lines = m.group(2).split()[-3:]
        if unit == 'a.u.':
            unit = 'hartree/bohr'
        force = [float(l) * units[unit] for l in lines]
        forces.append(force)
    return forces


def parse_openmx_log_stress(txt, version='3.9.2'):
    """
    >>> txt = '''
    ... *******************************************************
    ...                Stress tensor (Hartree/bohr^3)
    ... *******************************************************
    ...
    ...        0.00000862       -0.00000865        0.00000000
    ...       -0.00000865        0.00000862        0.00000000
    ...        0.00000000        0.00000000        0.00001181
    ...
    ... *******************************************************
    ... '''
    >>> import numpy as np
    >>> res = np.array([[ 0.0015829 , -0.00158841,  0.        ],
    ...                 [-0.00158841,  0.0015829 ,  0.        ],
    ...                 [ 0.        ,  0.        ,  0.00216869]])
    >>> np.all(np.isclose(parse_openmx_log_stress(txt), res))
    True

    """
    pattern = r'Stress tensor \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1).lower()
    fp = match.end(0)
    fp += re.search(r'[\r\n]', txt[fp:]).end(0)
    fp += re.search(r'[\r\n]', txt[fp:]).end(0)
    ep = fp + re.search(r'\*', txt[fp:]).start(0)
    raw = re.split(r'\n', txt[fp:ep])
    A = [float(val) for val in raw[1].split()]
    B = [float(val) for val in raw[2].split()]
    C = [float(val) for val in raw[3].split()]
    return np.array([A, B, C]) * units[unit]


def parse_openmx_log_version(txt, version='3.9.2'):
    """
    >>> txt = '''
    ... *******************************************************
    ... *******************************************************
    ...  Welcome to OpenMX   Ver. 3.9.2
    ...  Copyright (C), 2002-2019, T. Ozaki
    ...  OpenMX comes with ABSOLUTELY NO WARRANTY.
    ...  This is free software, and you are welcome to
    ...  redistribute it under the constitution of the GNU-GPL.
    ... *******************************************************
    ... *******************************************************'''
    >>> parse_openmx_log_version(txt)
    '3.9.2'
    """
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
    from ase.io.formats import string2index
    if isinstance(index, int):
        index = [index]
    elif isinstance(index, str):
        # index = np.s_[:]
        index = string2index(index)
    elif isinstance(index, (list, slice)):
        pass
    else:
        raise NotImplementedError('Index err', index)


    if isinstance(filename, IOBase):
        fd = filename
        txt = fd.read()
    else:
        with open(filename, 'r') as fd:
            txt = fd.read()



    version = parse_openmx_log_version(txt)
    steps, partition = parse_openmx_log_steps(txt, version=version,
                                              return_partition=True)

    md = False
    if len(steps) != 1:
        md = True

    images = []
    idx = np.arange(len(steps))[index]

    for i in idx:
        text = txt[partition[i]:partition[i+1]]

        cell = parse_openmx_log_cell(text, version=version)
        symbols = parse_openmx_log_symbols(text, version=version)
        if md:
            positions = parse_openmx_log_positions(text, version=version)
        else:
            positions = np.zeros((len(symbols), 3))

        energy = parse_openmx_log_energy(text, version=version)
        forces = parse_openmx_log_forces(text, version=version)
        # stress = parse_openmx_log_stress(text, version=version)

        pbc = parse_openmx_log_pbc(text, version=version)

        results = {'energy': energy, 'forces': forces}

        atoms = Atoms(symbols, positions=positions,
                      cell=cell, pbc=pbc)

        calc = SinglePointDFTCalculator(
            atoms, **results)
        atoms.calc = calc
        atoms.calc.name = 'openmx2'

        images.append(atoms)
    return images


def parse_openmx_out_scf_stress_tensor(txt, version='3.9.2'):
    pattern = r'scf\.stress\.tensor\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    if match is None:
        return None
    return match.group(1)


def parse_openmx_out_md_maxiter(txt, version='3.9.2'):
    """ Return Md.maxiter from given text

    """
    pattern = r'md\.maxiter\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    if match is None:
        return None
    return int(match.group(1))


def parse_openmx_out_cell(txt, version='3.9.2'):
    """ Parse the `cell` from the `.out` text

    >>> txt = '''
    ... atoms.unitvectors.unit    Ang
    ... <atoms.unitvectors
    ...    10.0     0.0      0.0
    ...    0.0      10.0     0.0
    ...    0.0      0.0      10.0
    ... atoms.unitvectors>'''
    >>> parse_openmx_out_cell(txt)
    [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]

    """
    pattern = r"atoms\.unitvectors\.unit\s+(\S+)"
    match = re.search(pattern, txt, re.M)

    if match is None:
        unit = 1.
    else:
        unit = units[match.group(1).lower()]

    pattern1 = r'<atoms\.unitvectors'
    pattern2 = r'atoms\.unitvectors>'
    match = re.search(pattern1, txt, re.M)
    if match is None:
        return None
    fp = match.end(0)
    ep = re.search(pattern2, txt[fp:], flags=re.M | re.I).start(0)

    lines = txt[fp:fp+ep].split('\n')[1:-1]
    cell = []
    cell.append([float(l) * unit for l in lines[0].split()])
    cell.append([float(l) * unit for l in lines[1].split()])
    cell.append([float(l) * unit for l in lines[2].split()])
    return cell


def parse_openmx_out_pbc(txt, version='3.9.2'):
    """ Parse the periodic boundary contion from `.out`
    >>> txt = ''
    >>> parse_openmx_out_pbc(txt)
    False
    >>> txt = '''
    ...  scf.eigenvaluesolver           band
    ... '''
    >>> parse_openmx_out_pbc(txt)
    True
    >>> txt = '''
    ...  scf.eigenvaluesolver           cluster
    ... '''
    >>> parse_openmx_out_pbc(txt)
    False

    """
    pattern = r'scf\.eigenvaluesolver\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    if match is None:
        return False
    elif match.group(1).lower() == 'cluster':
        return False
    return True


def parse_openmx_out_symbols(txt, version='3.9.2'):
    """
    >>> txt = '''
    ... ***********************************************************
    ... ***********************************************************
    ...        xyz-coordinates (Ang) and forces (Hartree/Bohr)
    ... ***********************************************************
    ... ***********************************************************
    ...
    ... <coordinates.forces
    ...   5
    ...     1     C     0.00000   0.00000   0.10000   0.00000  0.00 -0.077
    ...     2     H     0.68279   0.68279   0.68279   0.00011  0.00  0.006
    ...     3     H    -0.68279  -0.68279   0.68279  -0.00011 -0.00  0.006
    ...     4     H    -0.68279   0.68279  -0.68279   0.02249 -0.02  0.031
    ...     5     H     0.68279  -0.68279  -0.68279  -0.02249  0.02  0.031
    ... coordinates.forces>'''
    >>> parse_openmx_out_symbols(txt)
    ['C', 'H', 'H', 'H', 'H']

    """

    pattern1 = r'<coordinates\.forces'
    pattern2 = r'coordinates\.forces>'
    fp = re.search(pattern1, txt, re.M).end(0)
    ep = re.search(pattern2, txt[fp:], re.M).start(0)

    symbols = []
    lines = txt[fp:fp+ep].split('\n')[1:-1]
    N = int(lines[0])
    for i in range(N):
        symbols.append(lines[i+1].split()[1])
    return symbols


def parse_openmx_out_positions(txt, version='3.9.2'):
    """
    >>> txt = '''
    ... ***********************************************************
    ... ***********************************************************
    ...        xyz-coordinates (Ang) and forces (Hartree/Bohr)
    ... ***********************************************************
    ... ***********************************************************
    ...
    ... <coordinates.forces
    ...   5
    ...     1     C     0.00000   0.00000   0.10000   0.00  0.00 -0.0776
    ...     2     H     0.68279   0.68279   0.68279   0.00  0.00  0.0068
    ...     3     H    -0.68279  -0.68279   0.68279  -0.00 -0.00  0.0061
    ...     4     H    -0.68279   0.68279  -0.68279   0.02 -0.02  0.0312
    ...     5     H     0.68279  -0.68279  -0.68279  -0.02  0.02  0.0310
    ... coordinates.forces>'''
    >>> import numpy as np
    >>> res = np.array([[0.0, 0.0, 0.1],
    ...                 [0.68279, 0.68279, 0.68279],
    ...                 [-0.68279, -0.68279, 0.68279],
    ...                 [-0.68279, 0.68279, -0.68279],
    ...                 [0.68279, -0.68279, -0.68279]])
    >>> np.all(np.isclose(parse_openmx_out_positions(txt), res))
    True

    """

    pattern = r'xyz-coordinates \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1).lower()
    fp = match.end(0)

    pattern1 = r'<coordinates\.forces'
    pattern2 = r'coordinates\.forces>'
    fp += re.search(pattern1, txt[fp:], re.M).end(0)
    ep = re.search(pattern2, txt[fp:], re.M).start(0)

    positions = []
    lines = txt[fp:fp+ep].split('\n')[1:-1]
    N = int(lines[0])
    for i in range(N):
        line = lines[i+1].split()
        positions.append([float(l) for l in line[2:5]])
    return np.array(positions) * units[unit]


def parse_openmx_out_energy(txt, version='3.9.2'):
    pattern = r'Total energy \((\S+)\)'
    match = re.search(pattern, txt, re.M)
    unit = match.group(1).lower()
    fp = match.end(0)

    pattern = r'Utot\.\s+(\S+)'
    return float(re.search(pattern, txt[fp:], re.M).group(1)) * units[unit]


def parse_openmx_out_forces(txt, version='3.9.2'):
    """
    >>> txt = '''
    ... ***********************************************************
    ... ***********************************************************
    ...        xyz-coordinates (Ang) and forces (Hartree/Bohr)
    ... ***********************************************************
    ... ***********************************************************
    ...
    ... <coordinates.forces
    ...   5
    ...     1  C  0.0  0.0  0.1  0.000000491367  0.000000491392 -0.077373306556
    ...     2  H  0.6  0.6  0.6  0.000114113769  0.000114113769  0.006806503848
    ...     3  H -0.6 -0.6  0.6 -0.000114706014 -0.000114706027  0.006807122741
    ...     4  H -0.6  0.6 -0.6  0.022499289495 -0.022499188566  0.031873812162
    ...     5  H  0.6 -0.6 -0.6 -0.022499188589  0.022499289460  0.031873812150
    ... coordinates.forces>'''
    >>> import numpy as np
    >>> res = np.array(
    ... [[2.5267106840048202e-05, 2.5268392391725462e-05, -3.978695360734958],
    ... [0.005867965885465611, 0.005867965885465611, 0.3500044975234707],
    ... [-0.005898420347589612, -0.005898421016076484, 0.3500363222808385],
    ... [1.1569599739000365, -1.156954783922227, 1.6390173074237417],
    ... [-1.1569547851049347, 1.156959972100264, 1.6390173068066771]])
    >>> np.all(np.isclose(parse_openmx_out_forces(txt), res))
    True

    """
    pattern = r'and forces \((\S+)\)'
    unit = re.search(pattern, txt, re.M).group(1).lower()

    pattern1 = r'<coordinates\.forces'
    pattern2 = r'coordinates\.forces>'
    fp = re.search(pattern1, txt, re.M).end(0)
    ep = re.search(pattern2, txt[fp:], re.M).start(0)

    forces = []
    lines = txt[fp:fp+ep].split('\n')[1:-1]
    N = int(lines[0])
    for i in range(N):
        line = lines[i+1].split()
        forces.append([float(l) * units[unit] for l in line[5:8]])
    return forces


def parse_openmx_out_version(txt):
    """
    >>> txt = '''
    ... ***********************************************************
    ... ***********************************************************
    ...
    ...   This calculation was performed by OpenMX Ver. 3.9.2
    ...   using 1 MPI processes and 1 OpenMP threads.
    ...
    ...   Wed Nov 24 19:38:35 2021
    ...
    ... ***********************************************************
    ... ***********************************************************
    ... '''
    >>> parse_openmx_out_version(txt)
    '3.9.2'
    """
    pattern = r'This calculation was performed by OpenMX\s+Ver\.\s+(\S+)'
    match = re.search(pattern, txt, re.M)
    version = match.group(1)
    return version


def read_openmx_out(filename='openmx.out'):
    """
    return atoms with results in it
    """
    if isinstance(filename, IOBase):
        fd = filename
        txt = fd.read()
    else:
        with open(filename, 'r') as fd:
            txt = fd.read()

    version = parse_openmx_out_version(txt)
    cell = parse_openmx_out_cell(txt, version=version)
    symbols = parse_openmx_out_symbols(txt, version=version)
    positions = parse_openmx_out_positions(txt, version=version)
    energy = parse_openmx_out_energy(txt, version=version)
    forces = parse_openmx_out_forces(txt, version=version)

    pbc = parse_openmx_out_pbc(txt, version=version)

    results = {'energy': energy, 'forces': forces}

    atoms = Atoms(symbols, positions=positions,
                  cell=cell, pbc=pbc)

    calc = SinglePointDFTCalculator(
        atoms, **results)
    atoms.calc = calc
    atoms.calc.name = 'openmx2'

    return atoms


if __name__ == "__main__":
    import doctest
    doctest.testmod()
