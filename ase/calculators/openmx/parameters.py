"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2017 Charles Thomas Johnson ,Jae Hwan Shim and JaeJun Yu

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
from __future__ import print_function
import os
import numpy as np
from ase.calculators.openmx.dos import DOS
from ase.calculators.openmx.band import Band
from ase.calculators.calculator import Parameters
from ase.calculators.openmx.read_functions import ReadFunction
from ase.calculators.openmx.default_settings import default_dictionary
from ase.calculators.openmx.default_settings import default_kpath
from ase.dft.kpoints import special_points, special_paths
from ase.units import Bohr, Ha, Ry, fs


class OpenMXParameters(Parameters):
    """Parameters class for the calculator.
    Documented in BaseOpenMX.__init__
    """
    allowed_xc = [
        'LDA',
        'GGA',
        'PBE',
        'GGA-PBE'
        'LSDA',
        'LSDA-PW'
        'LSDA-CA'
        'CA',
        'PW',
    ]

    unit_dat_keywords = {
        'Hubbard.U.Values': 'eV',
        'scf.Constraint.NC.Spin.v': 'eV',
        'scf.ElectronicTemperature': 'K',        # not an ase unit!
        'scf.energycutoff': 'Ry',
        'scf.criterion': 'Ha',
        'scf.Electric.Field': 'GV / m',          # not an ase unit!
        '1DFFT.EnergyCutoff': 'Ry',
        'orbitalOpt.criterion': '(Ha/Borg)**2',  # not an ase unit!
        'MD.Opt.criterion': 'Ha/Bohr',
        'MD.TempControl': 'K',                   # not an ase unit!
        'NH.Mass.HeatBath': '_amu',
        'MD.Init.Velocity': 'm/s',
        'Dos.Erange': 'eV'
                         }
    allowed_dat_keywords = [
        'System.CurrentDir',             # Implemented
        'System.Name',                   # Implemented
        'DATA.PATH',                     # Implemented
        'level.of.stdout',               # Implemented
        'level.of.fileout',              # Implemented
        'Species.Number',                # Implemented
        'Definition.of.Atomic.Species',  # Implemented
        'Atoms.Number',                  # Implemented
        'Atoms.SpeciesAndCoordinates',   # Implemented
        'Atoms.UnitVectors.Unit',        # Implemented
        'Atoms.UnitVectors',             # Implemented
        'scf.XcType',                    # Implemented
        'scf.spinpolarization',          # Implemented
        'scf.spinorbit.coupling'         # Implemented
        'scf.partialCoreCorrection'
        'scf.Hubbard.U',                 # Implemented
        'scf.Hubbard.Occupation',        # Implemented
        'scf.Hubbard.U.values',          # Implemented
        'scf.Constraint.NC.Spin',        # Implemented
        'scf.Constraint.NC.Spin.v',      # Implemented
        'scf.ElectronicTemperature',     # Implemented
        'scf.energycutoff',              # Implemented
        'scf.Ngrid',
        'scf.maxIter',                   # Implemented
        'scf.EigenvalueSolver',          # Implemented
        'scf.Kgrid',                     # Implemented
        'scf.ProExpn.VNA',
        'scf.Mixing.Type',               # Implemented
        'scf.Init.Mixing.Weight',        # Implemented
        'scf.Min.MixingWeight',          # Implemented
        'scf.Kerker.factor',
        'scf.Mixing.History',            # Implemented
        'scf.Mixing.StartPulay',         # Implemented
        'scf.Mixing.EveryPulay',
        'scf.criterion',                 # Implemented
        'scf.Electric.Field',
        'scf.system.charge',
        '1DFFT.EnergyCutoff',
        '1DFFT.NumGridK',
        '1DFFT.NumGridR',
        'orbitalOpt.Method',
        'orbitalOpt.scf.maxIter',
        'orbitalOpt.Opt.maxIter',
        'orbitalOpt.Opt.Method',
        'orbitalOpt.StartPulay',
        'orbitalOpt.HistoryPulay',
        'orbitalOpt.SD.step',
        'orbitalOpt.criterion',
        'CntOrb.fileout',
        'Num.CntOrb.Atoms',
        'Atoms.Cont.Orbitals',
        'orderN.HoppingRanges',
        'orderN.KrylovH.order',
        'orderN.KrylovS.order',
        'orderN.Exact.Inverse.S',
        'orderN.Recalc.Buffer',
        'orderN.Expand.Core',
        'MD.Type',                      # Implemented
        'MD.Fixed.XYZ',
        'MD.maxIter',                   # Implemented
        'MD.TimeStep',                  # Implemented
        'MD.Opt.criterion',             # Implemented
        'MD.Opt.DIIS.History',
        'MD.Opt.StartDIIS',
        'MD.TempControl',
        'NH.Mass.HeatBath',
        'MD.Init.Velocity',
        'Band.Dispersion',              # Implemented
        'Band.KPath.UnitCell',          # Implemented
        'Band.Nkpath',                  # Implemented
        'Band.kpath',                   # Implemented
        'MO.fileout',                   # Implemented
        'num.HOMOs',                    # Implemented
        'num.LUMOs',                    # Implemented
        'MO.Nkpoint',                   # Implemented
        'MO.kpoint',                    # Implemented
        'Dos.fileout',                  # Implemented
        'Dos.Erange',                   # Implemented
        'Dos.Kgrid',                    # Implemented
        'HS.fileout',
        'Voronoi.charge',
        ]

    def __init__(
            self,
            restart=None,
            energy_cutoff=150 * Ry,  			# eV
            kpts=(4, 4, 4),
            xc='LDA',
            initial_magnetic_moments=None,  		# Bohr magneton,
            total_magnetic_moments=None,
            # overrides the magnetic moments of the Atoms object
            initial_magnetic_moments_euler_angles=None,  # degrees
            nc_spin_constraint_penalty=0,  		# eV
            nc_spin_constraint_euler_angles=None,  	# degrees
            nc_spin_constraint_atom_indices=None,  	# indices of atoms which
            # have constrained spin
            orbital_polarization_enhancement_atom_indices=None,
            magnetic_field=0,  				# Tesla
            smearing=None,  # electronic temperature # ('Fermi-Dirac',300kB)
            scf_max_iter=100,
            eigenvalue_solver='Band',
            mixing_type='Rmm-Diis',
            scf_init_mixing_weight=None,        # ... = 0.3
            min_mixing_weight=None,             # ... = 0.001
            max_mixing_weight=None,             # ... = 0.4
            mixing_history=None,                # ... = 5
            mixing_start_pulay=None,            # ... = 6
            kerker_factor=None,
            mixing_every_pulay=None,			# ... = 1
            scf_criterion=1e-6 * Ha,  			# eV
            md_type=None,
            md_maxiter=1,
            time_step=0.5 * fs,  				# s
            md_criterion=1e-4 * Ha / Bohr,  	# eV / Ang
            dos_erange=None,  			       	# eV
            dos_kgrid=None,
            band_dispersion=False,
            band_resolution=20,
            band_kpath=None,
            hubbard_occupation='dual',			# ... = 'dual'
            hubbard_u_values=None,
            fileout=1,
            stdout=1,
            homos=0,
            lumos=0,
            mo_kpts=None,
            absolute_path_of_vesta=None,
            species=tuple(),
            pseudo_qualifier=None,
            dft_data_path=None,
            dft_data_dict=None,
            dat_arguments=None,
            # function to read each line of stdout from OpenMX.
            # Must only take a string as an argument.
            read_function=None,
            _atoms=None,
            stress=False,
            # if specified, .mmn .amn .eig .win files will be produced
            wannier_initial_projectors=False,
            scf_system_charge=None,
            scf_fixed_grid=None,
            scf_restart=None,
            scf_stress_tensor=None,
            scf_spinpolarization=None,
            md_current_iter=None,
            debug=False,
            pbs=False,
            processes=1,
            nohup=False,
            walltime="10:00:00",
            threads=1,
    ):

        try:
            p = _atoms
            if p is not None:
                self.atoms = p
        except KeyError:
            print("Using %s as atoms" % self.atoms)
        if smearing is not None and smearing[0] is not 'Fermi-Dirac':
            print("only Fermi-Dirac smearing is supported")
            if smearing[0] in ['Gaussian', 'Methfessel-Paxton']:
                raise NotImplementedError
            raise ValueError
        if dft_data_path is None:
            try:
                dft_data_path = os.environ['OPENMX_DFT_DATA_PATH']
            except KeyError:
                print('Please either set OPENMX_DFT_DATA_PATH as an enviroment'
                      'variable or specify dft_data_path as a keyword argument'
                      )
        if kpts is None:
            kpts = (1, 1, 1)
        elif type(kpts) is list:
            raise NotImplementedError
        elif type(kpts) == tuple and len(kpts) != 3:
            raise NotImplementedError
        if kpts == (1, 1, 1):
            print("When only the gamma point is considered, the eigenvalue \
                  solver is changed to 'Cluster' with the periodic boundary \
                  condition.")
            eigenvalue_solver = 'Cluster'
        if lumos + homos:
            if mo_kpts is None:
                print('No molecular orbital k-points specified, assuming \
                      (0, 0 ,0)')
                mo_kpts = [(0, 0, 0)]
            no_kpts = False
            try:
                len(mo_kpts[0])
            except TypeError:
                mo_kpts = [mo_kpts]
            except IndexError:
                no_kpts = True
            if eigenvalue_solver.lower() == 'cluster':
                if len(mo_kpts) != 1:
                    print(
                        'For cluster calculation, the number of molecular \
                         orbital k-points must equal 1')
                    if no_kpts:
                        print(
                            'No molecular orbital k-points specified, assuming \
                            (0, 0 ,0)')
                        mo_kpts = [(0, 0, 0)]
                    else:
                        print('Using just the first k-point provided.')
                        mo_kpts = [mo_kpts[0]]
        if dos_kgrid is None:
            dos_kgrid = kpts
        if band_dispersion:
            if band_kpath is None:
                lattice_type = self.get_lattice_type()
                if lattice_type == 'not special':
                    raise Exception(
                        "No default kpath exists for this lattice. Please \
                        specify 'band_kpath'")
                points = special_points[lattice_type]
                path = special_paths[lattice_type]
                band_kpath = []
                npath = len(path)
                i = 0
                for j in range(1, npath):
                    if j == i:
                        continue
                    if path[j] == ',':
                        i = j + 1
                        continue
                    band_kpath.append({
                        'kpts': band_resolution,
                        'start_point': tuple(points[path[i]]),
                        'end_point':   tuple(points[path[j]]),
                        'path_symbols': (path[i], path[j])})
                    i += 1
            else:
                dict_list = band_kpath
                band_kpath = default_kpath
                if type(dict_list) == dict:
                    for i in range(default_kpath):
                        for key in dict_list[i].keys():
                            band_kpath[i][key] = dict_list[key]
                elif len(dict_list) == len(default_kpath):
                    for i in range(len(dict_list)):
                        for key in dict_list[i].keys():
                            band_kpath[i][key] = dict_list[i][key]
                else:
                    band_kpath = dict_list
            band = Band(self)
        if dft_data_dict is None:
            dft_data_dict = default_dictionary
        else:
            dict_dict = dft_data_dict
            dft_data_dict = default_dictionary
            for symbol in dict_dict.keys():
                for key in dict_dict[symbol].keys():
                    dft_data_dict[symbol][key] = dict_dict[symbol][key]
        if dos_erange:
            dos = DOS(self)
        if isinstance(read_function, list):
            read_function = ReadFunction(read_function)
        if read_function is True:
            read_function = ReadFunction()
        if (smearing is not None and smearing[0] != 'Fermi-Dirac'):
            print("only Fermi-Dirac smearing is supported")
            if smearing[0] in ['Gaussian', 'Methfessel-Paxton']:
                raise NotImplementedError
            raise ValueError

        if dft_data_path is None:
            try:
                dft_data_path = os.environ['OPENMX_DFT_DATA_PATH']
            except KeyError:
                raise KeyError(
                    'Please either set OPENMX_DFT_DATA_PATH as an environment \
                    variable or specify dft_data_path as a keyword argument.')
        if xc == 'LDA' and np.any(initial_magnetic_moments is not None):
            raise RuntimeError('LDA does not support spin polarised '
                               'calculations.\n' + 'Please either select a'
                               'different exchange correlation or turn spin'
                               'polarisation off')
        if type(kpts) is list:
            raise NotImplementedError
        elif type(kpts) == tuple and len(kpts) != 3:
            raise NotImplementedError

        if lumos + homos:
            if mo_kpts is None:
                print('No molecular orbital k-points specified, assuming \
                (0, 0, 0)')
                mo_kpts = [(0, 0, 0)]
            no_kpts = False
            try:
                len(mo_kpts[0])
            except TypeError:
                mo_kpts = [mo_kpts]
            except IndexError:
                no_kpts = True
            if eigenvalue_solver.lower() == 'cluster':
                if len(mo_kpts) != 1:
                    print('For cluster calculation, the number of molecular \
                                            orbital k-points must equal 1')
                    if no_kpts:
                        print('No molecular orbital k-points specified, \
                        assuming (0, 0 ,0)')
                        mo_kpts = [(0, 0, 0)]
                    else:
                        print('Using just the first k-point provided.')
                        mo_kpts = [mo_kpts[0]]

        if dos_kgrid is None:
            dos_kgrid = kpts

        if dft_data_dict is None:
            dft_data_dict = default_dictionary
        else:
            dict_dict = dft_data_dict
            dft_data_dict = default_dictionary
            for symbol in dict_dict.keys():
                for key in dict_dict[symbol].keys():
                    dft_data_dict[symbol][key] = dict_dict[symbol][key]

        '''if isinstance(read_function, list):
            read_function = ReadFunction(read_function)
        if read_function is True:
            read_function = ReadFunction()'''
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


class Specie(Parameters):
    """
    Parameters for specifying the behaviour for a single species in the
    calculation. If the tag argument is set to an integer then atoms with
    the specified element and tag will be a separate species.

    Pseudopotential and basis set can be specified. Additionally the species
    can be set be a ghost species, meaning that they will not be considered
    atoms, but the corresponding basis set will be used.
    """
    def __init__(self,
                 symbol,
                 basis_set='DZP',
                 pseudopotential=None,
                 tag=None,
                 ghost=False):
        kwargs = locals()
        kwargs.pop('self')
        Parameters.__init__(self, **kwargs)


def format_dat(key, value):
    """
    Write an dat key-word value pair.

    Parameters:
        - key   : The dat-key
        - value : The dat value.
    """
    if isinstance(value, (list, tuple)) and len(value) == 0:
        return ''

    key = format_key(key)
    new_value = format_value(value)

    if isinstance(value, list):
        string = '<' + key + '\n' +\
            new_value + '\n' + \
            key + '>' + '\n'
    else:
        string = '%s  %s\n' % (key, new_value)

    return string


def format_value(value):
    """
    Format python values to dat-format.

    Parameters:
        - value : The value to format.
    """
    if isinstance(value, tuple):
        sub_values = map(format_value, value)
        value = ' '.join(sub_values)
    elif isinstance(value, list):
        sub_values = map(format_value, value)
        value = '\n'.join(sub_values)
    elif isinstance(value, dict):
        key_list = value.keys()
        key_list.sort()
        value_list = [value[key] for key in key_list]
        sub_values = map(format_value, value_list)
        value = '\t'.join(sub_values)
    else:
        value = str(value)

    return value


def format_key(key):
    """ Fix the dat-key replacing '_' with '.' and '__' with '_' """
    key = key.replace('__', '#')
    key = key.replace('_', '.')
    key = key.replace('#', '_')

    return key
