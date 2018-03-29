"""
    The ASE Calculator for OpenMX <http://www.openmx-square.org>
    A Python interface to the software package for nano-scale
    material simulations based on density functional theories.
    Copyright (C) 2017 Charles Thomas Johnson, Jae Hwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License
    as published by the Free Software Foundation,
    either version 3 of the License, or (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.
    If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from ase.calculators.calculator import ReadError
import os
import time
from subprocess import Popen
from os.path import join
import numpy as np
from ase.units import Bohr, Ha, Ry, kB
from ase.calculators.openmx.import_functions import read_nth_to_last_value
from ase.calculators.openmx.import_functions import cartesian_to_spherical_polar
from ase.calculators.openmx.import_functions import input_command
from ase.calculators.calculator import FileIOCalculator, Calculator
from ase.calculators.calculator import all_changes, equal
from ase.calculators.calculator import kptdensity2monkhorstpack
from ase.calculators.openmx.parameters import Specie, format_dat
from ase.calculators.openmx.parameters import OpenMXParameters
from ase.calculators.openmx.band import Band
from ase.calculators.openmx.dos import DOS
from ase.calculators.openmx.default_settings import default_dictionary
from ase.geometry import cell_to_cellpar
from ase.atoms import Atoms


class OpenMX(FileIOCalculator):
    """Calculator interface to the OpenMX code.
    """

    implemented_properties = (
        'free_energy',       # Same value with energy
        'energy',
        'forces',
        'stress',
        'dipole',
        'chemical_potential',
        'magmom',
        'magmoms',
    )

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
        'scf.SpinOrbit.Coupling',
        'scf.SpinPolarization',          # Implemented
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
        'scf.restart',
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

    non_standard_parameters = {
        'energy_cutoff': Ha,
    }

    default_parameters = OpenMXParameters()

    default_pbs = {
        'processes': 1,
        'walltime': "10:00:00",
        'threads': 1
    }

    default_mpi = {
        'processes': 1,
        'threads': 1
    }

    default_output_setting = {
        'nohup': True,
        'debug': False
    }

    def __init__(self, restart=None, ignore_bad_restart_file=False, label=None,
                 atoms=None, command=None, mpi=None, pbs=None, **kwargs):

        # Please use Trajectory instead
        self.atoms_history = []
        self.energy_history = []
        self.forces_history = []
        self.dipole_history = []

        # Initialize and put the default parameters.
        self.initialize_pbs(pbs)
        self.initialize_mpi(mpi)
        self.initialize_output_setting(**kwargs)

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, command, **kwargs)

    def __getitem__(self, key):
        """Convenience method to retrieve a parameter as
        calculator[key] rather than calculator.parameters[key]

            Parameters:
                -key       : str, the name of the parameters to get.
        """
        return self.parameters[key]

    def __setitem__(self, key, value):
        self.parameters[key] = value

    def initialize_output_setting(self, **kwargs):
        output_setting = {}
        self.output_setting = dict(self.default_output_setting)
        for key, value in kwargs.items():
            if key in self.default_output_setting:
                output_setting[key] = value
        self.output_setting.update(output_setting)
        self.__dict__.update(self.output_setting)

    def initialize_pbs(self, pbs):
        if pbs:
            self.pbs = dict(self.default_pbs)
            for key in pbs:
                if key not in self.default_pbs:
                    allowed = ', '.join(list(self.default_pbs.keys()))
                    raise TypeError('Unexpected keyword "{0}" in "pbs" '
                                    'dictionary.  Must be one of: {1}'
                                    .format(key, allowed))
            # Put dictionary into python variable
            self.pbs.update(pbs)
            self.__dict__.update(self.pbs)
        else:
            self.pbs = None

    def initialize_mpi(self, mpi):
        if mpi:
            self.mpi = dict(self.default_mpi)
            for key in mpi:
                if key not in self.default_mpi:
                    allowed = ', '.join(list(self.default_mpi.keys()))
                    raise TypeError('Unexpected keyword "{0}" in "mpi" '
                                    'dictionary.  Must be one of: {1}'
                                    .format(key, allowed))
            # Put dictionary into python variable
            self.mpi.update(mpi)
            self.__dict__.update(self.mpi)
            print(self.processes)
        else:
            self.mpi = None

    def correct_pbc(self, atoms):
        """
        A method to correct the periodic boundary conditions of an atoms object
        to ensure that they correspond the what OpenMX has interpreted the unit
        cell as.
        """
        return NotImplementedError

    def species(self, atoms):
        """Find all relevant species depending on the atoms object and
        species input.

            Parameters :
                - atoms : An Atoms object.
        """
        # For each element use default specie from the species input, or set
        # up a default species  from the general default parameters.
        symbols = np.array(atoms.get_chemical_symbols())
        tags = atoms.get_tags()
        species = list(self['species'])
        default_species = [
            s for s in species
            if (s['tag'] is None) and s['symbol'] in symbols]
        default_symbols = [s['symbol'] for s in default_species]
        for symbol in symbols:
            if symbol not in default_symbols:
                specie = Specie(
                    symbol=symbol,
                    tag=None)
                default_species.append(specie)
                default_symbols.append(symbol)
        assert len(default_species) == len(np.unique(symbols))

        # Set default species as the first species.
        species_numbers = np.zeros(len(atoms), int)
        i = 1
        for specie in default_species:
            mask = symbols == specie['symbol']
            species_numbers[mask] = i
            i += 1

        # Set up the non-default species.
        non_default_species = [s for s in species if not s['tag'] is None]
        for specie in non_default_species:
            mask1 = (tags == specie['tag'])
            mask2 = (symbols == specie['symbol'])
            mask = np.logical_and(mask1, mask2)
            if sum(mask) > 0:
                species_numbers[mask] = i
                i += 1
        all_species = default_species + non_default_species

        return all_species, species_numbers

    def set(self, **kwargs):
        """Set all parameters.

            Parameters:
                -kwargs  : Dictionary containing the keywords defined in
                           OpenMXParameters.
        """
        # Check the kwargs inputs are Allowed.
        for key, value in kwargs.items():
            if key == 'xc' and value not in self.allowed_xc:
                raise KeyError('Given xc "%s" is not allowed' % value)
            if key in ['dat_arguments'] and isinstance(value, dict):
                # For values that are dictionaries, verify subkeys, too.
                default_dict = self.default_parameters[key]
                for subkey in kwargs[key]:
                    if subkey not in default_dict:
                        allowed = ', '.join(list(default_dict.keys()))
                        raise TypeError('Unknown subkeyword "{0}" of keyword '
                                        '"{1}".  Must be one of: {2}'
                                        .format(subkey, key, allowed))

        # Find out what parameter has been changed
        changed_parameters = {}
        for key, value in kwargs.items():
            oldvalue = self.parameters.get(key)
            if key not in self.parameters or not equal(value, oldvalue):
                changed_parameters[key] = value
                self.parameters[key] = value
        if(self.debug):
            print(' Changed', changed_parameters)

        # Set the parameters
        for key, value in kwargs.items():
            # print(' Setting the %s as %s'%(key, value))
            self.parameters[key] = value

        # If Changed Parameter is Critical, we have to reset the results
        for key, value in changed_parameters.items():
            if key in ['xc', 'kpts', 'energy_cutoff']:
                self.results = {}

        value = kwargs.get('energy_cutoff')
        if value is not None and not (isinstance(value, (float, int))
                                      and value > 0):
            mess = "'%s' must be a positive number(in eV), \
                got '%s'" % ('energy_cutoff', value)
            raise ValueError(mess)

        atoms = kwargs.get('_atoms')
        if atoms is not None and self.atoms is None:
            self.atoms = atoms

    def run(self):
        '''Check Which Running method we r going to use and run it'''
        if self.pbs is not None:
            run = self.run_pbs
        elif self.mpi is not None:
            run = self.run_mpi
        else:
            run = self.run_openmx
        run()

    def run_openmx(self):
        runfile = self.get_file_name('.dat')
        outfile = self.get_file_name('.log')
        olddir = os.getcwd()
        abs_dir = join(olddir, self.directory)
        os.chdir(abs_dir)
        if(self.command is None):
            self.command = 'openmx %s > %s'
        command = self.command
        command = command % (runfile, outfile)
        if(self.debug):
            print(command)
        p = Popen(command, shell=True)
        while p.poll() is None:  # Process strill alive
            if os.path.isfile(outfile):
                with open(outfile, 'r') as f:
                    last_position = 0
                    prev_position = 0
                    f.seek(last_position)
                    new_data = f.read()
                    prev_position = f.tell()
                    # print('pos', prev_position != last_position)
                    if(prev_position != last_position):
                        if(not self.nohup):
                            print(new_data)
                        last_position = prev_position
                    time.sleep(1)
            else:
                if(self.debug):
                    print('Waiting %s file ' % outfile)
                time.sleep(5)
        os.chdir(olddir)
        if(self.debug):
            print("calculation Finished")

    def run_pbs(self, prefix='test'):
        import subprocess
        import os.path
        import os
        import time
        prefix = self.prefix
        olddir = os.getcwd()
        try:
            os.chdir(self.abs_directory)
        except AttributeError:
            os.chdir(self.directory)

        def runCmd(exe):
            p = Popen(exe, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      universal_newlines=True)
            while True:
                line = p.stdout.readline()
                if line != '':
                    # the real code does filtering here
                    yield line.rstrip()
                else:
                    break

        def hasRQJob(jobNum, status='Q'):
            jobs = runCmd('qstat')
            # print('jobs',jobs)
            for line in jobs:
                # print('line=',line)
                # print('jobNum=%s'%jobNum,type(jobNum))
                if jobNum in line:
                    columns = line.split()
                    # print(columns)
            return columns[-2] == status

        inputfile = self.label + '.dat'
        outputfile = self.label + '.log'

        bashArgs = "#!/bin/bash \n cd $PBS_O_WORKDIR\n"
        jobName = prefix
        cmd = bashArgs + \
            'mpirun -hostfile $PBS_NODEFILE openmx %s > %s' % (
                inputfile, outputfile)
        echoArgs = ["echo", "$' %s'" % cmd]
        qsubArgs = ["qsub", "-N", jobName, "-l", "nodes=1:ppn=" +
                    str(self.processes), "-l", "walltime=" + self.walltime]
        wholeCmd = " ".join(echoArgs) + " | " + " ".join(qsubArgs)
        if(self.debug):
            print(wholeCmd)
        out = subprocess.Popen(wholeCmd, shell=True,
                               stdout=subprocess.PIPE, universal_newlines=True)
        out = out.communicate()[0]
        jobId = out.split()[0]
        digit = 0
        for c in jobId:
            if c.isdigit():
                digit += 1
            else:
                break
        jobNum = jobId[:digit]
        if(self.debug):
            print('Queue number is ' + jobNum)
        if(self.debug):
            print('Waiting for the Queue to start')
        while hasRQJob(jobNum, status='Q'):
            time.sleep(5)
            if(self.debug):
                print('.', end='', flush=True)
        if(self.debug):
            print('Start Calculating')
        while hasRQJob(jobNum, status='R'):
            if os.path.isfile(outputfile):
                last_position = 0
                prev_position = 0
                with open(outputfile, "r") as f:
                    while True:
                        f.seek(last_position)
                        new_data = f.read()
                        prev_position = f.tell()
                        if(prev_position != last_position):
                            if(not self.nohup):
                                print(new_data)
                            last_position = prev_position
                        if not hasRQJob(jobNum, status='R'):
                            if(self.debug):
                                print('Calculation Finished')
                            break
                        time.sleep(1)
            else:
                if(self.debug):
                    print('Waiting for the log file come out')
                time.sleep(5)
                # raise ValueError("%s isn't a file!" % outputfile)
        os.chdir(olddir)
        if(self.debug):
            print('Calculation Finished!')
        return jobNum

    def run_mpi(self):
        processes = self.processes
        threads = self.threads
        runfile = self.get_file_name('.dat')
        outfile = self.get_file_name('.log')
        olddir = os.getcwd()
        abs_dir = join(olddir, self.directory)
        os.chdir(abs_dir)
        import subprocess
        self.command = self.get_command(processes, threads, runfile, outfile)
        if(self.debug):
            print(self.command)
        p = subprocess.Popen(self.command, shell=True, universal_newlines=True)
        while p.poll() is None:  # Process strill alive
            if os.path.isfile(outfile):
                with open(outfile, 'r') as f:
                    last_position = 0
                    prev_position = 0
                    f.seek(last_position)
                    new_data = f.read()
                    prev_position = f.tell()
                    # print('pos', prev_position != last_position)
                    if(prev_position != last_position):
                        if(not self.nohup):
                            print(new_data)
                        last_position = prev_position
                    time.sleep(1)
            else:
                if(self.debug):
                    print('Waiting %s file ' % outfile)
                time.sleep(5)
        os.chdir(olddir)
        if(self.debug):
            print("calculation Finished")

    def clean(self, prefix='test', queue_num=None):
        """Method which cleans up after a calculation.

        The default files generated OpenMX will be deleted IF this
        method is called.

        """
        if(self.debug):
            print("Cleaning Data")
        fileName = self.get_file_name('')
        pbs_Name = self.get_file_name('')
        files = [
            # prefix+'.out',#prefix+'.dat',#prefix+'.BAND*',
            fileName + '.cif', fileName + '.dden.cube', fileName + \
            '.ene', fileName + '.md', fileName + '.md2',
            fileName + '.tden.cube', fileName + '.sden.cube', fileName + \
            '.v0.cube', fileName + '.v1.cube',
            fileName + '.vhart.cube', fileName + '.den0.cube', fileName + \
            '.bulk.xyz', fileName + '.den1.cube',
            fileName + '.xyz', pbs_Name + '.o' + \
            str(queue_num), pbs_Name + '.e' + str(queue_num)
        ]
        for f in files:
            try:
                if(self.debug):
                    print("Removing" + f)
                os.remove(f)
            except OSError:
                if(self.debug):
                    print("There is no such file named " + f)
                pass

    def calculate(self, atoms=None, properties=None,
                  system_changes=all_changes):
        """Capture the RuntimeError from FileIOCalculator.calculate
        and     add a little debug information from the OpenMX output.
        See base FileIOCalculator for documentation.
        """
        if(self.debug):
            print("Start Calculation")
        if properties is None:
            properties = self.implemented_properties
        try:
            Calculator.calculate(self, atoms, properties, system_changes)
            self.write_input(self.atoms, properties, system_changes)
            self.read_input()
            self.run()
            self.read_results()
            self.clean()
            if atoms is not None:
                self.update_atoms(atoms)
        except RuntimeError as e:
            try:
                with open(self.get_file_name('.log'), 'r') as f:
                    lines = f.readlines()
                debug_lines = 10
                print('##### %d last lines of the OpenMX output' % debug_lines)
                for line in lines[-20:]:
                    print(line.strip())
                print('##### end of openMX output')
                raise e
            except RuntimeError as e:
                raise e

    def set_initial_magnetic_moments(self, atoms):
        if np.all(self['initial_magnetic_moments'] is None):
            no_magnetic_moments = True
            nc = False
            for magnetic_moment in atoms.get_initial_magnetic_moments():
                try:
                    ml = len(magnetic_moment)
                    if ml == 3:
                        magnetic_moment = np.array(magnetic_moment)
                        if magnetic_moment.dot(magnetic_moment):
                            no_magnetic_moments = False
                        nc = True
                    else:
                        raise TypeError(
                            "Magnetic moments must be 3 dimensional!")
                except TypeError:
                    if magnetic_moment:
                        no_magnetic_moments = False
            if not no_magnetic_moments:
                self['initial_magnetic_moments'] = \
                    atoms.get_initial_magnetic_moments()
                if nc:
                    self['initial_magnetic_moments_euler_angles'] = np.ndarray(
                        len(atoms), tuple)
                    for i in range(len(atoms)):
                        magnetic_moment = self['initial_magnetic_moments'][i]
                        magnetic_moment = cartesian_to_spherical_polar(
                                          magnetic_moment[0],
                                          magnetic_moment[1],
                                          magnetic_moment[2])
                        self['initial_magnetic_moments'][i] = \
                            magnetic_moment[0]
                        self['initial_magnetic_moments_euler_angles'][i] = (
                            magnetic_moment[1], magnetic_moment[2])

    def write_input(self, atoms=None, properties=None, system_changes=None):
        """Write input (dat)-file.
        See calculator.py for further details.

        Parameters:
            - atoms        : The Atoms object to write.
            - properties   : The properties which should be calculated.
            - system_changes : List of properties changed since last run.
        """
        if(self.debug):
            print(' Calling write_input method...')
        if(self.atoms is not None):
            atoms = self.atoms
        if np.all(self['initial_magnetic_moments'] is None):
            no_magnetic_moments = True
            nc = False
            for magnetic_moment in atoms.get_initial_magnetic_moments():
                try:
                    ml = len(magnetic_moment)
                    if ml == 3:
                        magnetic_moment = np.array(magnetic_moment)
                        if magnetic_moment.dot(magnetic_moment):
                            no_magnetic_moments = False
                        nc = True
                    else:
                        raise TypeError(
                            "Magnetic moments must be 3 dimensional!")
                except TypeError:
                    if magnetic_moment:
                        no_magnetic_moments = False
            if not no_magnetic_moments:
                self['initial_magnetic_moments'] = \
                    atoms.get_initial_magnetic_moments()
                if nc:
                    self['initial_magnetic_moments_euler_angles'] = np.ndarray(
                        len(atoms), tuple)
                    for i in range(len(atoms)):
                        magnetic_moment = self['initial_magnetic_moments'][i]
                        magnetic_moment = cartesian_to_spherical_polar(
                            magnetic_moment[0],
                            magnetic_moment[1],
                            magnetic_moment[2])
                        self['initial_magnetic_moments'][i] = \
                            magnetic_moment[0]
                        self['initial_magnetic_moments_euler_angles'][i] = (
                            magnetic_moment[1], magnetic_moment[2])

        # Call base calculator.
        FileIOCalculator.write_input(
            self,
            atoms=atoms,
            properties=properties,
            system_changes=system_changes)
        filename = self.get_file_name('.dat')
        curdir = join(os.getcwd()+self.directory)

        # Start writing the file.
        with open(filename, 'w') as f:
            # First write explicitly given options to
            # allow the user to overwrite anything.
            self._write_dat_arguments(f)
            # Use the saved density matrix if only 'cell' and 'positions'
            # haved changes.
            if (system_changes is None or
                ('numbers' not in system_changes and
                 'initial_magmoms' not in system_changes and
                 'initial_charges' not in system_changes)):
                f.write(format_dat('DM.UseSaveDM', True))

            # Write system current directory and name.
            f.write(format_dat('System.CurrentDirectory', curdir))
            f.write(format_dat('System.Name', self.prefix))
            data_path = self['dft_data_path']
            f.write(format_dat('DATA.PATH', data_path))
            f.write(format_dat('level.of.stdout', self['stdout']))
            f.write(format_dat('level.of.fileout', self['fileout']))
            if self['wannier_initial_projectors']:
                f.write(format_dat('Wannier.Func.Calc', 'on'))
                f.write(format_dat('Wannier90.fileout', 'on'))
                f.write(format_dat('Wannier.Func.Num', self['nwannier']))
                f.write(format_dat('Wannier.Outer.Window.Bottom',
                                   self['wannier_outer_window'][0]))
                f.write(format_dat('Wannier.Outer.Window.Top',
                                   self['wannier_outer_window'][1]))
                f.write(format_dat('Wannier.Inner.Window.Bottom',
                                   self['wannier_inner_window'][0]))
                f.write(format_dat('Wannier.Inner.Window.Top',
                                   self['wannier_inner_window'][1]))
                f.write(format_dat('Wannier.Initial.Guess', 'on'))
                f.write(format_dat('Wannier.MaxShells',
                                   self['wannier_max_shells']))
                if type(self['wannier_kpts']) in [float, int]:
                    kptdensity = self['wannier_kpts']
                    self['kpts'] = tuple(kptdensity2monkhorstpack(
                        atoms, self['wannier_kpts'], False))
                    print("Using a %s Monkhorst pack which is closest to a k \
                          point density of %s Ang^3" %
                          (self['wannier_kpts'], kptdensity))
                f.write(format_dat('Wannier.Kgrid', self['wannier_kpts']))
                f.write(format_dat('Wannier.Minimizing.Max.Steps',
                                   self['wannier_minim_max_steps']))
                f.write(format_dat('Wannier.Minimizing.Conv.Criterion',
                                   self['wannier_minim_criterion']))
                f.write(format_dat('Wannier.Dis.Conv.Criterion',
                                   self['wannier_dis_criterion']))
                f.write(format_dat('Wannier.Dis.SCf.Max.Steps',
                                   self['wannier_dis_max_steps']))
                if self['restart']:
                    f.write(format_dat('Wannier.Readin.Overlap.Matrix', 'on'))
                else:
                    f.write(format_dat('Wannier.Readin.Overlap.Matrix', 'off'))
            # Write the rest.
            self._write_species(f, atoms)
            self._write_structure(f, atoms)
            self._write_scf(f, atoms)
            if self['band_dispersion']:
                self._write_band(f)
            if self['md_type']:
                self._write_md(f)
            if self['md_current_iter'] is not None:
                f.write(format_dat('MD.Current.Iter',
                                   str(self['md_current_iter'])))
            if self['dos_erange']:
                try:
                    len(self['dos_erange'])
                except TypeError:
                    print(
                        'Please specify dos_erange as a tuple of lower and \
                        upper energy bounds')
                    self['dos_erange'] = (float(input('Please specify a lower \
                    energy bound in eV for the DOS calculation: ')),
                                          float(input('And an upper energy \
                                                bound please: ')))
                self._write_dos(f)
            if self['homos'] + self['lumos']:
                self._write_mo(f)
        if(self.debug):
            print('Finished writing input file at %s' % filename)

    def check_state(self, atoms, tol=1e-15):
        system_changes = FileIOCalculator.check_state(self, atoms, tol)

        return system_changes

    def _write_scf(self, f, atoms):
        pa = self.parameters
        pag = pa.get
        if pa.get('restart') is not None:
            f.write(format_dat('scf.restart', 'on'))
        elif pa.get('restart') is None and pa.get('scf_restart') is 'on':
            f.write(format_dat('scf.restart', 'on'))
        elif pa.get('restart') is None and pa.get('scf_restart') is 'off':
            f.write(format_dat('scf.restart', 'off'))
        else:
            f.write(format_dat('scf.restart', 'off'))
        f.write(format_dat('scf.XcType', self.get_xc(pa.get('xc'))))

        if np.all(pa.get('initial_magnetic_moments') is None):
            pass
        elif np.any(pa.get('initial_magnetic_moments') is not None) and \
                np.all(pag('initial_magnetic_moments_euler_angles') is None):
            f.write(format_dat('scf.SpinPolarization', 'On'))
        elif np.any(pag('initial_magnetic_moments_euler_angles') is not None):
            f.write(format_dat('scf.SpinPolarization', 'NC'))
            if pa.get('nc_spin_constraint_penalty'):
                f.write(format_dat('scf.Constraint.NC.Spin', 'ON'))
                f.write(format_dat('scf.Constraint.NC.Spin.v',
                                   pa.get('nc_spin_constraint_penalty')))
            else:
                f.write(format_dat('scf.Constraint.NC.Spin', 'OFF'))
            if pa.get('magnetic_field'):
                f.write(format_dat('scf.NC.Zeeman.Orbital', 'on'))
                f.write(format_dat('scf.NC.Mag.Field.Orbital',
                                   pa.get('magnetic_field')))
                f.write(format_dat('scf.NC.Zeeman.Spin', 'on'))
                f.write(format_dat('scf.NC.Mag.Field.Spin',
                                   pa.get('magnetic_field')))
        if pa.get('hubbard_u_values') is not None:
            f.write(format_dat('scf.Hubbard.U', 'On'))
            f.write(format_dat('scf.Hubbard.Occupation',
                               pa.get('hubbard_occupation').lower()))
        if (pa.get('smearing') is not None):
            f.write(format_dat('scf.ElectronicTemperature',
                               str(pa.get('smearing')[1] / kB)))
        f.write(format_dat('scf.energycutoff',
                           str(pa.get('energy_cutoff') / Ry)))
        f.write(format_dat('scf.maxIter', str(pa.get('scf_max_iter'))))
        f.write(format_dat('scf.EigenvalueSolver', pag('eigenvalue_solver')))
        if type(pa.get('kpts')) in [float, int]:
            kptdensity = pa.get('kpts')
            pa['kpts'] = tuple(kptdensity2monkhorstpack(
                atoms, pa.get('kpts'), False))
            print("Using a %s Monkhorst pack which is closest to a k point \
                  density of %s Ang^3" % (pa.get('kpts'), kptdensity))
            if pa.get('kpts') == (1, 1, 1):
                print("When only the gamma point is considered, \n \
                        the eigenvalue solver is changed to 'Cluster' \n \
                        with the periodic boundary condition.")
                pa['eigenvalue_solver'] = 'Cluster'
        f.write(format_dat('scf.Kgrid', pa.get('kpts')))
        f.write(format_dat('scf.Mixing.Type', pa.get('mixing_type')))
        if(pa.get('scf_init_mixing_weight') is not None):
            f.write(format_dat('scf.Init.Mixing.Weight',
                               str(pa.get('scf_init_mixing_weight'))))
        if(pa.get('scf_fixed_grid') is not None):
            f.write(format_dat('scf.fixed.grid', ' '.join(
                               [str(i) for i in pa.get('scf_fixed_grid')])))
        if(pa.get('min_mixing_weight') is not None):
            f.write(format_dat('scf.Min.Mixing.Weight',
                               str(pa.get('min_mixing_weight'))))
        if(pa.get('max_mixing_weight') is not None):
            f.write(format_dat('scf.Max.Mixing.Weight',
                               str(pa.get('max_mixing_weight'))))
        if(pa.get('mixing_history') is not None):
            f.write(format_dat('scf.Mixing.History',
                               str(pa.get('mixing_history'))))
        if(pa.get('mixing_start_pulay') is not None):
            f.write(format_dat('scf.Mixing.StartPulay',
                               str(pa.get('mixing_start_pulay'))))
        if(pa.get('mixing_every_pulay') is not None):
            f.write(format_dat('scf.Mixing.EveryPulay',
                               str(pa.get('mixing_every_pulay'))))
        if pa.get('kerker_factor') is not None:
            f.write(format_dat('scf.Kerker.factor', pa.get('kerker_factor')))
        f.write(format_dat('scf.criterion', str(pa.get('scf_criterion') / Ha)))
        if(pa.get('stress') or (pa.get('scf_stress_tensor') is not None)):
            switch = 'on'
            if(pa.get('scf_stress_tensor') is not None):
                switch = pa.get('scf_stress_tensor')
            f.write(format_dat('scf.stress.tensor', switch))
        if pa.get('scf_system_charge') is not None:
            f.write(format_dat('scf.system.charge',
                               str(pa.get('scf_system_charge'))))

    def _write_md(self, f):
        f.write(format_dat('MD.Type', self['md_type']))
        f.write(format_dat('MD.maxIter', str(self['md_maxiter'])))
        f.write(format_dat('MD.TimeStep', str(self['time_step'])))
        f.write(format_dat('MD.Opt.criterion', str(
            self['md_criterion'] / Ha * Bohr)))

    def _write_dos(self, f):
        f.write(format_dat('Dos.fileout', 'On'))
        f.write(format_dat('Dos.Erange', self['dos_erange']))
        if type(self['dos_kgrid']) in [float, int]:
            kptdensity = self['dos_kgrid']
            self['dos_kgrid'] = tuple(kptdensity2monkhorstpack(
                self.atoms, self['dos_kgrid'], False))
            print("Using a %s Monkhorst pack which is closest to a k point \
                   density of %s Ang^3" % (self['dos_kgrid'], kptdensity))
        f.write(format_dat('Dos.Kgrid', self['dos_kgrid']))

    def _write_band(self, f):
        number_of_kpaths = len(self['band_kpath'])
        f.write(format_dat('Band.dispersion', 'On'))
        f.write(format_dat('Band.Nkpath', number_of_kpaths))
        f.write('<Band.Kpath\n')
        for kpath in self['band_kpath']:
            string = str(kpath['kpts'])
            string += ' ' + str(kpath['start_point'][0])
            string += ' ' + str(kpath['start_point'][1])
            string += ' ' + str(kpath['start_point'][2])
            string += ' ' + str(kpath['end_point'][0])
            string += ' ' + str(kpath['end_point'][1])
            string += ' ' + str(kpath['end_point'][2])
            string += ' ' + kpath['path_symbols'][0]
            string += ' ' + kpath['path_symbols'][1]
            f.write(string + '\n')
        f.write('Band.Kpath>\n')

    def _write_mo(self, f):
        f.write(format_dat('MO.fileout', 'ON'))
        f.write(format_dat('num.HOMOs', self['homos']))
        f.write(format_dat('num.LUMOs', self['lumos']))
        nkpts = len(self['mo_kpts'])
        f.write(format_dat('MO.Nkpoint', nkpts))
        if type(self['mo_kpts']) in [float, int]:
            kptdensity = self['mo_kpts']
            self['mo_kpts'] = tuple(kptdensity2monkhorstpack(
                self.atoms, self['mo_kpts'], False))
            print("Using a %s Monkhorst pack which is closest to a k point \
                   density of %s Ang^3" % (self['mo_kpts'], kptdensity))
        f.write(format_dat('MO.kpoint', self['mo_kpts']))

    def _write_dat_arguments(self, f):
        """Write directly given dat-arguments.
        """
        dat_arguments = self['dat_arguments']

        # Early return
        if dat_arguments is None:
            return

        for key, value in dat_arguments.iteritems():
            if key in self.unit_dat_keywords.keys():
                value = ('%.8f ' % value, self.unit_dat_keywords[key])
                f.write(format_dat(key, value))
            elif key in self.allowed_dat_keywords:
                f.write(format_dat(key, value))
            else:
                raise ValueError("%s not in allowed keywords." % key)

    def _write_structure(self, f, atoms):
        """Translate the Atoms object to dat-format.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        unit_cell = atoms.get_cell()
        f.write('\n')

        # Write lattice vectors
        f.write(format_dat('Atoms.UnitVectors.Unit', 'Ang'))
        f.write('<Atoms.UnitVectors\n')
        for i in range(3):
            for j in range(3):
                s = str.rjust('    %.15f' % unit_cell[i, j], 16) + ' '
                f.write(s)
            f.write('\n')
        f.write('Atoms.UnitVectors>\n')
        f.write('\n')

        self._write_atomic_coordinates(f, atoms)

    def _write_atomic_coordinates(self, f, atoms):
        """Write atomic coordinates.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """
        species, specie_numbers = self.species(atoms)
        if np.any(self['initial_magnetic_moments'] is not None):
            if type(self['initial_magnetic_moments']) == \
               float or type(self['initial_magnetic_moments']) == int:
                self['initial_magnetic_moments'] = [
                    self['initial_magnetic_moments'] for atom in atoms]
            elif len(self['initial_magnetic_moments']) == len(atoms):
                pass
            elif len(atoms) != len(species) and \
                    len(self['initial_magnetic_moments']) == len(species):
                new_list = []
                for atom in atoms:
                    for i in range(len(self['initial_magnetic_moments'])):
                        if atom.symbol == species[i]['symbol']:
                            new_list.append(
                                self['initial_magnetic_moments'][i])
                if len(new_list) != len(atoms):
                    raise RuntimeError
                self['initial_magnetic_moments'] = new_list
            else:
                raise TypeError
            if np.any(self['initial_magnetic_moments_euler_angles'] is not
                      None):
                try:
                    length = len(self['initial_magnetic_moments_euler_angles'])
                except TypeError:
                    raise TypeError(
                        'Please specify at least a tuple of two angles if not \
                         a list of these tuples')
                one_euler_angle = False
                if length == 2:
                    if type(self['initial_magnetic_moments_euler_angles'][0]) \
                        in [float, int] and \
                            type(self['initial_magnetic_moments_euler_\
                                       angles'][1]) in [float, int]:
                        one_euler_angle = True
                if one_euler_angle:
                    each_euler_angle = self['initial_magnetic_moments_\
                                             euler_angles']
                    self['initial_magnetic_moments_euler_angles'] = [
                        each_euler_angle for atom in atoms]
                else:
                    for euler_angle in self['initial_magnetic_moments_\
                                             euler_angles']:
                        if len(euler_angle) != 2:
                            raise TypeError(
                                'Need to specify both theta and phi angles')
                if self['nc_spin_constraint_euler_angles'] is not None:
                    try:
                        length = len(self['nc_spin_constraint_euler_angles'])
                    except TypeError:
                        raise TypeError(
                            'Please specify at least a tuple of two angles if \
                             not a list of these tuples')
                    one_euler_angle = False
                    if length == 2:
                        if type(self['nc_spin_constraint_euler_angles'][0]) \
                            in [float, int] and type(self['nc_spin_constraint_\
                                euler_angles'][1]) in [float, int]:
                            one_euler_angle = True
                    if one_euler_angle:
                        each_euler_angle = self['nc_spin_constraint_euler_\
                            angles']
                        self['nc_spin_constraint_euler_angles'] = [
                            each_euler_angle for atom in atoms]
                    else:
                        for euler_angle in self['nc_spin_constraint_euler_\
                             angles']:
                            if len(euler_angle) != 2:
                                raise TypeError('Need to specify both theta \
                                                and phi angles')
        if self['nc_spin_constraint_atom_indices'] is None:
            self['nc_spin_constraint_atom_indices'] = len(atoms)  # range
        if self['orbital_polarization_enhancement_atom_indices'] is None:
            self['orbital_polarization_enhancement_atom_indices'] = len(atoms)
        f.write('\n')
        f.write(format_dat('Atoms.Number', len(atoms)))
        f.write(format_dat('Atoms.SpeciesAndCoordinates.Unit', 'Ang'))
        f.write('<Atoms.SpeciesAndCoordinates\n')
        atomIndex = 0
        for atom, symbol in zip(atoms, atoms.get_chemical_symbols()):
            xyz = atom.position
            charge_of_spin_up_state_guess = \
                charge_of_spin_down_state_guess = \
                float(self.read_electron_valency(symbol + '_' +
                      self.pseudo_qualifier() + '13' +
                    self['dft_data_dict'][symbol]['pseudo-potential suffix']))
            if np.any(self['initial_magnetic_moments'] is not None):
                charge_of_spin_up_state_guess += \
                    self['initial_magnetic_moments'][atomIndex]
                charge_of_spin_down_state_guess -= \
                    self['initial_magnetic_moments'][atomIndex]
            charge_of_spin_up_state_guess /= 2.
            charge_of_spin_down_state_guess /= 2.
            line = str(atomIndex + 1) + ' '
            line += symbol + ' '
            line += str.rjust('    %.9f' % xyz[0], 16) + ' '
            line += str.rjust('    %.9f' % xyz[1], 16) + ' '
            line += str.rjust('    %.9f' % xyz[2], 16) + ' '
            line += str(charge_of_spin_up_state_guess) + ' '
            line += str(charge_of_spin_down_state_guess)
            if np.any(self['initial_magnetic_moments_euler_angles']
                      is not None):
                line += ' ' + str(float(self['initial_magnetic_moments\
                                      _euler_angles'][atomIndex][0])) + ' ' + \
                        str(float(
                            self['initial_magnetic_moments_euler_angles']
                                [atomIndex][1]))
                if self['nc_spin_constraint_euler_angles']:
                    line += ' ' + str(float(
                         self['nc_spin_constraint_euler_angles']
                         [atomIndex][0])) + ' ' + str(float(
                             self['nc_spin_constraint_euler_angles']
                             [atomIndex][1]))
                else:
                    line += ' 0.0 0.0'
                if (self['nc_spin_constraint_penalty'] or
                    self['magnetic_field']) \
                        and (atomIndex in
                             range(self['nc_spin_constraint_atom_indices'])):
                    line += ' ' + '1'
                else:
                    line += ' ' + '0'
            opea = self['orbital_polarization_enhancement_atom_indices']
            if(type(opea) == list):
                opea = opea
            elif(type(opea) == int):
                opea = range(opea)
            if np.any(self['initial_magnetic_moments'] is not None):
                if atomIndex in opea:
                    line += ' on'
                else:
                    line += ' off'
            line += '\n'
            f.write(line)
            atomIndex += 1
        f.write('Atoms.SpeciesAndCoordinates>\n')
        f.write('\n')

    def _write_species(self, f, atoms):
        """Write input related the different species.

        Parameters:
            - f:     An open file object.
            - atoms: An atoms object.
        """

        species, species_numbers = self.species(atoms)

        nspecies = len(species)
        if self['wannier_initial_projectors']:
            nspecies *= 2
        f.write(format_dat('Species.Number', nspecies))
        f.write('<Definition.of.Atomic.Species\n')
        orbital_letters = ['s', 'p', 'd', 'f', 'g', 'h']
        j = 0
        for specie in species:
            line = specie['symbol']
            pseudo_potential_suffix = self['dft_data_dict'
                                           ][specie['symbol']
                                             ]['pseudo-potential suffix']
            line += ' ' + specie['symbol']
            cutoff_radius = self['dft_data_dict'][specie['symbol']
                                                  ]['cutoff radius']
            cutoff_radius = self.find_closest_cutoff(
                specie['symbol'], cutoff_radius)
            line += str(float(cutoff_radius / Bohr))
            line += pseudo_potential_suffix + '-'
            orbital_numbers = self['dft_data_dict'][specie['symbol']
                                                    ]['orbitals used']
            for i in range(len(orbital_numbers)):
                line += orbital_letters[i] + str(orbital_numbers[i])
            line += ' ' + specie['symbol'] + '_' + \
                self.pseudo_qualifier() + '13' + pseudo_potential_suffix
            f.write(line + '\n')
            if self['wannier_initial_projectors']:
                j += 1
                line = 'proj' + str(j)
                pseudo_potential_suffix = self['dft_data_dict'
                                               ][specie['symbol']
                                                 ]['pseudo-potential suffix']
                line += ' ' + specie['symbol']
                cutoff_radius = self['dft_data_dict'][specie['symbol']
                                                      ]['cutoff radius']
                cutoff_radius = self.find_closest_cutoff(
                    specie['symbol'], cutoff_radius)
                line += str(float(cutoff_radius / Bohr))
                line += pseudo_potential_suffix + '-s1p1d1f1'
                line += ' ' + specie['symbol'] + '_' + \
                    self.pseudo_qualifier() + '13' + pseudo_potential_suffix
                f.write(line + '\n')
        f.write('Definition.of.Atomic.Species>\n')
        if self['wannier_initial_projectors']:
            f.write('Wannier.Initial.Projectors.Unit Ang\n')
            f.write('<Wannier.Initial.Projectors\n')
            symbols = [specie['symbol'] for specie in species]
            for i in range(len(self.atoms)):
                for orbital_type, direction in \
                        self['wannier_initial_projectors'][i]:
                    line = 'proj' + \
                        str(symbols.index(self.atoms[i].symbol) + 1)
                    line += '-' + orbital_type + ' '
                    x, y, z = self.atoms[i].position
                    line += '%f %f %f ' % (x, y, z)
                    zaxis = tuple(direction['z'])
                    line += '%f %f %f ' % zaxis
                    xaxis = tuple(direction['x'])
                    line += '%f %f %f \n' % xaxis
                    f.write(line)
            f.write('Wannier.Initial.Projectors>\n')
        if self['hubbard_u_values'] is not None:
            f.write('<Hubbard.U.values\n')
            hubbard_dict = {}
            for specie in species:
                orbital_code = self['dft_data_dict'][specie['symbol']
                                                     ]['orbitals used']
                orbital_letter_number_list = [
                    (orbital_letters[i], orbital_code[i])
                    for i in range(len(orbital_code))]
                hubbard_dict[specie['symbol']] = {}
                for orbital_letter_number in orbital_letter_number_list:
                    for i in range(orbital_letter_number[1]):
                        hubbard_dict[specie['symbol']][str(
                            i + 1) + orbital_letter_number[0]] = 0.
            if self['hubbard_u_values'] is not None:
                for symbol in self['hubbard_u_values'].keys():
                    for orbital in self['hubbard_u_values'][symbol].keys():
                        hubbard_dict[symbol][orbital] = \
                             self['hubbard_u_values'][symbol][orbital]
            for symbol in hubbard_dict.keys():
                line = symbol
                for orbital in hubbard_dict[symbol].keys():
                    line += ' ' + orbital + ' ' + \
                        str(hubbard_dict[symbol][orbital])
                line += '\n'
                f.write(line)
            f.write('Hubbard.U.values>\n')

    def pseudo_qualifier(self):
        """Get the extra string used in the middle of the pseudopotential.
        The retrieved pseudopotential for a specific element will be
        'Hxxx.vps' for the element 'H' with qualifier 'xxx'. If qualifier
        is set to None then the qualifier is set to "[functional author]".
        """
        if self['pseudo_qualifier'] is None:
            if self['xc'] == 'LDA':
                return 'CA'
            else:
                return 'PBE'
        else:
            return self['pseudo_qualifier']

    @staticmethod
    def read_electron_valency(filename='H_CA13'):
        array = []
        with open(join(os.environ['OPENMX_DFT_DATA_PATH'],
                       'VPS/' + filename + '.vps'), 'r') as f:
            array = f.readlines()
            f.close()
        required_line = ''
        for line in array:
            if 'valence.electron' in line:
                required_line = line
        return read_nth_to_last_value(required_line)

    def read_input(self):
        if(self.debug):
            print('Reading input files')
        filename = self.get_file_name('.dat')
        if not self.nohup:
            with open(filename, 'r') as f:
                while True:
                    line = f.readline()
                    print('%s' % line.strip())
                    if not line:
                        break

    def read_restart_input(self, filename=None):
        """ Read the .dat# file. Purpose of this function is to make user
            able to run molecular dynamics from python level
        """
        if filename is None:
            filename = self.get_file_name('.dat#')
        if(self.debug):
            print('Reading %s' % filename)
        line = '\n'
        rn = read_nth_to_last_value
        fl = float
        with open(filename, 'r') as f:
            while line != '':
                line = f.readline().lower()
                if 'scf.fixed.grid' in line:
                    self['scf_fixed_grid'] = (fl(rn(line, 3)), fl(rn(line, 2)),
                                              fl(rn(line, 1)))
                if 'md.current.iter' in line:
                    self['md_current_iter'] = int(rn(line))

    def read_log_file(self, filename=None):
        """READ .log file. This method is unnatural since all the results
           should be contained in .out file. However, in the version 3.8.3,
           stress tensor results are contained only in the .log file.
           This is reason why I made this method.
           get Hartree / bohr^3 and return eV / Aug^3
        """
        if filename is None:
            filename = self.get_file_name('.log')
        rn = read_nth_to_last_value
        fl = float
        line = '\n'
        with open(filename, 'r') as f:
            if(self.debug):
                print('Reading %s' % filename)
            while line != '':
                line = f.readline().lower()
                if 'stress tensor' in line:
                    if(self.debug):
                        print('Reading stress tensor..')
                    f.readline()
                    f.readline()
                    nl = f.readline()
                    xx, xy, xz = (fl(rn(nl, 3)), fl(rn(nl, 2)), fl(rn(nl, 1)))
                    nl = f.readline()
                    yx, yy, yz = (fl(rn(nl, 3)), fl(rn(nl, 2)), fl(rn(nl, 1)))
                    nl = f.readline()
                    zx, zy, zz = (fl(rn(nl, 3)), fl(rn(nl, 2)), fl(rn(nl, 1)))
                    break
            try:
                self.results['stress'] = np.array([xx, yy, zz, (zy + yz)/2,
                                                               (zx + xz)/2,
                                                               (yx + xy)/2])
                self.results['stress'] *= Ha / Bohr ** 3
                if(self.debug):
                    print('Finished reading stress%s' % filename)
            except UnboundLocalError:
                if(self.debug):
                    print('No stress info found')

    def read_xc(self, line):
        xc = read_nth_to_last_value(line).upper()
        if xc == 'LDA':
            return 'LDA'
        elif xc == 'GGA-PBE':
            return 'PBE'
        elif xc == 'LSDA-CA':
            return 'CA'
        elif xc == 'LSDA-PW':
            return 'PW'

    def read(self, label):
        print('Directly read prameters from Given label')
        self.set_label(label)
        if label[-5:] in ['.dat', '.out', '.log']:
            label = label[:-4]
        self.parameters = {}
        self.parameters['restart'] = self.label
        self.parameters['label'] = label
        self.read_results()

    def read_results(self, filename=None, properties=implemented_properties):
        """Read the results.
        """
        pa = self.parameters
        ln = '\n'  # line
        rn = read_nth_to_last_value
        if filename is None:
            filename = self.get_file_name('.out')
        if(self.debug):
            print('Read results from %s' % filename)
        with open(filename, 'r') as f:
            number_of_atoms = 0
            while ln != '':
                ln = f.readline().lower()
                if 'scf.restart' in ln:
                    pa['restart'] = rn(ln, 1)
                if 'scf.kgrid' in ln:
                    pa['kpts'] = (int(rn(ln, 3)), int(rn(ln, 2)),
                                  int(rn(ln, 1)))
                if 'scf.xctype' in ln:
                    pa['xc'] = self.read_xc(ln)
                if 'scf.eigenvaluesolver' in ln:
                    pa['eigenvalue_solver'] = rn(ln)
                if 'scf.energycutoff' in ln:
                    pa['energy_cutoff'] = float(rn(ln)) * Ry
                if 'scf.electronictemperature' in ln:
                    pa.smearing = ('Fermi-Dirac', float(rn(ln)) * kB)
                if 'scf.spinpolarization' in ln:
                    pa.scf_spinpolarization = rn(ln)
                if 'scf.stress.tensor' in ln:
                    pa.scf_stress_tensor = rn(ln)
                if 'wannier.kgrid' in ln:
                    pa.wannier_kpts = (int(rn(ln, 3)), int(rn(ln, 2)),
                                       int(rn(ln, 1)))
                if 'dipole' and 'dipole moment' in ln:
                    while 'total' not in ln:
                        ln = f.readline().lower()
                    dipole = np.ndarray(3)
                    dipole[0] = float(rn(ln, 3))
                    dipole[1] = float(rn(ln, 2))
                    dipole[2] = float(rn(ln))
                    self.results['dipole'] = dipole
                    self.dipole_history.append(self.results['dipole'])
                    continue
                if 'fractional coordinates of the final structure' in ln:
                    while '1' not in ln:
                        ln = f.readline().lower()
                    scaled_positions = np.ndarray((number_of_atoms, 3), float)
                    for i in range(number_of_atoms):
                        scaled_positions[i][0] = float(rn(ln, 3))
                        scaled_positions[i][1] = float(rn(ln, 2))
                        scaled_positions[i][2] = float(rn(ln))
                        ln = f.readline().lower()
                if 'total spin moment' in ln:
                    # need to incorporate NC spin
                    if 'magmom' in properties:
                        self.results['magmom'] = float(rn(ln))
                        pa['total_magnetic_moments'] = self.results['magmom']
                    if 'magmoms' in properties:
                        ln = f.readline()
                        while ln == '\n':
                            ln = f.readline()
                        ln = f.readline()
                        self.results['magmoms'] = np.ndarray(
                                                  number_of_atoms, float)
                        for atom_index in range(number_of_atoms):
                            self.results['magmoms'][atom_index] = rn(ln)
                            ln = f.readline()
                if 'chemical_potential' in properties \
                        and 'chemical potential' in ln:
                    self.results['chemical_potential'] = float(rn(ln)) * Ha
        self.read_atomic_info(filename)
        self.read_restart_input()
        self.read_log_file()
        self.read_eigenvalue_output()

    def read_atomic_info(self, filename):
        abs_dir = join(os.getcwd(), self.directory)
        abs_lab = join(abs_dir, self.prefix)
        ''' Read the atomic cell size, positions, magnetic moments and force'''
        line = '\n'
        cell = np.ndarray((3, 3), float)
        symbols = []
        forces = []
        magmoms = []
        positions = []
        rn = read_nth_to_last_value
        if filename is None:
            filename = join(abs_lab, '.out')
        with open(filename, 'r') as f:
            while line != '':
                line = f.readline().lower()
                if 'atoms.number' in line:
                    number_of_atoms = int(rn(line))
                if 'atoms.unitvectors.unit' in line:
                    unit = rn(line)
                    if unit == 'ang':
                        unit = 1
                    elif unit == 'au':
                        unit = Bohr
                    line = f.readline()
                    for i in range(3):
                        line = f.readline()
                        vector = []
                        for j in range(3):
                            vector.append(rn(line, 3 - j))
                        cell[i] = vector
                    cell *= unit
                    self.results['unit_cell'] = cell
                if '<atoms.speciesandcoordinates' in line:
                    for i in range(number_of_atoms):
                        line = f.readline()
                        ml = line.split(' ')
                        n = len(ml) - ml.count('')
                        symbols.append(rn(line, n - 1))
                        pos = list(filter(None, ml))
                        # print(pos)
                        positions.append(pos[2:5])
                        if(pos[6] is not None):
                            magmoms.append(float(pos[6])-float(pos[5]))
                if 'utot.' in line:
                    self.results['energy'] = float(rn(line)) * Ha
                    self.results['free_energy'] = self.results['energy']
                    self.energy_history.append(self.results['energy'])
                    continue
                if '<coordinates.forces' in line:
                    line = f.readline().lower()
                    forces = np.ndarray((number_of_atoms, 3), float)
                    for i in range(number_of_atoms):
                        line = f.readline().lower()
                        forces[i][0] = float(read_nth_to_last_value(line, 3))
                        forces[i][1] = float(read_nth_to_last_value(line, 2))
                        forces[i][2] = float(read_nth_to_last_value(line))
                    self.results['forces'] = forces * Ha / Bohr
                    self.forces_history.append(self.results['forces'])
                    continue
                self.atoms = Atoms(symbols, cell=cell, positions=positions,
                                   pbc=np.array([True, True, True],
                                                dtype=bool),
                                   magmoms=magmoms)

    def update_atoms(self, atoms):
        self.atoms_history.append(atoms.copy())
        self.atoms = atoms.copy()

    def get_file_name(self, extension='.out', directory=None,
                      prefix=None, label=None):
        # Get the filename. variabel 'label' is superceed others
        if label is not None:
            self.set_label(label)
            directory = None
            prefix = None
        if prefix is None:
            prefix = self.prefix
        if directory is None:
            directory = self.directory
        abs_dir = join(os.getcwd(), directory)
        abs_lab = join(abs_dir, prefix)
        return abs_lab + extension

    def get_xc(self, xc):
        if xc in ['PBE', 'GGA', 'GGA-PBE']:
            return 'GGA-PBE'
        elif xc in ['LDA']:
            return 'LDA'
        elif xc in ['CA', 'PW']:
            return 'LSDA-' + xc
        elif xc in ['LSDA']:
            return 'LSDA-CA'
        else:
            print('Set XcType as LDA')
            return 'LDA'

    def get_command(self, processes, threads, runfile=None, outfile=None):
        # Contruct the command to send to the operating system
        abs_dir = os.getcwd()
        command = ''
        # run processes specified by the system variable OPENMX_COMMAND
        if processes is None:
            command += os.environ.get('OPENMX_COMMAND')
            if command is None:
                print('Either specify OPENMX_COMMAND as an environment\
                variable or specify processes as a keyword argument')
        else:  # run with a specified number of processes
            threads_string = ' -nt ' + str(threads)
            if threads is None:
                threads_string = ''
            command += 'mpirun -np ' + \
                str(processes) + ' openmx %s' + threads_string + ' > %s'
        if runfile is None:
            runfile = abs_dir + '/' + self.prefix + '.dat'
        if outfile is None:
            outfile = abs_dir + '/' + self.prefix + '.log'
        try:
            command = command % (runfile, outfile)
            # command += '" > ./%s &' % outfile  # outputs
        except TypeError:  # in case the OPENMX_COMMAND is incompatible
            raise ValueError(
                "The 'OPENMX_COMMAND' environment must " +
                "be a format string" +
                " with four string arguments.\n" +
                "Example : 'mpirun -np 4 openmx ./%s -nt 2 > ./%s'.\n" +
                "Got '%s'" % command)
        return command

    def get_standard_unit(self, key, value):
        raise NotImplementedError(
            "Changing %s Unit %s to ASE standard is not currently supported.\n \
            \Please change it into standard yourself!" % (key, value))

    def get_dos(self, atoms=None, erange=None, **kwargs):
        if atoms is None:
            atoms = self.atoms
        if not self['dos_erange']:
            print('Turning DOS file output on')
            if erange:
                self['dos_erange'] = erange
            else:
                self['dos_erange'] = \
                     (float(input('Please specify a lower energy bound in eV, '
                                  'with respect to the Fermi level, for the '
                                  'DOS calculation: ')),
                      float(input('And an upper energy bound please: ')))
            self.dos = DOS(self)
            if atoms:
                print('Calculating Eigenvalues and Eigenvectors')
                self.calculate(atoms)
                print('Calculating DOS')
                return self.get_dos(atoms=atoms, erange=erange, **kwargs)
            else:
                print('Please supply an atoms argument')
                return 0
        else:
            if erange is None:
                erange = self['dos_erange']
            # try:
            #    f1 = open(os.path.join(self.directory,
            #                           self.prefix + '.Dos.val'), 'r')
            #    f2 = open(os.path.join(self.directory,
            #                           self.prefix + '.Dos.vec'), 'r')
            # except FileNotFoundError:
            #    print('Calculating Eigenvalues and Eigenvectors')
            #    self.calculate(atoms)
            #    print('Calculating DOS')
            #    return self.get_dos(atoms=atoms, erange=erange, **kwargs)
        return self.dos.get_dos(atoms=atoms, erange=erange, **kwargs)

    def get_band(self, erange=(-5, 5), plot='pyplot', gnuband=True, atoms=None,
                 spin=None, fermi_level=True, file_format=False,
                 fileName=None):
        if not self['band_dispersion']:
            raise RuntimeError(
                'Please specify calculator parameters for band structure')
        else:
            try:
                f = open(self.directory + '/' + self.prefix + '.Band', 'r')
                f.close()
            except ReadError:
                print('There is no ".BAND" file in ' +
                      self.directory + '/' + self.prefix + '.Band')
                print('Calculating band structure')
                if atoms is None:
                    atoms = self.atoms
                self.calculate(atoms)
                return self.get_band(erange=erange, plot=plot, gnuband=gnuband,
                                     atoms=atoms, spin=spin, fileName=fileName,
                                     fermi_level=fermi_level,
                                     file_format=file_format, )
        if gnuband:
            input_files = self.prefix + '.Band'
            executable_name = 'bandgnu13'
            input_command(self, executable_name, input_files)
            if plot == 'gnuplot':
                input_files = self.prefix + '.GNUBAND'
                lines = []
                with open(os.path.join(self.directory, input_files), 'r') as f:
                    lines = f.readlines()
                    for i in range(len(lines)):
                        if 'yra' in lines[i]:
                            lines[i] = 'set yra [' + \
                                str(erange[0]) + ':' + str(erange[1]) + ']\n'
                        if spin is 'up':
                            if 'plot' in lines[i]:
                                lines[i] = 'plot "' + \
                                    self.prefix + '.BANDDAT1"\n'
                        if spin is 'down':
                            if 'plot' in lines[i]:
                                lines[i] = 'plot "' + \
                                    self.prefix + '.BANDDAT2"\n'
                with open(os.path.join(self.directory, input_files), 'w') as f:
                    f.writelines(lines)
                executable_name = 'gnuplot'
                input_command(self, executable_name, input_files)
            elif plot == 'pyplot':
                self.band = Band(self)
                return self.band.get_band(erange=erange, fileName=fileName,
                                          fermi_level=fermi_level,
                                          file_format=file_format, )
            elif plot is None:
                if(self.debug):
                    print('Finished band structure calculation')
            else:
                print(plot + ' not implemented')
                NotImplementedError

    def get_mo(self, homos=None, lumos=None, real=True, imaginary=False,
               spins=None, atoms=None):
        if not self['lumos'] and not self['homos']:
            if atoms:
                if not homos and not lumos:
                    print('Please specify the numbers of HOMOs and LUMOs')
                elif not homos:
                    print('Please specify the numbers of HOMOs')
                elif not lumos:
                    print('Please specify the numbers of LUMOs')
                if not homos:
                    homos = input('HOMOs: ')
                self['homos'] = homos
                if not lumos:
                    lumos = input('LUMOs: ')
                self['lumos'] = lumos
                print('Calculating molecular orbitals')
                self.calculate(atoms)
                return self.get_mo(homos, lumos, real, imaginary, spins, atoms)
            else:
                print('Please supply an atoms argument')
                return 0
        spin_index = []
        if np.all(self['initial_magnetic_moments'] is None):
            if spins is not None:
                print('System is not spin polarised.')
                print('Showing orbitals including combining both spins')
            spins = [0]
        elif spins is None:
            spins = [0, 1]
        else:
            if 'up' in spins:
                spin_index.append(0)
            if 'down' in spins:
                spin_index.append(1)
            spins = spin_index
        if homos is None:
            homos = list(self['homos'])  # list -> range
        if lumos is None:
            lumos = list(self['lumos'])  # list -> range
        input_files = []
        for kpt in range(len(self['mo_kpts'])):
            for spin in spins:
                for homo in homos:
                    if real:
                        input_files.append(
                            self.label + '.homo' + str(kpt) + '_' + str(spin) +
                                         '_' + str(homo) + '_r.cube')
                    if imaginary:
                        input_files.append(
                            self.label + '.homo' + str(kpt) + '_' + str(spin) +
                                         '_' + str(homo) + '_i.cube')
                for lumo in lumos:
                    if real:
                        input_files.append(
                            self.label + '.lumo' + str(kpt) + '_' + str(spin) +
                                         '_' + str(lumo) + '_r.cube')
                    if imaginary:
                        input_files.append(
                            self.label + '.lumo' + str(kpt) + '_' + str(spin) +
                                         '_' + str(lumo) + '_i.cube')
        if self['absolute_path_of_vesta'] is None:
            self['absolute_path_of_vesta'] = \
                input("Please specify the absolute file path for this system's"
                      " VESTA executable: ")
        input_command(self, self['absolute_path_of_vesta'], input_files,
                      ' '.join(['%s' for i in range(len(input_files))]))

    def find_cutoff_radii_for_atom(self, symbol):
        pao_path = os.path.join(self['dft_data_path'], 'PAO')
        filenames = [a for a in os.walk(pao_path)][0][2]
        cutoff_radii = []
        for filename in filenames:
            if filename[:len(symbol)] == symbol and filename[len(symbol)] not \
                in ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                    'h', 'i', 'j', 'k', 'l', 'm', 'n',
                    'o', 'p', 'q', 'r', 's', 't', 'u',
                         'v', 'w', 'x', 'y', 'z'] and 'Si' not in filename[1:]:
                parts = filename.split('.')
                parts.remove(parts[-1])
                filename = '.'.join(parts).split(symbol)[1]
                number_string_list = []
                i = 0
                while filename[i] in ['0', '1', '2', '3', '4',
                                      '5', '6', '7', '8', '9', '.']:
                    number_string_list.append(filename[i])
                    i += 1
                    if i == len(filename):
                        break
                cutoff_radii.append(float(''.join(number_string_list)) * Bohr)
        return cutoff_radii

    def find_closest_cutoff(self, symbol, cutoff):
        available_cutoffs = self.find_cutoff_radii_for_atom(symbol)
        if cutoff in available_cutoffs:
            return cutoff
        else:
            difference_squared = map(lambda a: (
                a - cutoff) * (a - cutoff), available_cutoffs)
            available_cutoff = available_cutoffs[difference_squared.index[min(
                difference_squared)]]
            print('%d Ang cutoff radius for PAO of %s not available. Using %d '
                  'Ang instead.' % (cutoff, symbol, available_cutoff))
            return available_cutoff

    def get_bz_k_points(self):
        kgrid = self['kpts']
        if type(kgrid) in [int, float]:
            kgrid = kptdensity2monkhorstpack(self.atoms, kgrid, False)
        bz_k_points = []
        n1 = kgrid[0]
        n2 = kgrid[1]
        n3 = kgrid[2]
        for i in range(n1):
            for j in range(n2):
                # Monkhorst Pack Grid [H.J. Monkhorst and J.D. Pack,
                # Phys. Rev. B 13, 5188 (1976)]
                for k in range(n3):
                    bz_k_points.append((0.5 * float(2 * i - n1 + 1) / n1,
                                        0.5 * float(2 * j - n2 + 1) / n2,
                                        0.5 * float(2 * k - n3 + 1) / n3))
        return np.array(bz_k_points)

    # for now, irreducible Brillouin zone will be the first Brillouin zone
    def get_ibz_k_points(self):
        return self.get_bz_k_points()

    def get_lattice_type(self):
        cellpar = cell_to_cellpar(self.atoms.cell)
        abc = cellpar[:3]
        angles = cellpar[3:]
        min_lv = min(abc)
        if abc.ptp() < 0.01 * min_lv:
            if abs(angles - 90).max() < 1:
                return 'cubic'
            elif abs(angles - 60).max() < 1:
                return 'fcc'
            elif abs(angles - np.arccos(-1 / 3.) * 180 / np.pi).max < 1:
                return 'bcc'
        elif abs(angles - 90).max() < 1:
            if abs(abc[0] - abc[1]).min() < 0.01 * min_lv:
                return 'tetragonal'
            else:
                return 'orthorhombic'
        elif abs(abc[0] - abc[1]) < 0.01 * min_lv and \
                abs(angles[2] - 120) < 1 and abs(angles[:2] - 90).max() < 1:
            return 'hexagonal'
        else:
            return 'not special'

    def get_number_of_spins(self):
        try:
            magmoms = self.atoms.get_initial_magnetic_moments()
            if self['scf_spinpolarization'] is None:
                if isinstance(magmoms[0], float):
                    if abs(magmoms).max() < 0.1:
                        return 1
                    else:
                        return 2
                else:
                    raise NotImplementedError
            else:
                if self['scf_spinpolarization'] == 'on':
                    return 2
                elif self['scf_spinpolarization'] == 'nc':
                    return 1
        except KeyError:
            return 1

    def get_eigenvalues(self, kpt=None, spin=None):
        if 'eigenvalues' not in self.results:
            # print('Turning DOS file output on')
            # self['dos_erange']=(-5., 5.)
            self.calculate(self.atoms)
        if kpt is None and spin is None:
            return self.results['eigenvalues']
        else:
            return self.results['eigenvalues'][spin, kpt, :]

    def read_eigenvalue_output(self, filename=None):
        abs_dir = join(os.getcwd(), self.directory)
        abs_lab = join(abs_dir, self.prefix)
        rn = read_nth_to_last_value
        try:
            with open(os.path.join(abs_lab, '.Dos.val'), 'r') as f:
                string = ''
                while 'irange' not in string:
                    string = f.readline()
                number_of_bands = int(read_nth_to_last_value(string, 1))
                self.results['number_of_bands'] = number_of_bands
                while 'Kgrid' not in string:
                    string = f.readline()
                    number_of_k_points = \
                        int(read_nth_to_last_value(string, 1)) * \
                        int(read_nth_to_last_value(string, 2)) * \
                        int(read_nth_to_last_value(string, 3))
                while 'Eigenvalues' not in string:
                    string = f.readline()
                string = f.readline()
                number_of_spins = self.get_number_of_spins()
                eigenvalues = \
                    np.ndarray((number_of_k_points,
                                number_of_spins,
                                number_of_bands), float)
                for i in range(number_of_k_points):
                    for j in range(number_of_spins):
                        for k in range(number_of_bands):
                            eigenvalues[i, j, k] = float(
                                read_nth_to_last_value(
                                    string, number_of_bands - k)) * Ha
                        string = f.readline()
                self.results['eigenvalues'] = eigenvalues
        except ReadError:
            if(self.debug):
                print("No .Dos.val file found, trying .eigen file instead")
        try:
            with open(os.path.join(self.directory,
                                   self.prefix + '.eigen'), 'r') as f:
                string = f.readline()
                self.results['chemical_potential'] = float(
                    read_nth_to_last_value(string, 1)) * Ha
                string = f.readline()
                nbands = int(read_nth_to_last_value(string, 1))
                nspin = 1  # still need to implement spin polarization method
                eigenvalues = []
                string = f.readline()
                while 'WF' not in string:
                    eigenvalues.append(np.ndarray((nspin, nbands), float))
                    for n in range(nbands):
                        eigenvalues[-1][0, n] = \
                            float(read_nth_to_last_value(string)) * Ha
                        string = f.readline()
            self.results['eigenvalues'] = np.array(eigenvalues)
        except ReadError:
            if(self.debug):
                print("No .eigen file found")
        try:
            if(self.debug):
                print("Trying to read .out File")
            fl = float
            eig = []  # eigenvalues
            nkpts = len(self.get_ibz_k_points())
            nbands = self.get_number_of_bands()
            nspins = self.get_number_of_spins()
            eig = np.zeros((nspins, nkpts, nbands))
            if(self.debug):
                print('Spin, KPTS, bands', nspins, nkpts, nbands)
            k = 0
            if filename is None:
                filename = self.get_file_name('.out')
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
                            if(self.debug):
                                print('nbands != orbital number %d' % i)
                            k += 1
                            continue
                        k += 1
                    if nkpts == k or not line:
                        break
            self.results['eigenvalues'] = eig
        except ReadError:
            if(self.debug):
                print("No .out file found")

    def get_fermi_level(self):
        try:
            fermi_level = self.results['chemical_potential']
        except KeyError:
            self.calculate()
            fermi_level = self.results['chemical_potential']
        return fermi_level

    def get_number_of_bands(self):
        pag = self.parameters.get
        dfd = default_dictionary
        if 'number_of_bands' not in self.results:
            n = 0
            for atom in self.atoms:
                sym = atom.symbol
                orbitals = pag('dft_data_dict', dfd)[sym]['orbitals used']
                d = 1
                for orbital in orbitals:
                    n += d * orbital
                    d += 2
            self.results['number_of_bands'] = n
        return self.results['number_of_bands']

    def dirG(self, dk, bzone=(0, 0, 0)):
        nx, ny, nz = self['wannier_kpts']
        dx = dk // (ny * nz) + bzone[0] * nx
        dy = (dk // nz) % ny + bzone[1] * ny
        dz = dk % nz + bzone[2] * nz
        return dx, dy, dz

    def dk(self, dirG):
        dx, dy, dz = dirG
        nx, ny, nz = self['wannier_kpts']
        return ny * nz * (dx % nx) + nz * (dy % ny) + dz % nz

    def read_bloch_overlaps(self):
        try:
            with open(os.path.join(self.directory,
                                   self.prefix + '.mmn'), 'r') as f:
                string = f.readline()
                string = f.readline()
                # d_num = int(read_nth_to_last_value(string, 2))
                kpt_num = int(read_nth_to_last_value(string, 3))
                band_num = int(read_nth_to_last_value(string, 4))
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

    def get_wannier_localization_matrix(self, nbands, dirG, nextkpoint=None,
                                        kpoint=None, spin=0, G_I=(0, 0, 0)):
        # only expected to work for no spin polarization
        try:
            self['bloch_overlaps']
        except KeyError:
            self.read_bloch_overlaps()
        dirG = tuple(dirG)
        nx, ny, nz = self['wannier_kpts']
        nr3 = nx * ny * nz
        if kpoint is None and nextkpoint is None:
            return {kpoint: self['bloch_overlaps'
                                 ][kpoint][dirG][:nbands, :nbands
                                                 ] for kpoint in range(nr3)}
        if kpoint is None:
            kpoint = (nextkpoint - self.dk(dirG)) % nr3
        if nextkpoint is None:
            nextkpoint = (kpoint + self.dk(dirG)) % nr3
        if dirG not in self['bloch_overlaps'][kpoint].keys():
            return np.zeros((nbands, nbands), complex)
        return self['bloch_overlaps'][kpoint][dirG][:nbands, :nbands]

    def initial_wannier(self, mode, kptgrid=None, fixedstates_k=None,
                        edf_k=None, spin=0, nbands=None):
        try:
            self['initial_wannier_projections']
        except KeyError:
            self.read_initial_wannier_projections()
        nk = len(fixedstates_k)
        nwannier = fixedstates_k[0] + edf_k[0]
        U_kww = np.zeros((nk, nwannier, nwannier), complex)
        C_kul = []
        for k in range(nk):
            U_kww[k] = self['initial_wannier_projections'
                            ][k][:nwannier, :nwannier]
            C_kul.append(np.identity(
                nbands - fixedstates_k[k], complex)[:, :edf_k[k]])
        return C_kul, U_kww

    def read_initial_wannier_projections(self):
        try:
            with open(os.path.join(self.directory,
                                   self.prefix + '.amn'), 'r') as f:
                string = f.readline()
                string = f.readline()
                nbands = int(read_nth_to_last_value(string, 4))
                kpt_num = int(read_nth_to_last_value(string, 3))
                nwannier = int(read_nth_to_last_value(string, 2))
                self['initial_wannier_projections'] = [np.ndarray(
                    (nbands, nwannier), complex) for i in range(kpt_num)]
                for k in range(kpt_num):
                    for n in range(nwannier):
                        for m in range(nbands):
                            string = f.readline()
                            self['initial_wannier_projections'][k][m, n] = \
                                complex(
                                float(read_nth_to_last_value(string, 2)),
                                float(read_nth_to_last_value(string, 1)))
        except self.ReadError:
            raise Exception('Please calculate the overlap matrix elements for '
                            'the bloch states')
