""" This module defines a FileIOCalculator for DFTB+

http://www.dftbplus.org/
http://www.dftb.org/

Initial development: markus.kaukonen@iki.fi
"""

import os
from pathlib import Path
import numpy as np
from ase.io import read
from ase.calculators.calculator import (FileIOCalculator, kpts2ndarray,
                                        kpts2sizeandoffsets)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.units import Hartree, Bohr, fs
from ase.constraints import FixAtoms


class Dftb(FileIOCalculator):
    if 'DFTB_COMMAND' in os.environ:
        command = os.environ['DFTB_COMMAND'] + ' > PREFIX.out'
    else:
        command = 'dftb+ > PREFIX.out'

    implemented_properties = ['energy', 'forces', 'charges',
                              'stress', 'dipole']
    discard_results_on_any_change = True

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='dftb', atoms=None, kpts=None,
                 slako_dir=None,
                 **kwargs):
        """
        All keywords for the dftb_in.hsd input file (see the DFTB+ manual)
        can be set by ASE. Consider the following input file block:

        >>> Hamiltonian = DFTB {
        >>>     SCC = Yes
        >>>     SCCTolerance = 1e-8
        >>>     MaxAngularMomentum = {
        >>>         H = s
        >>>         O = p
        >>>     }
        >>> }

        This can be generated by the DFTB+ calculator by using the
        following settings:

        >>> calc = Dftb(Hamiltonian_='DFTB',  # line is included by default
        >>>             Hamiltonian_SCC='Yes',
        >>>             Hamiltonian_SCCTolerance=1e-8,
        >>>             Hamiltonian_MaxAngularMomentum_='',
        >>>             Hamiltonian_MaxAngularMomentum_H='s',
        >>>             Hamiltonian_MaxAngularMomentum_O='p')

        In addition to keywords specific to DFTB+, also the following keywords
        arguments can be used:

        restart: str
            Prefix for restart file.  May contain a directory.
            Default is None: don't restart.
        ignore_bad_restart_file: bool
            Ignore broken or missing restart file. By default, it is an
            error if the restart file is missing or broken.
        label: str (default 'dftb')
            Prefix used for the main output file (<label>.out).
        atoms: Atoms object (default None)
            Optional Atoms object to which the calculator will be
            attached. When restarting, atoms will get its positions and
            unit-cell updated from file.
        kpts: (default None)
            Brillouin zone sampling:

            * ``(1,1,1)`` or ``None``: Gamma-point only
            * ``(n1,n2,n3)``: Monkhorst-Pack grid
            * ``dict``: Interpreted as a path in the Brillouin zone if
              it contains the 'path_' keyword. Otherwise it is converted
              into a Monkhorst-Pack grid using
              ``ase.calculators.calculator.kpts2sizeandoffsets``
            * ``[(k11,k12,k13),(k21,k22,k23),...]``: Explicit (Nkpts x 3)
              array of k-points in units of the reciprocal lattice vectors
              (each with equal weight)

        Additional attribute to be set by the embed() method:

        pcpot: PointCharge object
            An external point charge potential (for QM/MM calculations)
        """

        if slako_dir is None:
            slako_dir = os.environ.get('DFTB_PREFIX', './')
            if not slako_dir.endswith('/'):
                slako_dir += '/'

        self.slako_dir = slako_dir

        self.default_parameters = dict(
            Hamiltonian_='DFTB',
            Hamiltonian_SlaterKosterFiles_='Type2FileNames',
            Hamiltonian_SlaterKosterFiles_Prefix=self.slako_dir,
            Hamiltonian_SlaterKosterFiles_Separator='"-"',
            Hamiltonian_SlaterKosterFiles_Suffix='".skf"',
            Hamiltonian_MaxAngularMomentum_='',
            Options_='',
            Options_WriteResultsTag='Yes')

        self.pcpot = None
        self.lines = None
        self.atoms = None
        self.atoms_input = None
        self.do_forces = False
        self.outfilename = 'dftb.out'

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms,
                                  **kwargs)

        # Determine number of spin channels
        try:
            entry = kwargs['Hamiltonian_SpinPolarisation']
            spinpol = 'colinear' in entry.lower()
        except KeyError:
            spinpol = False
        self.nspin = 2 if spinpol else 1

        # kpoint stuff by ase
        self.kpts = kpts
        self.kpts_coord = None

        if self.kpts is not None:
            initkey = 'Hamiltonian_KPointsAndWeights'
            mp_mesh = None
            offsets = None

            if isinstance(self.kpts, dict):
                if 'path' in self.kpts:
                    # kpts is path in Brillouin zone
                    self.parameters[initkey + '_'] = 'Klines '
                    self.kpts_coord = kpts2ndarray(self.kpts, atoms=atoms)
                else:
                    # kpts is (implicit) definition of
                    # Monkhorst-Pack grid
                    self.parameters[initkey + '_'] = 'SupercellFolding '
                    mp_mesh, offsets = kpts2sizeandoffsets(atoms=atoms,
                                                           **self.kpts)
            elif np.array(self.kpts).ndim == 1:
                # kpts is Monkhorst-Pack grid
                self.parameters[initkey + '_'] = 'SupercellFolding '
                mp_mesh = self.kpts
                offsets = [0.] * 3
            elif np.array(self.kpts).ndim == 2:
                # kpts is (N x 3) list/array of k-point coordinates
                # each will be given equal weight
                self.parameters[initkey + '_'] = ''
                self.kpts_coord = np.array(self.kpts)
            else:
                raise ValueError('Illegal kpts definition:' + str(self.kpts))

            if mp_mesh is not None:
                eps = 1e-10
                for i in range(3):
                    key = initkey + '_empty%03d' % i
                    val = [mp_mesh[i] if j == i else 0 for j in range(3)]
                    self.parameters[key] = ' '.join(map(str, val))
                    offsets[i] *= mp_mesh[i]
                    assert abs(offsets[i]) < eps or abs(offsets[i] - 0.5) < eps
                    # DFTB+ uses a different offset convention, where
                    # the k-point mesh is already Gamma-centered prior
                    # to the addition of any offsets
                    if mp_mesh[i] % 2 == 0:
                        offsets[i] += 0.5
                key = initkey + '_empty%03d' % 3
                self.parameters[key] = ' '.join(map(str, offsets))

            elif self.kpts_coord is not None:
                for i, c in enumerate(self.kpts_coord):
                    key = initkey + '_empty%09d' % i
                    c_str = ' '.join(map(str, c))
                    if 'Klines' in self.parameters[initkey + '_']:
                        c_str = '1 ' + c_str
                    else:
                        c_str += ' 1.0'
                    self.parameters[key] = c_str

    def write_dftb_in(self, outfile):
        """ Write the innput file for the dftb+ calculation.
            Geometry is taken always from the file 'geo_end.gen'.
        """

        outfile.write('Geometry = GenFormat { \n')
        outfile.write('    <<< "geo_end.gen" \n')
        outfile.write('} \n')
        outfile.write(' \n')

        params = self.parameters.copy()

        s = 'Hamiltonian_MaxAngularMomentum_'
        for key in params:
            if key.startswith(s) and len(key) > len(s):
                break
        else:
            # User didn't specify max angular mometa.  Get them from
            # the .skf files:
            symbols = set(self.atoms.get_chemical_symbols())
            for symbol in symbols:
                path = os.path.join(self.slako_dir,
                                    '{0}-{0}.skf'.format(symbol))
                l = read_max_angular_momentum(path)
                params[s + symbol] = '"{}"'.format('spdf'[l])

        # --------MAIN KEYWORDS-------
        previous_key = 'dummy_'
        myspace = ' '
        for key, value in sorted(params.items()):
            current_depth = key.rstrip('_').count('_')
            previous_depth = previous_key.rstrip('_').count('_')
            for my_backsclash in reversed(
                    range(previous_depth - current_depth)):
                outfile.write(3 * (1 + my_backsclash) * myspace + '} \n')
            outfile.write(3 * current_depth * myspace)
            if key.endswith('_') and len(value) > 0:
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' = ' + str(value) + '{ \n')
            elif (key.endswith('_') and (len(value) == 0)
                  and current_depth == 0):  # E.g. 'Options {'
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' ' + str(value) + '{ \n')
            elif (key.endswith('_') and (len(value) == 0)
                  and current_depth > 0):  # E.g. 'Hamiltonian_Max... = {'
                outfile.write(key.rstrip('_').rsplit('_')[-1] +
                              ' = ' + str(value) + '{ \n')
            elif key.count('_empty') == 1:
                outfile.write(str(value) + ' \n')
            elif ((key == 'Hamiltonian_ReadInitialCharges') and
                  (str(value).upper() == 'YES')):
                f1 = os.path.isfile(self.directory + os.sep + 'charges.dat')
                f2 = os.path.isfile(self.directory + os.sep + 'charges.bin')
                if not (f1 or f2):
                    print('charges.dat or .bin not found, switching off guess')
                    value = 'No'
                outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
            else:
                outfile.write(key.rsplit('_')[-1] + ' = ' + str(value) + ' \n')
            if self.pcpot is not None and ('DFTB' in str(value)):
                outfile.write('   ElectricField = { \n')
                outfile.write('      PointCharges = { \n')
                outfile.write(
                    '         CoordsAndCharges [Angstrom] = DirectRead { \n')
                outfile.write('            Records = ' +
                              str(len(self.pcpot.mmcharges)) + ' \n')
                outfile.write(
                    '            File = "dftb_external_charges.dat" \n')
                outfile.write('         } \n')
                outfile.write('      } \n')
                outfile.write('   } \n')
            previous_key = key
        current_depth = key.rstrip('_').count('_')
        for my_backsclash in reversed(range(current_depth)):
            outfile.write(3 * my_backsclash * myspace + '} \n')
        outfile.write('ParserOptions { \n')
        outfile.write('   IgnoreUnprocessedNodes = Yes  \n')
        outfile.write('} \n')
        if self.do_forces:
            outfile.write('Analysis { \n')
            outfile.write('   CalculateForces = Yes  \n')
            outfile.write('} \n')

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)
        # Ignore unit cell for molecules:
        if not atoms.pbc.any() and 'cell' in system_changes:
            system_changes.remove('cell')
        if self.pcpot and self.pcpot.mmpositions is not None:
            system_changes.append('positions')
        return system_changes

    def write_input(self, atoms, properties=None, system_changes=None):
        from ase.io import write
        if properties is not None:
            if 'forces' in properties or 'stress' in properties:
                self.do_forces = True
        FileIOCalculator.write_input(
            self, atoms, properties, system_changes)
        with open(os.path.join(self.directory, 'dftb_in.hsd'), 'w') as fd:
            self.write_dftb_in(fd)
        write(os.path.join(self.directory, 'geo_end.gen'), atoms,
              parallel=False)
        # self.atoms is none until results are read out,
        # then it is set to the ones at writing input
        self.atoms_input = atoms
        self.atoms = None
        if self.pcpot:
            self.pcpot.write_mmcharges('dftb_external_charges.dat')

    def read_results(self):
        """ all results are read from results.tag file
            It will be destroyed after it is read to avoid
            reading it once again after some runtime error """

        with open(os.path.join(self.directory, 'results.tag'), 'r') as fd:
            self.lines = fd.readlines()

        self.atoms = self.atoms_input
        charges, energy, dipole = self.read_charges_energy_dipole()
        if charges is not None:
            self.results['charges'] = charges
        self.results['energy'] = energy
        if dipole is not None:
            self.results['dipole'] = dipole
        if self.do_forces:
            forces = self.read_forces()
            self.results['forces'] = forces
        self.mmpositions = None

        # stress stuff begins
        sstring = 'stress'
        have_stress = False
        stress = list()
        for iline, line in enumerate(self.lines):
            if sstring in line:
                have_stress = True
                start = iline + 1
                end = start + 3
                for i in range(start, end):
                    cell = [float(x) for x in self.lines[i].split()]
                    stress.append(cell)
        if have_stress:
            stress = -np.array(stress) * Hartree / Bohr**3
            self.results['stress'] = stress.flat[[0, 4, 8, 5, 2, 1]]
        # stress stuff ends

        # eigenvalues and fermi levels
        fermi_levels = self.read_fermi_levels()
        if fermi_levels is not None:
            self.results['fermi_levels'] = fermi_levels

        eigenvalues = self.read_eigenvalues()
        if eigenvalues is not None:
            self.results['eigenvalues'] = eigenvalues

        # calculation was carried out with atoms written in write_input
        os.remove(os.path.join(self.directory, 'results.tag'))

    def read_forces(self):
        """Read Forces from dftb output file (results.tag)."""
        from ase.units import Hartree, Bohr

        # Initialise the indices so their scope
        # reaches outside of the for loop
        index_force_begin = -1
        index_force_end = -1

        # Force line indexes
        for iline, line in enumerate(self.lines):
            fstring = 'forces   '
            if line.find(fstring) >= 0:
                index_force_begin = iline + 1
                line1 = line.replace(':', ',')
                index_force_end = iline + 1 + \
                    int(line1.split(',')[-1])
                break

        gradients = []
        for j in range(index_force_begin, index_force_end):
            word = self.lines[j].split()
            gradients.append([float(word[k]) for k in range(0, 3)])

        return np.array(gradients) * Hartree / Bohr

    def read_charges_energy_dipole(self):
        """Get partial charges on atoms
            in case we cannot find charges they are set to None
        """
        with open(os.path.join(self.directory, 'detailed.out'), 'r') as fd:
            lines = fd.readlines()

        for line in lines:
            if line.strip().startswith('Total energy:'):
                energy = float(line.split()[2]) * Hartree
                break

        qm_charges = []
        for n, line in enumerate(lines):
            if ('Atom' and 'Charge' in line):
                chargestart = n + 1
                break
        else:
            # print('Warning: did not find DFTB-charges')
            # print('This is ok if flag SCC=No')
            return None, energy, None

        lines1 = lines[chargestart:(chargestart + len(self.atoms))]
        for line in lines1:
            qm_charges.append(float(line.split()[-1]))

        dipole = None
        for line in lines:
            if 'Dipole moment:' in line and 'au' in line:
                words = line.split()
                dipole = np.array(
                    [float(w) for w in words[-4:-1]]) * Bohr

        return np.array(qm_charges), energy, dipole

    def get_charges(self, atoms):
        """ Get the calculated charges
        this is inhereted to atoms object """
        if 'charges' in self.results:
            return self.results['charges']
        else:
            return None

    def read_eigenvalues(self):
        """ Read Eigenvalues from dftb output file (results.tag).
            Unfortunately, the order seems to be scrambled. """
        # Eigenvalue line indexes
        index_eig_begin = None
        for iline, line in enumerate(self.lines):
            fstring = 'eigenvalues   '
            if line.find(fstring) >= 0:
                index_eig_begin = iline + 1
                line1 = line.replace(':', ',')
                ncol, nband, nkpt, nspin = map(int, line1.split(',')[-4:])
                break
        else:
            return None

        # Take into account that the last row may lack
        # columns if nkpt * nspin * nband % ncol != 0
        nrow = int(np.ceil(nkpt * nspin * nband * 1. / ncol))
        index_eig_end = index_eig_begin + nrow
        ncol_last = len(self.lines[index_eig_end - 1].split())
        self.lines[index_eig_end - 1] += ' 0.0 ' * (ncol - ncol_last)

        eig = np.loadtxt(self.lines[index_eig_begin:index_eig_end]).flatten()
        eig *= Hartree
        N = nkpt * nband
        eigenvalues = [eig[i * N:(i + 1) * N].reshape((nkpt, nband))
                       for i in range(nspin)]

        return eigenvalues

    def read_fermi_levels(self):
        """ Read Fermi level(s) from dftb output file (results.tag). """
        # Fermi level line indexes
        for iline, line in enumerate(self.lines):
            fstring = 'fermi_level   '
            if line.find(fstring) >= 0:
                index_fermi = iline + 1
                break
        else:
            return None

        fermi_levels = []
        words = self.lines[index_fermi].split()
        assert len(words) in [1, 2], 'Expected either 1 or 2 Fermi levels'

        for word in words:
            e = float(word)
            # In non-spin-polarized calculations with DFTB+ v17.1,
            # two Fermi levels are given, with the second one being 0,
            # but we don't want to add that one to the list
            if abs(e) > 1e-8:
                fermi_levels.append(e)

        return np.array(fermi_levels) * Hartree

    def get_ibz_k_points(self):
        return self.kpts_coord.copy()

    def get_number_of_spins(self):
        return self.nspin

    def get_eigenvalues(self, kpt=0, spin=0):
        return self.results['eigenvalues'][spin][kpt].copy()

    def get_fermi_levels(self):
        return self.results['fermi_levels'].copy()

    def get_fermi_level(self):
        return max(self.get_fermi_levels())

    def embed(self, mmcharges=None, directory='./'):
        """Embed atoms in point-charges (mmcharges)
        """
        self.pcpot = PointChargePotential(mmcharges, self.directory)
        return self.pcpot

    def optimize(self, atoms, fmax, steps=None):
        """Relax internally with dftb

        atoms: atoms object
        fmax: maximal force
        steps: maximal number of relaxation steps
        """
        params = {'Driver_': 'LBFGS',
                  'Driver_MaxForceComponent': fmax * Bohr / Hartree}
        if steps:
            params.update({'Driver_MaxSteps': steps})
        self.parameters.update(params)
        self.parameters.update(params_constraints(atoms))

        # runs relaxation
        atoms.get_potential_energy()

        relaxed_positions = read(Path(self.directory)
                                 .joinpath('geo_end.gen')).positions
        self.atoms.set_positions(relaxed_positions, apply_constraint=False)

    def md(self, atoms, timestep, temperature, trajectory, steps):
        """Internal MD propagation

        atoms: atoms object
        timestep: time step in ase units
        temperature: initial temperature
        trajectory: Trajectory object open for writing
        steps: numer of steps to propagate
        """
        params = {
            'Driver_Steps': steps - 1,
            'Driver_TimeStep': 41.3413733 * timestep / fs,
            'Driver_Thermostat_InitialTemperature': temperature / Hartree}
        params['Driver_'] = self.parameters.get('Driver_', 'VelocityVerlet')
        params['Driver_Thermostat_'] = self.parameters.get(
            'Driver_Thermostat_', 'None')
        self.parameters.update(params)
        self.parameters.update(params_constraints(atoms))

        # runs md
        atoms.get_potential_energy()

        direc = Path(self.directory)
        structures = read(direc.joinpath('geo_end.xyz'), ':')
        with open(direc.joinpath('md.out')) as out:
            lines = out.readlines()

        for i, structure in enumerate(structures):
            imd = int(lines.pop(0).split()[-1])
            assert i == imd

            results = {}
            lines.pop(0)  # Potential Energy
            lines.pop(0)  # MD Kinetic Energy
            results['energy'] = float(lines.pop(0).split()[5])
            lines.pop(0)  # MD Temperature
            
            # Dipole moment
            line = lines.pop(0).replace('au', '')
            dipole = [float(w) for w in line.split()[-3:]]
            results['dipole'] = np.array(dipole) * Bohr
            
            lines.pop(0)  # Dipole moment Debye
            
            structure.calc = SinglePointCalculator(structure, **results)
            structure.calc.name = self.name
            params = self.parameters.copy()
            params['timestep'] = timestep
            structure.calc.parameters = params
            trajectory.write(structure)


def params_constraints(atoms):
    """Translate constraint objects into parameters"""
    params = {}
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            indices = constraint.get_indices()
            ranges = []
            start = False
            for i, a in enumerate(atoms):
                if i in indices:
                    if start:
                        ranges.append(':'.join([start, str(i)]))
                        start = False
                else:
                    if not start:
                        start = str(i + 1)
            if start:
                ranges.append(':'.join([start, str(i + 1)]))
            params['Driver_MovedAtoms'] = ' '.join(ranges)
        else:
            assert 0, 'Can not translate constraint {}'.format(constraint)
    return params


class PointChargePotential:
    def __init__(self, mmcharges, directory='./'):
        """Point-charge potential for DFTB+.
        """
        self.mmcharges = mmcharges
        self.directory = directory
        self.mmpositions = None
        self.mmforces = None

    def set_positions(self, mmpositions):
        self.mmpositions = mmpositions

    def set_charges(self, mmcharges):
        self.mmcharges = mmcharges

    def write_mmcharges(self, filename):
        """ mok all
        write external charges as monopoles for dftb+.

        """
        if self.mmcharges is None:
            print("DFTB: Warning: not writing exernal charges ")
            return
        with open(os.path.join(self.directory, filename), 'w') as charge_file:
            for [pos, charge] in zip(self.mmpositions, self.mmcharges):
                [x, y, z] = pos
                charge_file.write('%12.6f %12.6f %12.6f %12.6f \n'
                                  % (x, y, z, charge))

    def get_forces(self, calc, get_forces=True):
        """ returns forces on point charges if the flag get_forces=True """
        if get_forces:
            return self.read_forces_on_pointcharges()
        else:
            return np.zeros_like(self.mmpositions)

    def read_forces_on_pointcharges(self):
        """Read Forces from dftb output file (results.tag)."""
        from ase.units import Hartree, Bohr
        with open(os.path.join(self.directory, 'detailed.out'), 'r') as fd:
            lines = fd.readlines()

        external_forces = []
        for n, line in enumerate(lines):
            if ('Forces on external charges' in line):
                chargestart = n + 1
                break
        else:
            raise RuntimeError(
                'Problem in reading forces on MM external-charges')
        lines1 = lines[chargestart:(chargestart + len(self.mmcharges))]
        for line in lines1:
            external_forces.append(
                [float(i) for i in line.split()])
        return np.array(external_forces) * Hartree / Bohr


def read_max_angular_momentum(path):
    """Read maximum angular momentum from .skf file.

    See dftb.org for A detailed description of the Slater-Koster file format.
    """
    with open(path, 'r') as fd:
        line = fd.readline()
        if line[0] == '@':
            # Extended format
            fd.readline()
            l = 3
            pos = 9
        else:
            # Simple format:
            l = 2
            pos = 7

        # Sometimes there ar commas, sometimes not:
        line = fd.readline().replace(',', ' ')

        occs = [float(f) for f in line.split()[pos:pos + l + 1]]
        for f in occs:
            if f > 0.0:
                return l
            l -= 1
