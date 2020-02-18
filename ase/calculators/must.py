"""ASE interface for MST implemented in MuST package available at
 https://github.com/mstsuite/MuST"""

import numpy as np
from ase.units import Rydberg
from ase.calculators.calculator import FileIOCalculator
from ase.io.must import must as io
import os
import subprocess
import glob
import warnings


def generate_starting_potentials(atoms, crystal_type, a, nspins=1, moment=0., xc=1, lmax=3,
                                 print_level=1, ncomp=1, conc=1., mt_radius=0., ws_radius=0,
                                 egrid=(10, -0.4, 0.3), ef=0.7, niter=50, mp=0.1):
    species = np.unique(atoms.get_chemical_symbols())

    for symbol in species:
        # Generate atomic potential
        io.write_atomic_pot_input(symbol, nspins=nspins, moment=moment,
                                  xc=xc, niter=niter, mp=mp)

        newa = 'newa < ' + str(symbol) + '_a_in'
        try:
            proc = subprocess.Popen(newa, shell=True, cwd='.')
        except OSError as err:
            msg = 'Failed to execute "{}"'.format(newa)
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath('.')
            msg = ('newa failed with command "{}" failed in '
                   '{} with error code {}'.format(newa, path, errorcode))
            print(msg)
            break

        # Generate single site potential
        io.write_single_site_pot_input(symbol=symbol, crystal_type=crystal_type,
                                       a=a, nspins=nspins, moment=moment, xc=xc,
                                       lmax=lmax, print_level=print_level, ncomp=ncomp,
                                       conc=conc, mt_radius=mt_radius, ws_radius=ws_radius,
                                       egrid=egrid, ef=ef, niter=niter, mp=mp)

        newss = 'newss < ' + str(symbol) + '_ss_in'
        try:
            proc = subprocess.Popen(newss, shell=True, cwd='.')
        except OSError as err:
            msg = 'Failed to execute "{}"'.format(newss)
            raise EnvironmentError(msg) from err

        errorcode = proc.wait()

        if errorcode:
            path = os.path.abspath('.')
            msg = ('newss failed with command "{}" failed in '
                   '{} with error code {}'.format(newss, path, errorcode))
            print(msg)
            break


class MuST(FileIOCalculator):
    """
    Multiple Scattering Theory based ab-initio calculator
    """

    implemented_properties = ['energy']
    command = 'mst2 < i_new'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='mst', atoms=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties=None, system_changes=None)

        # Write positions using CPA sites if self.parameters['method'] == 3
        if 'method' in self.parameters.keys():
            if self.parameters['method'] == 3:
                io.write_positions_input(atoms, method=self.parameters['method'])
            else:
                io.write_positions_input(atoms, method=None)

        else:
            io.write_positions_input(atoms, method=None)

        io.write_input_parameters_file(atoms=atoms, parameters=self.parameters)

    def read_results(self):
        outfile = glob.glob('k_n00000_*')[0]
        with open(outfile, 'r') as file:
            lines = file.readlines()

        e_offset = float(lines[7].split()[-1])

        results = {tag: value for tag, value in zip(lines[9].split(), lines[-1].split())}
        read_energy = (float(results['Energy']) + e_offset)

        convergence = False

        outfile = glob.glob('o_n00000_*')[0]
        with open(outfile, 'r') as file:
            lines = file.readlines()

        for line in lines:
            if 'SCF Convergence is reached' in line:
                convergence = True
                break

        if convergence is False:
            warnings.warn('SCF Convergence not reached', UserWarning)

        self.results['energy'] = read_energy * Rydberg
