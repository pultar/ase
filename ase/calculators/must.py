import numpy as np
from ase.units import Bohr, Rydberg
from ase.calculators.calculator import FileIOCalculator
from ase.io.must import write_atomic_pot_input, write_positions_input, write_single_site_pot_input, \
    write_input_parameters_file
import os
import subprocess
import glob
import warnings


class CpaAtoms:
    """
    CPA atoms class for KKR-CPA method.
    """

    def __init__(self, a, bravais_lattice, cpa_atoms):
        """
        a (float) : lattice parameter in Angstrom
        bravais_lattice (str or array) : bravais lattice vectors
        cpa_atoms (list): list of dictionaries having key-value pairs as position:(x,y,z) and symbol:occupancy
        """
        self.lattice_parameter = a / Bohr  # Bohr Radius

        lattice_structures = {'bcc': np.array([[0.500, 0.500, -0.500],
                                               [0.500, -0.500, 0.500],
                                               [-0.500, 0.500, 0.500]]),
                              'fcc': np.array([[0.500, 0.500, 0.000],
                                               [0.500, 0.000, 0.500],
                                               [0.000, 0.500, 0.500]])}
        if bravais_lattice in lattice_structures.keys():
            self.lattice = lattice_structures[bravais_lattice]

        else:
            self.lattice = bravais_lattice
        self.cpa_atoms = cpa_atoms


def generate_starting_potentials(atoms, crystal_type, a, nspins=1, moment=0., xc=1, lmax=3,
                                 print_level=1, ncomp=1, conc=1., mt_radius=0., ws_radius=0,
                                 egrid=(10, -0.4, 0.3), ef=0.7, niter=50, mp=0.1):
    species = np.unique(atoms.get_chemical_symbols())

    for symbol in species:
        write_atomic_pot_input(symbol, nspins=nspins, moment=moment,
                               xc=xc, niter=niter, mp=mp)

        write_single_site_pot_input(symbol=symbol, crystal_type=crystal_type,
                                    a=a, nspins=nspins, moment=moment, xc=xc,
                                    lmax=lmax, print_level=print_level, ncomp=ncomp,
                                    conc=conc, mt_radius=mt_radius, ws_radius=ws_radius,
                                    egrid=egrid, ef=ef, niter=niter, mp=mp)
        # Generate atomic potential
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
    default_parameters = dict(pot_in_form=0, pot_out_form=0,
                              stop_rout_name='main', nscf=60, method=2, out_to_scr='n',
                              temperature=0.0, val_e_rel=0, core_e_rel=0,
                              potential_type=0, xc=4, uniform_grid=(64, 64, 64),
                              read_mesh=False, n_egrids=30, erbot=-0.40, real_axis_method=0, real_axis_points=300,
                              spin=1, lmax_T=3, ndivin=1001, liz_cutoff=4.5, k_scheme=0, kpts=(10, 10, 10), bzsym=1,
                              mix_algo=2, mix_quantity=1, mix_param=0.1, etol=5e-5, ptol=1e-6,
                              em_iter=80, em_scheme=1, em_mix_param=(0.1, 0.1), em_eswitch=0.04, em_tm_tol=1e-7)

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='mst', atoms=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties=None, system_changes=None)

        write_positions_input(atoms)
        write_input_parameters_file(atoms=atoms, pot_in_form=self.parameters['pot_in_form'],
                                    pot_out_form=self.parameters['pot_out_form'],
                                    stop_rout_name=self.parameters['stop_rout_name'],
                                    nscf=self.parameters['nscf'],
                                    method=self.parameters['method'],
                                    out_to_scr=self.parameters['out_to_scr'],
                                    temperature=self.parameters['temperature'],
                                    val_e_rel=self.parameters['val_e_rel'],
                                    core_e_rel=self.parameters['core_e_rel'],
                                    potential_type=self.parameters['potential_type'],
                                    xc=self.parameters['xc'],
                                    uniform_grid=self.parameters['uniform_grid'],
                                    read_mesh=self.parameters['read_mesh'],
                                    n_egrids=self.parameters['n_egrids'],
                                    erbot=self.parameters['erbot'],
                                    real_axis_method=self.parameters['real_axis_method'],
                                    real_axis_points=self.parameters['real_axis_points'],
                                    spin=self.parameters['spin'],
                                    lmax_T=self.parameters['lmax_T'],
                                    ndivin=self.parameters['ndivin'],
                                    liz_cutoff=self.parameters['liz_cutoff'],
                                    k_scheme=self.parameters['k_scheme'],
                                    kpts=self.parameters['kpts'],
                                    bzsym=self.parameters['bzsym'],
                                    mix_algo=self.parameters['mix_algo'],
                                    mix_quantity=self.parameters['mix_quantity'],
                                    mix_param=self.parameters['mix_param'],
                                    etol=self.parameters['etol'],
                                    ptol=self.parameters['ptol'],
                                    em_iter=self.parameters['em_iter'],
                                    em_scheme=self.parameters['em_scheme'],
                                    em_mix_param=self.parameters['em_mix_param'],
                                    em_eswitch=self.parameters['em_eswitch'],
                                    em_tm_tol=self.parameters['em_tm_tol'])

    def read_results(self):
        outfile = glob.glob('k_n00000_*')[0]
        with open(outfile, 'r') as file:
            lines = file.readlines()

        e_offset = float(lines[7].split()[-1])

        results = {tag: value for tag, value in zip(lines[9].split(), lines[-1].split())}

        read_energy = (float(results['Energy']) + e_offset) * Rydberg

        if float(results['Rms_pot']) > self.parameters['ptol']:
            warnings.warn('SCF Convergence not reached (Rms_pot > ptol)', UserWarning)

        self.results['energy'] = read_energy


if __name__ == '__main__':
    from ase.build import bulk

    Atoms = bulk('Al', 'bcc', a=3.624, cubic=True)
    Atoms[1].symbol = 'Fe'
    calc = MuST(atoms=Atoms)
    Atoms.set_calculator(calc)
    generate_starting_potentials(Atoms, crystal_type=1, a=3.624)
    energy = calc.get_potential_energy()
    print(energy)
