"""This module defines an ASE interface to SPHInX

https://sxrepo.mpie.de/

Written by:

    Noam Bernstein, noam.bernstein@nrl.navy.mil
"""

import os
import re
import warnings

from pathlib import Path
import numpy as np

from ase.units import Ry, Ha, Bohr
from ase.calculators.calculator import FileIOCalculator, Calculator, all_changes


class SPHInX(FileIOCalculator):
    """ A SPHInX calculator based on ase.calculator.FileIOCalculator 
    """

    implemented_properties = ['energy', 'free_energy', 'forces', 'magmoms']

    def __init__(self, restart=None,
                 ignore_bad_restart_file=FileIOCalculator._deprecated,
                 label='sphinx', atoms=None, clean=False, use_prev_calc=True,
                 **kwargs):
        """Construct a sphinx calculator.

        """
        # default parameters
        self.default_parameters = dict(
            potential_style='paw',
            potentials_dir='.',
            xc='PBE',
            spinpol=False,
            constrain_spins=False,
            guess_from_file=True,
            kpts_MP_offset=[0.0, 0.0, 0.0],
            kpts=None,
            smearing_type='Gaussian',
            smearing=0.1,
            energy_tol=1.0e-7,
            symmetry=False,
            scfDiag_prelim_energy_tol=[1.0e-3],
            scfDiag_blockSize=32,
            scfDiag_maxStepsCCG=4,
            scfDiag_rhoMixing=[0.5, 1.0],
            scfDiag_spinMixing=[0.5, 1.0],
            scfDiag_maxSteps=100,
            scfDiag_nPulaySteps=20,
            scfDiag_preconditioner_scaling=0.5,
            otherkeys=[])

        self.atoms = None
        self.do_clean = clean
        self.use_prev_calc = use_prev_calc

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms,
                                  **kwargs)

    def write_sphinx_in(self, atoms, initialGuess_rho_file=False, initialGuess_waves_file=False):
        """Write the input files base.sx and struct.sx
        """

        self._base_file = 'base.sx'
        self._struct_file = 'struct.sx'
        self._forces_file = 'forces.sx'

        self_dir = Path(self.directory)
        with open(self_dir / self._base_file, 'w') as fout:
            params = self.parameters
            fout.write(f'format {params["potential_style"]};\n')
            fout.write('include <parameters.sx>;\n')
            fout.write('\n')
            fout.write(f'include "{self._struct_file}";\n')
            fout.write('\n')

            # basis
            fout.write('basis {\n')
            fout.write(f'    eCut = {params["eCut"]} / {Ry};\n')
            kpt_offset = [0.0, 0.0, 0.0]
            if params['kpts'] is not None:
                assert len(params['kpts']) == 3
            if params['kpts'] is not None and np.product(params['kpts']) > 1:
                kpt_offset = params['kpts_MP_offset']
            fout.write(f'    kPoint {{ coords = {kpt_offset}; relative; }}\n')
            if params['kpts'] is not None and np.product(params['kpts']) > 1:
                fout.write(f'    folding = {list(params["kpts"])};\n')
            fout.write('}\n')

            pot_dir = Path(params.get('potentials_dir', os.environ.get('ASE_SPHINX_POT_DIR')))
            if params['potential_style'] == 'paw':
                # potentials
                fout.write('pawPot {\n')
                for pot_species in sorted(set(atoms.symbols)):
                    pot_type, pot_file = params['potentials'][pot_species]
                    fout.write('    species {\n')
                    fout.write(f'        name = "{pot_species}_{pot_file.replace("/","_")}";\n')
                    fout.write(f'        potType = "{pot_type}";\n')
                    fout.write(f'        element = "{pot_species}";\n')
                    fout.write(f'        potential = "{pot_dir / pot_file}";\n')
                    fout.write('    }\n')
                fout.write('}\n')

                # general params
                fout.write('PAWHamiltonian {\n')
                fout.write(f'    spinPolarized = {int(params["spinpol"])};\n')
                n_empty_states = params.get("empty_states", max(10, len(atoms) * 2))
                fout.write(f'    nEmptyStates = {n_empty_states};\n')
                fout.write(f'    ekt = {params["smearing"]};\n')
                if params["smearing_type"] == "Gaussian":
                    fout.write('    MethfesselPaxton = 1;\n')
                elif params["smearing_type"] == "FermiDirect":
                    fout.write('    FermiDirac = 0;\n')
                elif params["smearing_type"].startswith("MethfesselPaxton"):
                    t_m, t_a = params["smearing_type"].split(maxsplit=1)
                    fout.write(f'    {t_m} = {t_a};\n')
                else:
                    raise ValueError(f'Unsupported smearing_type {params["smearing_type"]}')
                fout.write(f'    xc = {params["xc"]};\n')
                fout.write('}\n')
            else:
                raise ValueError(f'Unsupported potential_style {params["potential_style"]}')

            def _get(val, index):
                try:
                    return val[index]
                except TypeError:
                    return val

            fout.write('main {\n')
            for subSCF_i, energy_tol in enumerate(params["scfDiag_prelim_energy_tol"] + [params["energy_tol"]]):
                fout.write('    scfDiag {\n')
                fout.write(f'        blockCCG {{ blockSize={_get(params["scfDiag_blockSize"], subSCF_i)}; ')
                fout.write(f'maxStepsCCG={_get(params["scfDiag_maxStepsCCG"], subSCF_i)}; }}\n')
                fout.write(f'        dEnergy = {energy_tol} / {Ha};\n')
                fout.write(f'        rhoMixing = {_get(params["scfDiag_rhoMixing"], subSCF_i)};\n')
                if params["spinpol"]:
                    fout.write(f'        spinMixing = {_get(params["scfDiag_spinMixing"], subSCF_i)};\n')
                fout.write(f'        maxSteps = {_get(params["scfDiag_maxSteps"], subSCF_i)};\n')
                fout.write(f'        nPulaySteps = {_get(params["scfDiag_nPulaySteps"], subSCF_i)};\n')
                fout.write(f'        preconditioner {{ type = KERKER; scaling = {_get(params["scfDiag_preconditioner_scaling"], subSCF_i)}; }}\n')
                fout.write('    }\n')
            fout.write(f'    evalForces {{ file = "{self._forces_file}"; }}\n')
            fout.write('}\n')

        with open(self_dir / self._struct_file, 'w') as fout:
            self.write_sphinx_struct(atoms, fout,
                                     initialGuess_rho_file=initialGuess_rho_file,
                                     initialGuess_waves_file=initialGuess_waves_file)

    def write_sphinx_struct(self, atoms, fout, initialGuess_rho_file=False, initialGuess_waves_file=False):
        spin_polarized = self.parameters["spinpol"]
        constrain_spins = self.parameters["constrain_spins"]

        fout.write('structure {\n')

        l = f'    cell = {1.0 / Bohr} * [ '
        fout.write(l)
        fout.write(f'[ {atoms.cell[0, 0]}, {atoms.cell[0, 1]}, {atoms.cell[0, 2]}],\n')
        fout.write((' ' * len(l)) + f'[ {atoms.cell[1, 0]}, {atoms.cell[1, 1]}, {atoms.cell[1, 2]}],\n')
        fout.write((' ' * len(l)) + f'[ {atoms.cell[2, 0]}, {atoms.cell[2, 1]}, {atoms.cell[2, 2]} ] ];\n')

        ats_order = []
        for symbol in sorted(set(atoms.symbols)):
            fout.write('    species {\n')
            fout.write(f'        element = "{symbol}";\n')
            at_inds = [i for i in range(len(atoms)) if atoms.symbols[i] == symbol]
            for at_ind in at_inds:
                ats_order.append(at_ind)
                fout.write(f'         atom {{ coords = {1.0 / Bohr} * [ '
                           f' {atoms.positions[at_ind,0]}, {atoms.positions[at_ind,1]}, {atoms.positions[at_ind,2]} ]; '
                           f' label="L_{symbol}_{at_ind}"; }}\n')
            fout.write('    }\n')

        if not self.parameters['symmetry']:
            # Empty symmetry section disables symmetry, and default is to use species
            # but not magnetic order
            fout.write('    symmetry { }\n')
        else:
            magmoms = atoms.get_initial_magnetic_moments()
            if spin_polarized and (magmoms != magmoms[0]).any():
                warnings.warn("SPHInX symmetry enabled, but automatic symmetry ignores magmoms, which are not identical and may break symmetry")

        fout.write('}\n')

        if spin_polarized:
            init_magmoms = atoms.get_initial_magnetic_moments()
        else:
            init_magmoms = None

        fout.write('initialGuess {\n')
        if initialGuess_waves_file:
            fout.write('    waves { file = "waves.sxb" }\n')
        else:
            fout.write('    waves { lcao {} }\n')
        if initialGuess_rho_file:
            fout.write('    rho { file = "rho.sxb"; }\n')
        else:
            fout.write('    rho { atomicOrbitals;')
            if init_magmoms is not None:
                fout.write('\n')
                for at_ind in ats_order:
                    fout.write(f'        atomicSpin {{ label="L_{atoms.symbols[at_ind]}_{at_ind}"; spin={init_magmoms[at_ind]}; }}\n')
                fout.write('    }\n')
            else:
                fout.write(' }\n')
        fout.write('}\n')

        if spin_polarized and constrain_spins:
            for at_ind in ats_order:
                fout.write('spinConstraint {\n')
                fout.write(f'    label = "L_{atoms.symbols[at_ind]}_{at_ind}";\n')
                fout.write(f'    constraint = {init_magmoms[at_ind]};\n')
                fout.write('}\n')

    def set_atoms(self, atoms):
        self.clean_restart()

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(
            self, atoms, properties, system_changes)
        use_prev_calc_rho = self.use_prev_calc and (Path(self.directory) / 'rho.sxb').is_file()
        use_prev_calc_waves = self.use_prev_calc and (Path(self.directory) / 'waves.sxb').is_file()
        self.write_sphinx_in(atoms, use_prev_calc_rho, use_prev_calc_waves)
        # self.atoms is none until results are read out,
        # then it is set to the ones at writing input
        self.atoms_input = atoms
        self.atoms = None

    # copied from FileIOCalculator.calculate
    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        self.write_input(self.atoms, properties, system_changes)
        self.command = re.sub(r' -i [^ ]+$', '', self.command)
        self.command += f' -i {self._base_file}'
        self.execute()
        self.read_results()

    def read_sphinx_forces(self, infile):
        text = ' '.join([l.strip() for l in infile.readlines()])
        # should really select just "structure" section, but need to parse balanced {} for that
        structure_text = text

        cell_re = re.search(r'cell\s*=\s*\[\s*\[\s*(\S*)\s*,\s*(\S*)\s*,\s*(\S*)\s*\]\s*,'
                                    r'\s*\[\s*(\S*)\s*,\s*(\S*)\s*,\s*(\S*)\s*\]\s*,'
                                    r'\s*\[\s*(\S*)\s*,\s*(\S*)\s*,\s*(\S*)\s*\]\s*]\s*;', structure_text)
        # assuming no subgroups in "atom", just scalars/flags/arrays
        atoms_re = re.findall(r'\b(?:atom\s*{([^}]*)}|element\s*=\s*"([A-Z][a-z]?)")', structure_text)
        pos_a = []
        forces_a = []
        labels_a = []
        symbols_a = []
        for atom_str, element_str in atoms_re:
            if len(element_str) > 0:
                cur_species = element_str
                continue

            symbols_a.append(cur_species)

            pos_re = re.search(r'\bcoords\s*=\s*\[\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*\];', atom_str)
            pos_a.append([float(pos_re.group(i)) for i in range(1, 4)])

            forces_re = re.search(r'\bforce\s*=\s*\[\s*(\S+)\s*,\s*(\S+)\s*,\s*(\S+)\s*\];', atom_str)
            if forces_re:
                forces_a.append([float(forces_re.group(i)) for i in range(1, 4)])

            label_re = re.search(r'\blabel\s*=\s*"([^"]*)"\s*;', atom_str)
            labels_a.append(label_re.group(1))

        cell_a = np.asarray([float(cell_re.group(i)) for i in range(1, 10)]).reshape((3, 3))
        pos_a = np.asarray(pos_a)
        labels_a = np.asarray(labels_a)
        symbols_a = np.asarray(symbols_a)
        if len(forces_a) > 0:
            forces_a = np.asarray(forces_a)
        else:
            forces_a = None

        return cell_a, symbols_a, pos_a, labels_a, forces_a

    def clean_restart(self):
        for f in ['rho.sxb', 'waves.sxb']:
            try:
                (Path(self.directory) / f).unlink()
            except FileNotFoundError:
                pass

    def clean(self, clean_restart=True):
        for g in ['AtomsOrbitals*.dat', 'energy.dat', 'eps.*.dat', 'fftwisdon.dat',
                  'parallelHierarchy.sx.actual', 'residue.dat', 'spins.dat', 'vElStat-eV.sxb',
                  self._base_file, self._struct_file, self._forces_file]:
            for f in Path(self.directory).glob(g):
                f.unlink()

        if clean_restart:
            self.clean_restart()

    def read_results(self):
        """ all results are read from forces.sx, energy.dat, and spins.dat files
            It will be destroyed after it is read to avoid
            reading it once again after some runtime error """
        self_dir = Path(self.directory)

        # read structure/forces
        with open(self_dir / self._forces_file) as forces_infile:
            cell, symbols, pos, labels, forces = self.read_sphinx_forces(forces_infile)

        order = [int(re.sub('.*_', '', l)) for l in labels]
        order_rev = np.zeros((len(order)), dtype=int)
        order_rev[order] = np.arange(len(order))
        symbols = symbols[order_rev]
        pos = pos[order_rev]
        labels = labels[order_rev]
        forces = forces[order_rev]
        self.results['forces'] = forces

        # read energy
        # Use usual convention that 'energy' is T -> 0 extrapolated, 'free_energy'
        # is energy consistent with forces, and unmodified DFT energy isn't actually
        # returned
        with open(self_dir / 'energy.dat') as fin:
            f = fin.readlines()[-1].strip().split()
            # E = float(f[2])
            EF = float(f[3])
            E0 = float(f[4])
        self.results['energy'] = E0
        self.results['free_energy'] = EF

        # read spins
        if self.parameters["spinpol"]:
            with open(self_dir / 'spins.dat') as fin:
                l = fin.readlines()[-1].strip()
            self.results['magmoms'] = np.asarray([float(s) for s in l.split()[1:]])

        if self.do_clean:
            self.clean()

        self.atoms = self.atoms_input
        # if we ever implement moving atoms, this will be needed,
        # although file round trip roundoff breaks caching
        #
        # self.atoms.set_cell(cell * Bohr, True)
        # self.atoms.positions[:] = pos * Bohr
