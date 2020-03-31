"""Votca calculator interface.

API
---
.. autoclass:: votca

"""
import os

import h5py
import numpy as np
import re

from ..io.votca import write_votca
from ..units import Bohr, Hartree
from .calculator import FileIOCalculator, Parameters, ReadError


class VOTCA(FileIOCalculator):
    """ASE interface to VOTCA-XTP Only supports energies for now."""

    implemented_properties = ['energy', 'forces', 'singlets',
                              'triplets', 'qp', 'ks', 'qp_pert', 'transition_dipoles']

    command = f"xtp_tools -e dftgwbse -o dftgwbse.xml -t {os.cpu_count()} >  dftgwbse.log"

    default_parameters = {
        "charge": 0, "mult": 1, "task": "forces",
        "orcasimpleinput": "tightscf PBE def2-SVP",
        "orcablocks": "%scf maxiter 200 end"}

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='orca', atoms=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        self.pcpot = None

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write Input file for Orca."""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        p = self.parameters
        p.write(f'{self.label}.ase')
        p['label'] = self.label
        # if self.pcpot:  # also write point charge file and add things to input
        #    p['pcpot'] = self.pcpot

        write_votca(atoms, **p)

    def read(self, label):
        FileIOCalculator.read(self, label)
        if not os.path.isfile(f"{self.label}.out"):
            raise ReadError

        with open(f'{self.label}.inp') as f:
            for line in f:
                if line.startswith('geometry'):
                    break
            symbols = []
            positions = []
            for line in f:
                if line.startswith('end'):
                    break
                words = line.split()
                symbols.append(words[0])
                positions.append([float(word) for word in words[1:]])

        self.parameters = Parameters.read(self.label + '.ase')
        self.read_results(self)

    def read_results(self):
        self.read_energy()
        if self.parameters.task.find('forces') > -1:
            self.read_forces()

    def tdipoles_sorter(orb):
        groupedData = []
        permutation = []
        for ind in orb['transition_dipoles'].keys():
            permutation.append(int(ind[3:]))  # 3: skips over the 'ind' bit
            groupedData.append(orb['transition_dipoles'][ind][:].transpose()[0])
        groupedData = np.asarray(groupedData)
        return(groupedData[np.argsort(permutation)])

    def read_energy(self):
        """Read Energy from VOTCA-XTP log file."""
        orbFile = h5py.File('system.orb', 'r')
        orb = orbFile['QMdata']
        self.results['energy'] = orb.attrs['qm_energy']
        self.results['singlets'] = np.array(
            orb['BSE_singlet']['eigenvalues'][()]).transpose()[0]
        self.results['triplets'] = np.array(
            orb['BSE_triplet']['eigenvalues'][()]).transpose()[0]
        self.results['ks'] = np.array(
            orb['mos']['eigenvalues'][()]).transpose()[0]
        self.results['qp'] = np.array(
            orb['QPdiag']['eigenvalues'][()]).transpose()[0]
        groupedData = []
        permutation = []
        for ind in orb['transition_dipoles'].keys():
            permutation.append(int(ind[3:]))  # 3: skips over the 'ind' bit
            groupedData.append(orb['transition_dipoles'][ind][:].transpose()[0])
        groupedData = np.asarray(groupedData)
        self.results['transition_dipoles'] = groupedData[np.argsort(
            permutation)]
        self.results['qp_pert'] = np.array(
            orb['QPpert_energies'][()]).transpose()[0]

    def read_forces(self) -> None:
        """Read Forces from VOTCA logfile.

        The Votca-XTP Engrad output looks like:

        ... ... =========== ENGRAD SUMMARY =================================
        ... ...    Total energy:     -112.90643941 Hartree
        ... ...    0    -0.0000  -0.0000  -0.1626
        ... ...    1    +0.0000  +0.0000  +0.1626
        ... ... Saving data to system.orb
        ... ... Writing output to dftgwbse.out.xml
        """
        with open('dftgwbse.log', 'r') as f:
            raw_data = f.read()

        # Search for gradient
        start = re.search(r"ENGRAD SUMMARY", raw_data)
        lines = raw_data[start.start():].splitlines()

        # the second line containings the energy
        # energy = float(lines[1].split()[4])

        # read the gradient
        number_of_atoms = len(self.atoms.symbols)
        end_of_lines = 2 + number_of_atoms
        gradients = np.hstack(
            list(
                map(lambda x: np.array(x.split()[3:], np.float),
                    lines[2:end_of_lines]))).reshape(3, number_of_atoms)

        # Read the gradient
        self.results['forces'] = -gradients * Hartree / Bohr


    def embed(self, mmcharges=None, **parameters):
        """Embed atoms in point-charges (mmcharges)."""
        self.pcpot = PointChargePotential(mmcharges, label=self.label)
        return self.pcpot
