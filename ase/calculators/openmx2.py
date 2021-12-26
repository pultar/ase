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

import ase.io.openmx as io
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)

default_orbitals = {
    'H': '6.0-s3p2', 'He': '8.0-s2p2d1', 'Li': '10.0-s3p3d2',
    'Be': '8.0-s3p2', 'B': '8.0-s2p2d1', 'C': '6.0-s2p2d1',
    'N': '6.0-s3p3d2f1', 'O': '6.0-s3p3d2', 'F': '6.0-s2p2d1',
    'Ne': '9.0-s3p2d2', 'Na': '11.0-s3p3d2', 'Mg': '9.0-s3p3d2',
    'Al': '8.0-s4p4d2', 'Si': '8.0-s2p2d1', 'P': '8.0-s4p3d3f2',
    'S': '8.0-s4p3d3f2', 'Cl': '8.0-s2p2d1', 'Ar': '9.0-s3p2d2f1',
    'K': '12.0-s4p3d3f1', 'Ca': '11.0-s4p3d2', 'Sc': '9.0-s4p3d2',
    'Ti': '9.0-s3p3d3f1', 'V': '8.0-s3p3d3f1',
    'Cr': '8.0-s3p3d2', 'Mn': '8.0-s3p3d3f1', 'Fe': '8.0-s3p3d2',
    'Co': '8.0-s3p4d3f2', 'Ni': '8.0-s4p4d3f2', 'Cu': '8.0-s2p2d2',
    'Zn': '6.0-s3p3d2f2', 'Ga': '7.0-s2p2d2', 'Ge': '7.0-s3p3d3f2',
    'As': '7.0-s2p2d2f1', 'Se': '9.0-s3p3d3f2', 'Br': '9.0-s3p3d3f2',
    'Kr': '9.0-s3p3d3f2', 'Rb': '9.0-s3p3d3f2', 'Sr': '9.0-s3p3d3f2',
    'Y': '9.0-s3p3d3f2', 'Zr': '9.0-s3p3d3f2', 'Nb': '9.0-s3p3d3f2',
    'Mo': '9.0-s3p3d3f2', 'Tc': '9.0-s3p3d3f2', 'Ru': '9.0-s3p3d3f2',
    'Rh': '9.0-s3p3d3f2', 'Pd': '9.0-s3p3d3f2', 'Ag': '9.0-s3p3d3f2',
    'Cd': '9.0-s3p3d3f2', 'In': '9.0-s3p3d3f2', 'Sn': '9.0-s3p3d3f2',
    'Sb': '9.0-s3p3d3f2', 'Te': '9.0-s3p3d3f2', 'I': '9.0-s3p3d3f2',
    'Xe': '9.0-s3p3d3f2', 'Cs': '9.0-s3p3d3f2', 'Ba': '9.0-s3p3d3f2',
    'Nd': '9.0-s3p3d3f2', 'Sm': '9.0-s3p3d3f2', 'Dy': '9.0-s3p3d3f2',
    'Ho': '9.0-s3p3d3f2', 'Lu': '9.0-s3p3d3f2', 'Hf': '9.0-s3p3d3f2',
    'Ta': '9.0-s3p3d3f2', 'W': '9.0-s3p3d3f2', 'Re': '9.0-s3p3d3f2',
    'Os': '9.0-s3p3d3f2', 'Ir': '9.0-s3p3d3f2', 'Pt': '9.0-s3p3d3f2',
    'Au': '9.0-s3p3d3f2', 'Hg': '9.0-s3p3d3f2', 'Tl': '9.0-s3p3d3f2',
    'Pb': '9.0-s3p3d3f2', 'Bi': '9.0-s3p3d3f2', 'Po': '9.0-s3p3d3f2',
    'Rn': '9.0-s3p3d3f2'
}


class OpenmxProfile:
    def __init__(self, argv):
        self.argv = argv

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        with open(outputfile, 'w') as fd:
            print(self.argv, str(inputfile))
            check_call(self.argv + [str(inputfile)], stdout=fd,
                       cwd=directory)


class OpenmxTemplate(CalculatorTemplate):

    def __init__(self, name='openmx'):
        super().__init__(
            name=name,
            implemented_properties=['energy', 'free_energy',
                                    'forces', 'stress', 'magmom'])

    def execute(self, directory, profile) -> None:
        profile.run(directory, (self.name + '.dat'), (self.name + '.log'))

    def write_input(self, directory, atoms, parameters, properties):
        self.system_name = parameters.get('system_name', self.name)
        directory.mkdir(exist_ok=True, parents=True)
        dst = directory / (self.name + '.dat')
        io.write_openmx_in(dst, atoms,
                           properties=properties,
                           parameters=parameters)

    def read_results(self, directory):
        """ Read results from `.log` and `.out` files
        Since the `stress` is only written in the `.log` and structure info
        such as cell or position is

        """

        outfile = directory / (self.system_name + '.out')
        logfile = directory / (self.name + '.log')

        with open(outfile, 'r') as fd:
            outtext = fd.read()
        with open(logfile, 'r') as fd:
            logtext = fd.read()

        version = io.parse_openmx_out_version(outtext)
        scf_stress_tensor = \
            io.parse_openmx_out_scf_stress_tensor(outtext, version=version)
        md_maxiter = io.parse_openmx_out_md_maxiter(outtext, version=version)

        # Initial structure & energy, forces
        if md_maxiter is not None and md_maxiter > 1:
            outatoms = io.read_openmx_log(logfile)[-1]
        else:
            outatoms = io.read_openmx_out(outfile)
        results = outatoms.calc.results

        # Stress
        if scf_stress_tensor is not None:
            stress = io.parse_openmx_log_stress(logtext, version=version)
            results.update({'stress': stress})

        return results


class Openmx2(GenericFileIOCalculator):
    def __init__(self, *args, profile=None, template=None,
                 directory='.', **kwargs):

        profile = profile or OpenmxProfile(['openmx'])
        template = template or OpenmxTemplate()
        super().__init__(directory=directory,
                         template=template, profile=profile, parameters=kwargs)


def get_definition_of_atomic_species(atoms, scf_xctype=None, data_year=None):
    vps_dct = {'gga-pbe': 'PBE'}
    vps = vps_dct.get(scf_xctype.lower(), 'CA') + str(data_year)
    definition_of_atomic_species = []
    for s in list(set(atoms.get_chemical_symbols())):
        orbital = default_orbitals[s]
        definition_of_atomic_species.append([s, s + orbital, s + '_' + vps])

    return definition_of_atomic_species
