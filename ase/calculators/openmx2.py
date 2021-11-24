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


import re

import ase.io.openmx as io
from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)


def parse_omx_version(txt):
    """Parse version number from stdout header."""
    match = re.search(r'Welcome to OpenMX\s+Ver\.\s+(\S+)', txt, re.M)
    return match.group(1)

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
        outfile = directory / (self.system_name + '.out')
        logfile = directory / (self.name + '.log')
        return io.read_openmx_results(outfile, logfile)


class Openmx2(GenericFileIOCalculator):

    def __init__(self, *args, profile=None, template=None,
                 directory='.', **kwargs):

        profile = profile or OpenmxProfile(['openmx'])
        template = template or OpenmxTemplate()
        super().__init__(directory=directory,
                         template=template, profile=profile, parameters=kwargs)
