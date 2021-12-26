"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>
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

    def version(self):
        from subprocess import check_output
        return check_output(self.argv + ['--version'])

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        with open(outputfile, 'w') as fd:
            check_call(self.argv + [str(inputfile)], stdout=fd,
                       cwd=directory)


class OpenmxTemplate(CalculatorTemplate):
    _label = 'abinit'  # Controls naming of files within calculation directory

    def __init__(self):
        super().__init__(
            name='openmx',
            implemented_properties=['energy', 'free_energy',
                                    'forces', 'stress', 'magmom'])

        self.input_file = f'{self._label}.dat'
        self.output_file = f'{self._label}.log'

    def execute(self, directory, profile) -> None:
        profile.run(directory, self.input_file, self.output_file)

    def write_input(self, directory, atoms, parameters, properties):
        directory.mkdir(exist_ok=True, parents=True)
        dst = directory / self.inputname
        write(dst, atoms, format='openmx-in', properties=properties,
              **parameters)


    def read_results(self, directory):
        path = directory / self.outputname
        atoms = read(path, format='openmx-out')
        return dict(atoms.calc.properties())


class Openmx(GenericFileIOCalculator):
    """Class for doing OpenMX calculations.

    """

    def __init__(self, *, profile=None, directory='.', **kwargs):
        """Construct OpenMX-calculator object.

        Parameters
        ==========
        label: str
            Prefix to use for filenames (label.in, label.txt, ...).
            Default is 'openmx'.


        """

        if profile is None:
            profile = OpenmxProfile(['openmx'])

        super().__init__(template=OpenmxTemplate(),
                         profile=profile,
                         directory=directory,
                         parameters=kwargs)
