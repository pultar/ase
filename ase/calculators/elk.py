from pathlib import Path

from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import FileIOCalculator
from ase.io import write
from ase.io.elk import ElkReader

import os
from ase.calculators.genericfileio import (
    GenericFileIOCalculator, CalculatorTemplate, read_stdout)
from ase.io import read, write
from ase.io.elk import write_elk_in

import warnings


class ElkProfile:
    def __init__(self, argv):
        self.argv = tuple(argv)

    @staticmethod
    def parse_version(stdout):
        import re
        # match = re.match(r'\s*Program PWSCF\s*v\.(\S+)', stdout, re.M)
        match = re.match(r"\s*Elk code version\s*\.(\S+) started", stdout, re.M)
        assert match is not None
        return match.group(1)

    def version(self):
        stdout = read_stdout(self.argv)
        return self.parse_version(stdout)

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        import os
        argv = list(self.argv)
        with open(directory / outputfile, 'wb') as fd:
            check_call(argv, cwd=directory, stdout=fd, env=os.environ)

    # def socketio_argv_unix(self, socket):
    #     template = EspressoTemplate()
    #     # It makes sense to know the template for this kind of choices,
    #     # but is there a better way?
    #     return list(self.argv) + ['--ipi', f'{socket}:UNIX', '-in',
    #                               template.inputname]


class ElkTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            'elk',
            ['energy'])
        self.inputname = 'elk.in'
        self.outputname = 'elk.out'

    def write_input(self, directory, atoms, parameters, properties):
        dst = directory / self.inputname
        write_elk_in(dst, atoms, parameters=parameters)

    def execute(self, directory, profile):
        profile.run(directory,
                    self.inputname,
                    self.outputname)

    def read_results(self, directory):
        from ase.outputs import Properties
        reader = ElkReader(directory)
        dct = dict(reader.read_everything())

        converged = dct.pop('converged')
        if not converged:
            warnings.warn('Did not converge')

        # (Filter results thorugh Properties for error detection)
        props = Properties(dct)
        return dict(props)



class Elk(GenericFileIOCalculator):
    def __init__(self, *, profile=None,
                 command=GenericFileIOCalculator._deprecated,
                 label=GenericFileIOCalculator._deprecated,
                 directory='.',
                 **kwargs):

        if command is not self._deprecated:
            raise RuntimeError(compatibility_msg)

        if label is not self._deprecated:
            import warnings
            warnings.warn('Ignoring label, please use directory instead',
                          FutureWarning)

        if 'ASE_ELK_COMMAND' in os.environ and profile is None:
            import warnings
            warnings.warn(compatibility_msg, FutureWarning)

        template = ElkTemplate()
        if profile is None:
            profile = ElkProfile(argv=['elk'])
        super().__init__(profile=profile, template=template,
                         directory=directory,
                         parameters=kwargs)
