from pathlib import Path

from ase.calculators.abc import GetOutputsMixin
from ase.calculators.calculator import FileIOCalculator
from ase.io import write
from ase.io.elk import ElkReader


class ELK(FileIOCalculator, GetOutputsMixin):
    implemented_properties = ['energy', 'forces']
    ignored_changes = {'pbc'}
    discard_results_on_any_change = True

    def __init__(self, **kwargs):
        """Construct ELK calculator.

        The keyword arguments (kwargs) can be one of the ASE standard
        keywords: 'xc', 'kpts' and 'smearing' or any of ELK'
        native keywords.
        """
        command = kwargs.pop("command", None)
        if command is None:
            command = 'elk > elk.out'
        super().__init__(command=command, **kwargs)

    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        parameters = dict(self.parameters)
        if 'forces' in properties:
            parameters['tforce'] = True

        directory = Path(self.directory)
        write(directory / 'elk.in', atoms, parameters=parameters,
              format='elk-in')

    def read_results(self):
        from ase.outputs import Properties
        reader = ElkReader(self.directory)
        dct = dict(reader.read_everything())

        converged = dct.pop('converged')
        if not converged:
            raise RuntimeError('Did not converge')

        # (Filter results thorugh Properties for error detection)
        props = Properties(dct)
        self.results = dict(props)

    def _outputmixin_get_results(self):
        return self.results
