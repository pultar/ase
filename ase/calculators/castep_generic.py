from ase.calculators.genericfileio import (
    GenericFileIOCalculator, CalculatorTemplate)


class CastepProfile:
    def __init__(self, argv):
        self.argv = argv

    def run(self):
        # execute argv and direct the output somewhere
        ...


class CastepTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            name='castep',
            implemented_properties=['energy', 'forces'],  # ?
        )

    def write_input(self, directory, atoms, parameters, properties):
        ...

    def execute(self, directory, profile):
        ...

    def read_results(self, directory):
        ...


class Castep(GenericFileIOCalculator):
    def __init__(self, directory, profile=None, **kwargs):

        if profile is None:
            profile = CastepProfile(...)

        super().__init__(
            self,
            profile=profile,
            template=CastepTemplate(),
            directory=directory,
            parameters=kwargs)
