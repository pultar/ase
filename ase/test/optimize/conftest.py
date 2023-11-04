import pytest

from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT


ATOMS = [
    bulk("Au"),
    molecule("CO"),
]

@pytest.fixture(name="atoms", params=ATOMS)
def fixture_atoms(request: pytest.FixtureRequest) -> Atoms:
    atoms: Atoms = request.param
    atoms.calc = EMT()
    _ = atoms.get_potential_energy()
    return atoms


# atoms, NEB, filter, dimer
@pytest.fixture(name="optimizable")
def fixture_optimizable():...
