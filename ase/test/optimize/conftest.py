import pytest

from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.emt import EMT


ATOMS = [
    bulk("Au"),
    molecule("CO"),
]

@pytest.fixture(name="reference_atoms", params=ATOMS)
def fixture_reference_atoms(request: pytest.FixtureRequest) -> Atoms:
    reference_atoms: Atoms = request.param
    reference_atoms.calc = EMT()
    _ = reference_atoms.get_potential_energy()
    return reference_atoms


@pytest.fixture(name="rattled_atoms")
def fixture_rattled_atoms(reference_atoms: Atoms) -> Atoms:
    rattled_atoms = reference_atoms.copy()
    rattled_atoms.rattle(stdev=0.1, seed=42)
    return rattled_atoms


# atoms, NEB, filter, dimer
@pytest.fixture(name="optimizable")
def fixture_optimizable():...
