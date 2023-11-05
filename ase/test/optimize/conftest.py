import pytest

from ase import Atoms
from ase.build import bulk
from ase.calculators.emt import EMT


ATOMS = [
    bulk("Au") * 2,
    Atoms("CO", positions=[[0, 0, 0], [1.0559702954160726, 0, 0]], cell=[5, 5, 5]),
]


@pytest.fixture(name="reference_atoms", params=ATOMS)
def fixture_reference_atoms(request: pytest.FixtureRequest) -> Atoms:
    reference_atoms: Atoms = request.param
    reference_atoms.calc = EMT()
    return reference_atoms


@pytest.fixture(name="reference_energy")
def fixture_reference_energy(reference_atoms: Atoms) -> float:
    reference_energy: float = reference_atoms.get_potential_energy()
    return reference_energy


@pytest.fixture(name="rattled_atoms")
def fixture_rattled_atoms(reference_atoms: Atoms) -> Atoms:
    rattled_atoms = reference_atoms.copy()
    rattled_atoms.rattle(stdev=0.1, seed=42)
    rattled_atoms.calc = EMT()
    return rattled_atoms


# atoms, NEB, filter, dimer
@pytest.fixture(name="optimizable")
def fixture_optimizable():
    ...
