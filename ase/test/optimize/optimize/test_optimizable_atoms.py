import pytest

from ase import Atoms
from ase.optimize.optimize import OptimizableAtoms


@pytest.fixture(name="optimizable_atoms")
def fixture_optimizable_atoms(rattled_atoms: Atoms) -> OptimizableAtoms:
    return OptimizableAtoms(rattled_atoms)


class TestOptimizableAtoms:
    @staticmethod
    def test_should_get_positions(optimizable_atoms: OptimizableAtoms, rattled_atoms: Atoms):
        assert (optimizable_atoms.get_positions() == rattled_atoms.get_positions()).all()

    @staticmethod
    def test_should_set_positions(optimizable_atoms: OptimizableAtoms):
        copied_atoms = optimizable_atoms.atoms.copy()
        copied_optimizable = OptimizableAtoms(copied_atoms)
        new_positions = copied_optimizable.get_positions() + [1, 1, 1]
        copied_optimizable.set_positions(new_positions)
        assert (copied_optimizable.get_positions() == new_positions).all()

    @staticmethod
    def test_should_get_forces(optimizable_atoms: OptimizableAtoms, rattled_atoms: Atoms):
        assert (optimizable_atoms.get_forces() == rattled_atoms.get_forces()).all()

    @staticmethod
    def test_should_get_potential_energy(
        optimizable_atoms: OptimizableAtoms, rattled_atoms: Atoms
    ):
        assert optimizable_atoms.get_potential_energy() == rattled_atoms.get_potential_energy()

    @staticmethod
    def test_should_iterimages(optimizable_atoms: OptimizableAtoms, rattled_atoms: Atoms):
        assert next(optimizable_atoms.iterimages()) == rattled_atoms

    @staticmethod
    def test_should_get_chemical_symbols(
        optimizable_atoms: OptimizableAtoms, rattled_atoms: Atoms
    ):
        assert optimizable_atoms.get_chemical_symbols() == rattled_atoms.get_chemical_symbols()
