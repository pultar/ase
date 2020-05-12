import numpy as np
import pytest

from ase import Atoms
from ase.calculators.lj import LennardJones


# See https://en.wikipedia.org/wiki/Lennard-Jones_potential
potential_energy_reference = pytest.approx(-1.0)


@pytest.fixture
def atoms(rc=1.0e5):
    """parametrized Atoms object with Calculator attached ready for testing"""
    atoms = Atoms("H2", positions=[[0, 0, 0], [0, 0, 2 ** (1.0 / 6.0)]])
    calc = LennardJones(rc=rc)
    atoms.calc = calc

    return atoms


def test_minimum_energy(atoms):
    assert atoms.get_potential_energy() == potential_energy_reference


def test_system_changes(atoms):
    atoms.calc.calculate(atoms, system_changes=["positions"])
    assert atoms.get_potential_energy() == potential_energy_reference


def test_forces(atoms):
    forces = atoms.get_forces()

    r = 2 ** (1.0 / 6.0)

    analytical_f_10 = (
        -24
        * atoms.calc.parameters.epsilon
        * (2 * atoms.calc.parameters.sigma ** 12 / r ** 14 - atoms.calc.parameters.sigma ** 6 / r ** 8)
        * np.array([0.0, 0.0, -r])
    )

    np.testing.assert_array_almost_equal(forces[0], -analytical_f_10)
    np.testing.assert_array_almost_equal(forces[1], analytical_f_10)
