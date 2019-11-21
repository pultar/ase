import numpy as np
from ase import Atoms
from ase.calculators.aims import Aims, AimsCube
from ase.calculators.calculator import PropertyNotImplementedError

# fmt: off
forces_reference = np.array(
    [[-8.24725677e-01, -1.07717924e+00, -1.01412140e-29],
     [ 1.90190491e+00,  1.90190491e+00, -5.07060702e-29],
     [-1.07717924e+00, -8.24725677e-01, -1.01412140e-29]]
)

energy_reference = -2078.34265186113
# fmt: on


water = Atoms("HOH", [(1, 0, 0), (0, 0, 0), (0, 1, 0)])

water_cube = AimsCube(
    points=(29, 29, 29),
    plots=("total_density", "delta_density", "eigenstate 5", "eigenstate 6"),
)

calc = Aims(xc="PBE", output=["dipole"], cubes=water_cube, compute_forces=True)

water.set_calculator(calc)


def test_energy(atoms=water):
    energy = water.get_total_energy()
    assert np.allclose(energy, energy_reference), energy


def test_forces(atoms=water):
    forces = water.get_forces()
    assert np.allclose(forces, forces_reference), forces


def test_stress(atoms=water):
    """this test should fail since water is not periodic"""
    try:
        _ = water.get_stress()
    except PropertyNotImplementedError:
        pass


if __name__ == "__main__":
    test_energy()
    test_forces()
    test_stress()
