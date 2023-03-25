import io
import numpy as np
from numpy.testing import assert_allclose
import pytest
import re

import ase.build
from ase.io import write, read
from ase.io.castep import (_read_forces_from_castep_file,
                           _read_stress_from_castep_file)


# create mol with custom mass - from a list of positions or using
# ase.build.molecule
def write_read_atoms(atom, tmp_path):
    write("{0}/{1}".format(tmp_path, "castep_test.cell"), atom)
    return read("{0}/{1}".format(tmp_path, "castep_test.cell"))


# write to .cell and check that .cell has correct species_mass block in it
@pytest.mark.parametrize(
    "mol, custom_masses, expected_species, expected_mass_block",
    [
        ("CH4", {2: [1]}, ["C", "H:0", "H", "H", "H"], ["H:0 2.0"]),
        ("CH4", {2: [1, 2, 3, 4]}, ["C", "H", "H", "H", "H"], ["H 2.0"]),
        ("C2H5", {2: [2, 3]}, ["C", "C", "H:0",
         "H:0", "H", "H", "H"], ["H:0 2.0"]),
        (
            "C2H5",
            {2: [2], 3: [3]},
            ["C", "C", "H:0", "H:1", "H", "H", "H"],
            ["H:0 2.0", "H:1 3.0"],
        ),
    ],
)
def test_custom_mass_write(
    mol, custom_masses, expected_species, expected_mass_block, tmp_path
):

    custom_atoms = ase.build.molecule(mol)
    atom_positions = custom_atoms.positions

    for mass, indices in custom_masses.items():
        for i in indices:
            custom_atoms[i].mass = mass

    atom_masses = custom_atoms.get_masses()
    with pytest.warns(UserWarning):
        # CASTEP IO is noisy while handling keywords JSON
        new_atoms = write_read_atoms(custom_atoms, tmp_path)

    # check atoms have been written and read correctly
    np.testing.assert_allclose(atom_positions, new_atoms.positions)
    np.testing.assert_allclose(atom_masses, new_atoms.get_masses())

    # check that file contains appropriate blocks
    with open("{0}/{1}".format(tmp_path, "castep_test.cell"), "r") as f:
        data = f.read().replace("\n", "\\n")

    position_block = re.search(
        r"%BLOCK POSITIONS_ABS.*%ENDBLOCK POSITIONS_ABS", data)
    assert position_block

    pos = position_block.group().split("\\n")[1:-1]
    species = [p.split(" ")[0] for p in pos]
    assert species == expected_species

    mass_block = re.search(r"%BLOCK SPECIES_MASS.*%ENDBLOCK SPECIES_MASS", data)
    assert mass_block

    masses = mass_block.group().split("\\n")[1:-1]
    for line, expected_line in zip(masses, expected_mass_block):
        species_name, mass_read = line.split(' ')
        expected_species_name, expected_mass = expected_line.split(' ')
        assert pytest.approx(float(mass_read), abs=1e-6) == float(expected_mass)
        assert species_name == expected_species_name


# test setting a custom species on different atom before write
def test_custom_mass_overwrite(tmp_path):
    custom_atoms = ase.build.molecule("CH4")
    custom_atoms[1].mass = 2
    with pytest.warns(UserWarning):
        # CASTEP IO is noisy while handling keywords JSON
        atoms = write_read_atoms(custom_atoms, tmp_path)

    # test that changing masses when custom masses defined causes errors
    atoms[3].mass = 3
    with pytest.raises(ValueError,
                       match="Could not write custom mass block for H."):
        atoms.write("{0}/{1}".format(tmp_path, "castep_test2.cell"))


def test_read_forces_from_castep_file():
    # removed some ' ' and '*' from example for line length limit
    example = """
 ***************************** Constrained Forces *****************************
 *                                                                            *
 *                        Cartesian components (eV/A)                         *
 * -------------------------------------------------------------------------- *
 *                     x                    y                    z            *
 *                                                                            *
 * Si            1     1.12300            -1.45600             1.78900        *
 * Si            2    -2.12300             2.45600            -2.78900        *
 ******************************************************************************

 ***************************** Constrained Forces *****************************
 *                                                                            *
 *                        Cartesian components (eV/A)                         *
 * -------------------------------------------------------------------------- *
 *                     x                    y                    z            *
 *                                                                            *
 * Si            1    -1.12300             1.45600            -1.78900        *
 * Si            2     2.12300            -2.45600             2.78900        *
 ******************************************************************************
"""
    fd = io.StringIO(example)

    # First block
    forces = _read_forces_from_castep_file(fd)
    desired = np.array([[1.123, -1.456, 1.789], [-2.123, 2.456, -2.789]])
    assert_allclose(forces, desired)

    # Second block
    forces = _read_forces_from_castep_file(fd)
    desired *= -1
    assert_allclose(forces, desired)

    # No block
    forces = _read_forces_from_castep_file(fd)
    assert forces is None


def test_read_stress_from_castep_file():
    # removed some ' ' and '*' from example for line length limit
    example = """
 ***************** Stress Tensor *****************
 *                                               *
 *          Cartesian components (GPa)           *
 * --------------------------------------------- *
 *             x             y             z     *
 *                                               *
 *  x      1.123000     -1.456000      1.789000  *
 *  y     -2.123000      2.456000     -2.789000  *
 *  z      3.123000     -3.456000      3.789000  *
 *                                               *
 *  Pressure:   -2.4563                          *
 *                                               *
 *************************************************

 ***************** Stress Tensor *****************
 *                                               *
 *          Cartesian components (GPa)           *
 * --------------------------------------------- *
 *             x             y             z     *
 *                                               *
 *  x     -1.123000      1.456000     -1.789000  *
 *  y      2.123000     -2.456000      2.789000  *
 *  z     -3.123000      3.456000     -3.789000  *
 *                                               *
 *  Pressure:    2.4563                          *
 *                                               *
 *************************************************
"""
    fd = io.StringIO(example)

    # First block
    stress = _read_stress_from_castep_file(fd)
    desired = np.array([[1.123, -1.456, 1.789],
                        [-2.123, 2.456, -2.789],
                        [3.123, -3.456, 3.789]])
    assert_allclose(stress, desired)

    # Second block
    stress = _read_stress_from_castep_file(fd)
    desired *= -1
    assert_allclose(stress, desired)

    # No block
    stress = _read_stress_from_castep_file(fd)
    assert stress is None


def test_atom_order():
    from ase.io.castep import atom_order
    custom_species = ['Li:a', 'Si', 'Si:a', 'O:a',
                      'Si', 'Si:a', 'O', 'Si']

    expected_sort = [6, 1, 4, 7, 0, 2, 5, 3]

    # i.e. Li:a Si Si:a O:a Si Si:a O Si
    # -->  O Si Si Si Li:a Si:a Si:a O:a

    assert atom_order(custom_species) == expected_sort
