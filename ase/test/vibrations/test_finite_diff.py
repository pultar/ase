import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from ase.atoms import Atoms
from ase.calculators.qmmm import ForceConstantCalculator
from ase.calculators.singlepoint import SinglePointCalculator
import ase.db
import ase.vibrations.finite_diff


@pytest.fixture
def random_dimer():
    rng = np.random.RandomState(42)

    d = 1 + 0.5 * rng.random()
    z_values = rng.randint(1, high=50, size=2)

    hessian = rng.random((6, 6))
    hessian += hessian.T  # Ensure the random Hessian is symmetric

    atoms = Atoms(z_values, [[0, 0, 0], [0, 0, d]])
    ref_atoms = atoms.copy()
    atoms.calc = ForceConstantCalculator(D=hessian,
                                         ref=ref_atoms,
                                         f0=np.zeros((2, 3)))
    return atoms


def simple_dimer():
    return Atoms('CuS', positions=[[0., 0., 0.], [1., 0., 0.]])


def displacements_from_list(ref_atoms, specifications, h=0.01):
    displacements = []
    for (atom_index, cartesian_index, direction) in specifications:
        atoms = ref_atoms.copy()
        atoms[atom_index].position[cartesian_index] += h * direction
        displacements.append(atoms)
    return displacements


def labels_from_list(specifications):
    def dir_to_index(direction: int):
        if direction == -1:
            return 0
        elif direction == 1:
            return 1
        else:
            raise ValueError(f'Could not interpret direction "{direction}"')

    return [f'{atom_index}-{cart_index}-{dir_to_index(direction)}'
            for atom_index, cart_index, direction in specifications]


# Standard case: two directions, default distance
full_displacement_spec = [(0, 0, -1), (0, 0, 1), (0, 1, -1), (0, 1, 1),
                          (0, 2, -1), (0, 2, 1), (1, 0, -1), (1, 0, 1),
                          (1, 1, -1), (1, 1, 1), (1, 2, -1), (1, 2, 1)]
dimer_full_displacements = displacements_from_list(simple_dimer(),
                                                   full_displacement_spec,
                                                   h=0.01)
dimer_full_labels = labels_from_list(full_displacement_spec)

# Limit indices to one atom, increase displacement
indexed_displacement_spec = [(1, 0, -1), (1, 0, 1), (1, 1, -1), (1, 1, 1),
                             (1, 2, -1), (1, 2, 1)]
indexed_displacements = displacements_from_list(
    simple_dimer(), indexed_displacement_spec, h=0.02)
indexed_labels = labels_from_list(indexed_displacement_spec)

# Forward displacements only
forward_displacement_spec = [(0, 0, 1), (0, 1, 1), (0, 2, 1), (1, 0, 1),
                             (1, 1, 1), (1, 2, 1)]
forward_displacements = displacements_from_list(
    simple_dimer(), forward_displacement_spec)
forward_labels = labels_from_list(forward_displacement_spec)

# Backward displacements only
backward_displacement_spec = [(0, 0, -1), (0, 1, -1), (0, 2, -1), (1, 0, -1),
                              (1, 1, -1), (1, 2, -1)]
backward_displacements = displacements_from_list(
    simple_dimer(), backward_displacement_spec)
backward_labels = labels_from_list(backward_displacement_spec)


@pytest.mark.parametrize('options, expected_output',
                         [({}, list(zip(dimer_full_displacements,
                                        dimer_full_labels))),
                          ({'indices': [1], 'delta': 0.02},
                           list(zip(indexed_displacements,
                                    indexed_labels))),
                          ({'direction': 'forward'},
                           list(zip(forward_displacements, forward_labels))),
                          ({'direction': 'backward'},
                           list(zip(backward_displacements, backward_labels))),
                          ])
def test_get_displacements_with_identities(options, expected_output):
    output = ase.vibrations.finite_diff.get_displacements_with_identities(
        simple_dimer(), **options)

    for ((atoms, label), (expected_atoms, expected_label)
         ) in zip(output, expected_output):
        assert label == expected_label
        assert_array_almost_equal(atoms.positions, expected_atoms.positions)
        assert (atoms.get_chemical_symbols()
                == expected_atoms.get_chemical_symbols())


@pytest.mark.parametrize('options, expected_error',
                         [({'direction': 'inside-out'}, ValueError)])
def test_get_displacements_invalid(options, expected_error):
    with pytest.raises(expected_error):
        _ = ase.vibrations.finite_diff.get_displacements_with_identities(
            simple_dimer(), **options)


def attach_forces(atoms, forces):
    atoms.calc = SinglePointCalculator(atoms, forces=forces)
    return atoms


def test_read_axis_aligned_forces(random_dimer):
    """Check that Hessian is recovered perfectly when using Hessian calculator
    """
    ref_hessian = random_dimer.calc.D

    displacements = displacements_from_list(random_dimer,
                                            full_displacement_spec)
    for displacement in displacements:
        displacement.calc = ForceConstantCalculator(D=ref_hessian,
                                                    ref=random_dimer,
                                                    f0=np.zeros((2, 3)))

    # With ref atoms
    vib_data = ase.vibrations.finite_diff.read_axis_aligned_forces(
        displacements, ref_atoms=random_dimer)
    assert_array_almost_equal(vib_data.get_hessian_2d(), ref_hessian)

    # Without ref atoms
    vib_data = ase.vibrations.finite_diff.read_axis_aligned_forces(
        displacements)
    assert_array_almost_equal(vib_data.get_hessian_2d(), ref_hessian)

    # Raise error if user requests use of unavailable equilibrium forces
    dimer_no_forces = random_dimer.copy()
    with pytest.raises(ValueError):
        ase.vibrations.finite_diff.read_axis_aligned_forces(
            displacements, ref_atoms=dimer_no_forces,
            use_equilibrium_forces=True)

    dimer_no_forces.calc = SinglePointCalculator(dimer_no_forces)
    with pytest.raises(ValueError):
        ase.vibrations.finite_diff.read_axis_aligned_forces(
            displacements, ref_atoms=dimer_no_forces,
            use_equilibrium_forces=True)

    # Raise error if displacements not all axis-aligned
    displacements[4].rattle()
    with pytest.raises(ValueError):
        ase.vibrations.finite_diff.read_axis_aligned_forces(
            displacements, ref_atoms=random_dimer)

    # Raise warning if not enough displacements available
    del displacements[4]
    del displacements[4]
    with pytest.warns(UserWarning):
        ase.vibrations.finite_diff.read_axis_aligned_forces(
            displacements, ref_atoms=random_dimer)


def test_db_workflow(testdir, random_dimer):
    """Check that Hessian is recovered when using database workflow"""

    ref_hessian = random_dimer.calc.D
    metadata = {'phase': 'random'}
    db_file = 'vibtest.db'

    ase.vibrations.finite_diff.write_displacements_to_db(random_dimer,
                                                         db=db_file,
                                                         metadata=metadata)

    with ase.db.connect(db_file, append=True) as db:
        for row in db.select(**metadata):
            atoms = row.toatoms()
            atoms.calc = ForceConstantCalculator(D=ref_hessian,
                                                 ref=random_dimer,
                                                 f0=np.zeros((2, 3)))
            atoms.get_forces()
            db.write(atoms, name='extra', **metadata)

        # Write some irrelevant atoms to DB with different metadata
        junk = Atoms('H')
        junk.calc = SinglePointCalculator(junk, forces=[[1., 0., 0.]])
        db.write(junk, name='extra')

    vib_data = ase.vibrations.finite_diff.read_axis_aligned_db(
        db=db_file, ref_atoms=random_dimer, metadata=metadata)
    assert_array_almost_equal(vib_data.get_hessian_2d(), ref_hessian)

    # Should also work without ref atoms
    vib_data = ase.vibrations.finite_diff.read_axis_aligned_db(
        db=db_file, metadata=metadata)
    assert_array_almost_equal(vib_data.get_hessian_2d(), ref_hessian)


def test_guess_ref_atoms(random_dimer):
    displacements = displacements_from_list(
        random_dimer, full_displacement_spec, h=1e-2)

    guess_atoms = ase.vibrations.finite_diff.guess_ref_atoms(displacements)

    assert_array_almost_equal(guess_atoms.positions, random_dimer.positions)
    assert (guess_atoms.get_chemical_symbols()
            == random_dimer.get_chemical_symbols())
    assert_array_equal(guess_atoms.pbc, random_dimer.pbc)

    # Raise error if no repeated values to work with
    ambiguous_displacements = displacements_from_list(
        random_dimer, [(0, 0, 1), (0, 0, 2), (0, 0, 3)])

    with pytest.raises(ValueError):
        ase.vibrations.finite_diff.guess_ref_atoms(ambiguous_displacements)
