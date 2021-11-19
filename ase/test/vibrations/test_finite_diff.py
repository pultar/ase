import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from ase.atoms import Atoms
from ase.calculators.qmmm import ForceConstantCalculator
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

@pytest.fixture
def simple_dimer():
    return Atoms('CuS', positions=[[0., 0., 0.], [1., 0., 0.]])

#        label = f'{atom_index}-{cartesian_index}-{(sign + 1) // 2}'

dimer_full_displacements = [Atoms('CuS', positions=[[-0.01, 0., 0.], [1., 0., 0.]]),
                            Atoms('CuS', positions=[[ 0.01, 0., 0.], [1., 0., 0.]]),
                            Atoms('CuS', positions=[[0., -0.01, 0.], [1., 0., 0.]]),
                            Atoms('CuS', positions=[[0.,  0.01, 0.], [1., 0., 0.]]),
                            Atoms('CuS', positions=[[0., 0., -0.01], [1., 0., 0.]]),
                            Atoms('CuS', positions=[[0., 0.,  0.01], [1., 0., 0.]]),
                            Atoms('CuS', positions=[[0., 0., 0.], [0.99, 0., 0.]]), 
                            Atoms('CuS', positions=[[0., 0., 0.], [1.01, 0., 0.]]), 
                            Atoms('CuS', positions=[[0., 0., 0.], [1., -0.01, 0.]]),
                            Atoms('CuS', positions=[[0., 0., 0.], [1.,  0.01, 0.]]),
                            Atoms('CuS', positions=[[0., 0., 0.], [1., 0., -0.01]]),
                            Atoms('CuS', positions=[[0., 0., 0.], [1., 0.,  0.01]])]
dimer_full_labels = [ '0-0-0', '0-0-1', '0-1-0', '0-1-1', '0-2-0', '0-2-1',
                      '1-0-0', '1-0-1', '1-1-0', '1-1-1', '1-2-0', '1-2-1']

@pytest.mark.parametrize('options, expected_output',
                         [({}, list(zip(dimer_full_displacements,
                                        dimer_full_labels))),
                          ])
def test_get_displacements_with_identities(simple_dimer, options, expected_output):
    output = ase.vibrations.finite_diff.get_displacements_with_identities(simple_dimer, **options)

    for ((atoms, label), (expected_atoms, expected_label)) in zip(output, expected_output):
        assert label == expected_label
        assert_array_almost_equal(atoms.positions, expected_atoms.positions)
        assert atoms.get_chemical_symbols() == expected_atoms.get_chemical_symbols()
        
from ase.calculators.singlepoint import SinglePointCalculator

def attach_forces(atoms, forces):
    atoms.calc = SinglePointCalculator(atoms, forces=forces)
    return atoms

full_displacement_set = [(0,0,-1), (0,0,1), (0,1,-1), (0,1,1), (0,2,-1), (0,2,1),
                         (1,0,-1), (1,0,1), (1,1,-1), (1,1,1), (1,2,-1), (1,2,1)]

def displacements_from_list(ref_atoms, specifications, h=0.01):
    displacements = []
    for (atom_index, cartesian_index, direction) in specifications:
        atoms = ref_atoms.copy()
        atoms[atom_index].position[cartesian_index] += h * direction
        displacements.append(atoms)
    return displacements

def test_read_forces_direct(random_dimer):

    ref_hessian = random_dimer.calc.D

    displacements = displacements_from_list(random_dimer, full_displacement_set)
    for displacement in displacements:
        displacement.calc = ForceConstantCalculator(D=ref_hessian,
                                                    ref=random_dimer,
                                                    f0=np.zeros((2, 3)))
        displacement.get_forces()

    ase.vibrations.finite_diff.read_forces_direct(random_dimer,
                                                  displacements,
                                                  method='standard')

