import numpy as np
from ase import Atoms
from ase.calculators.harmonic import Harmonic
from ase.optimize import BFGS
import pytest

pos = np.asarray([[8.7161, 7.96276, 8.48206], [8.60594, 8.04985, 9.44464],
                      [8.0154, 8.52264, 8.10545]])
ref_atoms = Atoms('OH2', positions=pos)  # relaxed water molecule
ref_energy = -14.222189  # y shift of the 'parabola' (harmonic potential)

# example Hessian matrix as obtained from central finite differences
hessian_x = np.asarray([[2.82630333e+01, -2.24763667e+01, 7.22478333e+00,
                         -2.96970000e+00, 2.34363333e+00, 2.72788333e+00,
                         -2.52159833e+01, 2.01307833e+01, -9.94651667e+00],
                        [-2.24763667e+01, 1.78621333e+01, -5.77378333e+00,
                         2.33703333e+00, -1.85276667e+00, -2.15118333e+00,
                         2.01258667e+01, -1.60350833e+01, 7.93248333e+00],
                        [7.22478333e+00, -5.77378333e+00, 5.72735000e+01,
                         7.25470000e+00, -5.75313333e+00, -4.69477333e+01,
                         -1.44613000e+01, 1.15504833e+01, -1.03523333e+01],
                        [-2.96970000e+00, 2.33703333e+00, 7.25470000e+00,
                         2.96170000e+00, -2.36901667e+00, -3.76841667e+00,
                         -2.83833333e-02, 3.06833333e-02, -3.49190000e+00],
                        [2.34363333e+00, -1.85276667e+00, -5.75313333e+00,
                         -2.36901667e+00, 1.89046667e+00, 2.95495000e+00,
                         2.90666667e-02, -1.80666667e-02, 2.79565000e+00],
                        [2.72788333e+00, -2.15118333e+00, -4.69477333e+01,
                         -3.76841667e+00, 2.95495000e+00, 4.89340000e+01,
                         1.03146667e+00, -8.18450000e-01, -1.96118333e+00],
                        [-2.52159833e+01, 2.01258667e+01, -1.44613000e+01,
                         -2.83833333e-02, 2.90666667e-02, 1.03146667e+00,
                         2.52034000e+01, -2.01516833e+01, 1.34293167e+01],
                        [2.01307833e+01, -1.60350833e+01, 1.15504833e+01,
                         3.06833333e-02, -1.80666667e-02, -8.18450000e-01,
                         -2.01516833e+01, 1.60592333e+01, -1.07369667e+01],
                        [-9.94651667e+00, 7.93248333e+00, -1.03523333e+01,
                         -3.49190000e+00, 2.79565000e+00, -1.96118333e+00,
                         1.34293167e+01, -1.07369667e+01, 1.23150000e+01]])


def assert_water_is_relaxed(atoms):
    forces = atoms.get_forces()
    assert np.allclose(np.zeros(forces.shape), forces)
    assert np.allclose(ref_energy, atoms.get_potential_energy())
    assert np.allclose(atoms.get_angle(1, 0, 2), ref_atoms.get_angle(1, 0, 2))
    assert np.allclose(atoms.get_distance(0, 1), ref_atoms.get_distance(0, 1))
    assert np.allclose(atoms.get_distance(0, 2), ref_atoms.get_distance(0, 2))


def run_optimize(atoms):
    opt = BFGS(atoms)
    opt.run(fmax=1e-9)


def test_cartesians():
    """In Cartesian coordinates the first 6 trash eigenvalues (translations and
    rotations) can have significant absolute values; hence set them to zero
    using an increased parameter zero_thresh.
    """
    zero_thresh = 0.06  # set eigvals to zero if abs(eigenvalue) < zero_thresh
    calc = Harmonic(ref_atoms=ref_atoms, ref_energy=ref_energy,
                    hessian_x=hessian_x, zero_thresh=zero_thresh)
    atoms = ref_atoms.copy()
    atoms.calc = calc
    assert_water_is_relaxed(atoms)  # atoms has not been distorted
    run_optimize(atoms)             # nothing should happen
    assert_water_is_relaxed(atoms)  # atoms should still be relaxed
    atoms.set_distance(0, 1, 3.5)   # now distort atoms along axis, no rotation
    run_optimize(atoms)             # optimization should recover original
    assert_water_is_relaxed(atoms)    # relaxed geometry

    with pytest.raises(AssertionError):
        atoms.rattle()                  # relaxation should fail to recover the
        atoms.rotate(90, 'x')           # original geometry of the atoms,
        run_optimize(atoms)             # because Cartesian coordinates are
        assert_water_is_relaxed(atoms)  # not rotationally invariant.


def test_constraints_with_cartesians():
    """Project out forces along x-component of H-atom (index 0 in the q-vector
    with the Cartesian coordinates (here: x=q)). A change in the x-component of
    the H-atom should not result in restoring forces, when they were projected
    out from the Hessian matrix.
    """
    def test_forces(calc):
        atoms = ref_atoms.copy()
        atoms.calc = calc
        newpos = pos.copy()
        newpos[0, 0] *= 2
        atoms.set_positions(newpos)
        run_optimize(atoms)  # (no) restoring force along distorted x-component
        xdiff = atoms.get_positions() - pos
        return all(xdiff[xdiff != 0] == newpos[0, 0] / 2)

    zero_thresh = 0.06  # set eigvals to zero if abs(eigenvalue) < zero_thresh
    calc = Harmonic(ref_atoms=ref_atoms, ref_energy=ref_energy,
                    hessian_x=hessian_x, zero_thresh=zero_thresh)
    assert not test_forces(calc)  # restoring force along distorted x-component

    calc.set(constrained_q=[0])  # project out the coordinate with index 0
    assert test_forces(calc)  # no restoring force along distorted x-component
