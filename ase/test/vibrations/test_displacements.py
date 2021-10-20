import pytest
import numpy as np
import typing as tp

from ase.atoms import Atoms
from ase.vibrations.displacements import AxisAlignedDisplacements


class AxisAlignedPolynomial:
    """A function whose value is a polynomial with respect to all atom positions.

    The polynomial has no mixed terms.  It will look like (order = 4):
                         4          k
               sum    sum     c    x
                  a,i    n=0   aik  a,i
    (a = atom index, ijkl = cartesian indices)

    The output can be a tensor of arbitrary rank, if there are additional
    dimensions after the first 3 in ``coeffs``. (the first three are atom,
    cartesian axis, x power)
    """

    def __init__(self, coeffs):
        assert coeffs.ndim >= 3
        assert coeffs.shape[1] == 3

        self.coeffs = coeffs.copy()
        self.order = coeffs.shape[2] - 1

        deriv_coeffs = coeffs.copy()
        deriv_coeffs = np.moveaxis(deriv_coeffs, 2, -1)  # put power in last axis
        deriv_coeffs *= np.arange(self.order + 1)  # multiply by power
        deriv_coeffs = np.roll(deriv_coeffs, -1, axis=-1)  # reduce power by 1
        deriv_coeffs = np.moveaxis(deriv_coeffs, -1, 2)  # put axes back in original order
        self.deriv_coeffs = deriv_coeffs

    @classmethod
    def random(cls, rng, atoms, order, shape=()):
        coeffs = rng.random((len(atoms), 3, order + 1) + tuple(shape))
        return cls(coeffs)

    def value(self, atoms):
        positions = atoms.get_positions()
        pos_powers = positions[:, :, None] ** np.arange(self.order + 1)[None, None, :]
        return np.einsum('ain...,ain->...', self.coeffs, pos_powers)

    def derivative(self, atoms):
        """Compute the analytically known derivative.

        Output dimensions:  (atom, axis, ...field_dimensions)"""
        positions = atoms.get_positions()
        pos_powers = positions[:, :, None] ** np.arange(self.order + 1)[None, None, :]
        return np.einsum('ain...,ain->ai...', self.deriv_coeffs, pos_powers)


def test_axis_aligned_stencil_approximations():
    class TestCase(tp.NamedTuple):
        direction: str  # AxisAlignedDisplacements arg
        nfree: int  # AxisAlignedDisplacements arg
        exact_order: int  # max polynomial order for which the approximation is exact
        # test relative tolerance for a step of 1e-3, which may
        # include considerations of the power of h in the error term
        e3_reltol: float

    test_cases = [
        TestCase(direction='central', nfree=2, exact_order=2, e3_reltol=1e-4),
        TestCase(direction='central', nfree=4, exact_order=4, e3_reltol=1e-7),
        TestCase(direction='forward', nfree=2, exact_order=1, e3_reltol=1e-2),
        TestCase(direction='backward', nfree=2, exact_order=1, e3_reltol=1e-2),
    ]

    rng = np.random.RandomState(42)

    # factors out some of the test body, as we want to check level of precision
    # against polynomials of different order
    def compute_deriv_via_displacements(poly: AxisAlignedPolynomial,
                                        case: TestCase,
                                        step: float):
        displacements = AxisAlignedDisplacements(atoms, direction=case.direction,
                                                 nfree=case.nfree, delta=step)
        disp_values = np.array([
            poly.value(new_atoms)
            for (_, new_atoms) in displacements.iter_with_atoms(atoms)
        ])
        actual_derivs = displacements.compute_cartesian_derivatives(disp_values)
        return actual_derivs

    natom = 4
    atoms = Atoms(symbols=f'H{natom}', positions=np.ones((natom, 3)))

    for _ in range(10000):
        for case in test_cases:
            # typical case: a small step to compute something for which
            #               our approximation is inexact
            poly = AxisAlignedPolynomial.random(rng, atoms, order=case.exact_order + 1)
            actual_derivs = compute_deriv_via_displacements(poly, case, step=1e-3)
            expected_derivs = poly.derivative(atoms)
            assert pytest.approx(actual_derivs, rel=case.e3_reltol) == expected_derivs, case

            # error test: demand high precision for a polynomial where
            #             our approximation is exact
            poly = AxisAlignedPolynomial.random(rng, atoms, order=case.exact_order)
            actual_derivs = compute_deriv_via_displacements(poly, case, step=1e0)
            expected_derivs = poly.derivative(atoms)
            assert pytest.approx(actual_derivs, rel=1e-9) == expected_derivs
