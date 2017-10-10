"""This test calls the lower triangular form reduction
code, which is self-validating
"""

import numpy as np
from ase.build.tools import lower_triangular_form, standardize_unit_cell
from ase.lattice.cubic import FaceCenteredCubic


def verify_cell(cell):

    """verify that the result is in lower triangular form."""

    assert abs(cell[0, 1]) < 1E-12
    assert abs(cell[0, 2]) < 1E-12
    assert abs(cell[1, 2]) < 1E-12


def test_lower_triangular():

    """check that the cell matrix is correctly reduced."""

    for i in range(500):
        cell = np.random.uniform(-1, 1, (3, 3))
        L = lower_triangular_form(cell)
        verify_cell(L)

        assert np.sign(np.linalg.det(cell)) == np.sign(np.linalg.det(L))
        '''scipy.linalg.rq (called in lower_triangular_form) appears to
        always return a right-handed Q matrix.  The assertion above is
        to verify this behaviour.  If it changes in the future, code
        must be added to change Q.
        '''

    for i in range(500):
        atoms = FaceCenteredCubic(size=(1, 1, 1), symbol='Cu', pbc=(1, 1, 1))
        standardize_unit_cell(atoms)
        verify_cell(atoms.cell)

test_lower_triangular()
