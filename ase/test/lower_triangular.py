"""This test calls the lower triangular form reduction
code, which is self-validating
"""

import numpy as np
from ase.build.tools import lower_triangular

def test_lower_triangular():
    """check that the cell matrix is correctly reduced."""

    for i in range(1000):
        cell = np.random.uniform(-1, 1, (3, 3))
        lower_triangular_form(cell)

test_lower_triangular()
