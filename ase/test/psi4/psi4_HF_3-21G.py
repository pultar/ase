#!/usr/bin/env python3

import os
import numpy as np
from numpy.testing import assert_allclose

from ase.build import molecule
from ase.calculators.psi4 import Psi4


def main():
    atoms = molecule('CH4')
    atoms.calc = Psi4(basis='3-21G')
    assert_allclose(atoms.get_potential_energy(), -1087.8229067535328)

    F1_ref = np.array([
        [-3.20324666e-09, +7.13619716e-10, -1.22699616e-07],
        [-2.03993160e-01, +4.51311288e-09, -1.44244915e-01],
        [+2.03993163e-01, -4.59098567e-09, -1.44244918e-01],
        [-4.50580226e-09, +2.03993067e-01, +1.44244978e-01],
        [+4.47061790e-09, -2.03993067e-01, +1.44244978e-01]])
    assert_allclose(atoms.get_forces(), F1_ref, atol=1e-14)

    # Test the reader
    calc2 = Psi4()
    calc2.read('psi4-calc')
    assert_allclose(calc2.results['energy'], atoms.get_potential_energy())
    assert_allclose(calc2.results['forces'], atoms.get_forces(), atol=1e-14)

    os.remove('psi4-calc.dat')
    # Unfortunately, we can't currently remove timer.dat, because Psi4
    # creates the file after this script exits. Not even atexit works.
    # os.remove('timer.dat')


main()
