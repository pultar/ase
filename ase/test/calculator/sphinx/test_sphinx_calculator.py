import os
import time
from pathlib import Path

import pytest

import numpy as np

from ase.atoms import Atoms
from ase.calculators.sphinx import SPHInX


@pytest.fixture
def atoms_E_F_ref():
    atoms = Atoms('FeCo', cell=[4, 4, 4], positions=[[0, 0, 0], [2.1, 2, 2]], pbc=[True]*3)
    atoms.set_initial_magnetic_moments([2.0, -2.1])

    # consistent with PAWs from
    # from "small core datasets" at http://users.wfu.edu/natalie/papers/pwpaw/newperiodictable/
    Eref = -269.04643944536
    Fref = [[-3.7064726e-03, 1.2093000e-06, 1.3123000e-06],
            [3.6769230e-03, 3.9721000e-06, -2.5040000e-06]]

    return atoms, Eref, Fref


@pytest.mark.skipif('ASE_SPHINX_COMMAND' not in os.environ, reason='Need $ASE_SPHINX_COMMAND for full SPHInX Calculator test')
def test_sphinx(tmpdir, atoms_E_F_ref):
    (atoms, Eref, Fref) = atoms_E_F_ref
    calc = SPHInX(eCut=300.0, spinpol=True, constrain_spins=True, kpts=[2, 2, 2],
            potentials_dir=Path(__file__).parent,
            potentials={'Fe': ('AtomPAW', 'Fe.atomicdata'), 'Co': ('AtomPAW', 'Co.atomicdata')},
            energy_tol=0.01, directory=tmpdir)
    atoms.calc = calc

    t0 = time.time()
    E = atoms.get_potential_energy()
    time_E = time.time() - t0
    t0 = time.time()
    F = atoms.get_forces()
    time_F = time.time() - t0

    print('E', E)
    print('F', F)
    assert np.isclose(E, Eref, rtol=1e-4, atol=2.0e-2)
    assert np.isclose(F, Fref, rtol=0.05, atol=1.0e-4).all()

    assert time_F < time_E / 100.0
