import os
import time
from pathlib import Path

import pytest

import numpy as np

from ase.units import Ha, Bohr
from ase.atoms import Atoms
from ase.calculators.sphinx import SPHInX


@pytest.fixture
def atoms_E_F_ref():
    atoms = Atoms('FeCo', cell=[2.8, 2.8, 2.9], positions=[[0, 0, 0], [1.5, 1.4, 1.4]], pbc=[True]*3)
    atoms.set_initial_magnetic_moments([2.0, -2.1])

    # consistent with PAWs from
    # from "small core datasets" at http://users.wfu.edu/natalie/papers/pwpaw/newperiodictable/
    Eref = -7324.57959504685
    Fref = [[ 5.99842999e-01, -5.34789498e-07 ,-1.99365797e-01],
            [-5.99848203e-01 ,-5.96495978e-07,  1.99345953e-01]]

    return atoms, Eref, Fref


@pytest.mark.skipif('ASE_SPHINX_COMMAND' not in os.environ, reason='Need $ASE_SPHINX_COMMAND for full SPHInX Calculator test')
def test_sphinx_ref(tmpdir, atoms_E_F_ref):
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

@pytest.mark.skipif('ASE_SPHINX_COMMAND' not in os.environ, reason='Need $ASE_SPHINX_COMMAND for full SPHInX Calculator test')
def test_sphinx_finite_diff(tmpdir):
    atoms = Atoms('FeCo', cell=[2.8, 2.8, 2.8], positions=[[0, 0, 0], [1.5, 1.4, 1.4]], pbc=[True]*3)
    atoms.set_initial_magnetic_moments([1.5, 1.6])

    calc = SPHInX(eCut=300.0, spinpol=True, constrain_spins=True, kpts=[2, 2, 2],
            potentials_dir=Path(__file__).parent,
            potentials={'Fe': ('AtomPAW', 'Fe.atomicdata'), 'Co': ('AtomPAW', 'Co.atomicdata')},
            energy_tol=1e-6, scfDiag_maxSteps=300, scfDiag_preconditioner_scaling=0.1, directory=tmpdir)
    atoms.calc = calc

    # E0 = atoms.get_potential_energy()
    F0 = atoms.get_forces()

    dx = 0.02
    atoms.positions[0, 0] += dx / 2.0
    Ep = atoms.get_potential_energy()

    atoms.positions[0, 0] -= dx
    Em = atoms.get_potential_energy()

    print("FD", dx, (Ep - Em) / dx, -F0[0, 0])

    assert np.isclose((Ep - Em) / dx, -F0[0, 0], rtol=1e-4, atol=2.0e-2)
