import os
import time
from pathlib import Path

import pytest

import numpy as np

from ase.atoms import Atoms
from ase.calculators.sphinx import SPHInX
from ase.calculators.ssa import SSA


@pytest.fixture
def atoms_E_F_ref():
    atoms = Atoms('FeCo', cell=[4, 4, 4], positions=[[0, 0, 0], [2.1, 2, 2]], pbc=[True]*3)

    atoms.new_array('SSA_initial_magnetic_moments',
                    np.asarray([[1.5, 2.0], [-1.6, -2.0]]))

    # consistent with PAWs from
    # from "small core datasets" at http://users.wfu.edu/natalie/papers/pwpaw/newperiodictable/
    Eref = -269.05145818943504
    Fref = [[-0.00271627, 0., 0.],
            [0.00282854, 0., 0.]]

    return atoms, Eref, Fref


@pytest.mark.skipif('ASE_SPHINX_COMMAND' not in os.environ, reason='Need $ASE_SPHINX_COMMAND for full SPHInX Calculator test')
def test_ssa_sphinx(tmpdir, atoms_E_F_ref):
    (atoms, Eref, Fref) = atoms_E_F_ref
    calc = SSA(SPHInX, eCut=300.0, spinpol=True, constrain_spins=True, kpts=[2, 2, 2],
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
    assert np.isclose(E, Eref, rtol=0.2, atol=1e-5)
    assert np.isclose(F, Fref, rtol=0.2, atol=1e-5).all()

    assert time_F < time_E / 100.0
