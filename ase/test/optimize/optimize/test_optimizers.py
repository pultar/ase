from functools import partial
from typing import Any, Dict, Type
import pathlib

import pytest

from ase import Atoms
from ase.optimize import (
    BFGS,
    FIRE,
    LBFGS,
    Berny,
    BFGSLineSearch,
    GoodOldQuasiNewton,
    GPMin,
    LBFGSLineSearch,
    MDMin,
    ODE12r,
)
from ase.optimize.optimize import Dynamics
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.optimize.sciopt import (
    OptimizerConvergenceError,
    SciPyFminBFGS,
    SciPyFminCG,
)

optclasses = [
    MDMin,
    FIRE,
    LBFGS,
    LBFGSLineSearch,
    BFGSLineSearch,
    BFGS,
    GoodOldQuasiNewton,
    GPMin,
    SciPyFminCG,
    SciPyFminBFGS,
    PreconLBFGS,
    PreconFIRE,
    Berny,
    ODE12r,
    PreconODE12r,
]


@pytest.fixture(name="optcls", scope="module", params=optclasses)
def fixture_optcls(request):
    optcls = request.param
    if optcls is Berny:
        pytest.importorskip("berny")  # check if pyberny installed
        optcls = partial(optcls, dihedral=False)
        optcls.__name__ = Berny.__name__

    return optcls


@pytest.fixture(name="to_catch", scope="module")
def fixture_to_catch(optcls):
    if optcls in (ODE12r, PreconODE12r):
        return (OptimizerConvergenceError,)
    return []


@pytest.fixture(name="kwargs", scope="module")
def fixture_kwargs(optcls):
    kwargs = {}
    if optcls is PreconLBFGS:
        kwargs["precon"] = None
    yield kwargs
    kwargs = {}


@pytest.mark.optimize
@pytest.mark.filterwarnings("ignore: estimate_mu")
def test_optimize(
    optcls: Type[Dynamics],
    rattled_atoms: Atoms,
    reference_atoms: Atoms,
    testdir: pathlib.Path,
    kwargs: Dict[str, Any],
):
    fmax = 0.01
    with optcls(rattled_atoms, logfile=testdir / "opt.log", **kwargs) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged  # check if opt.run() returns True when converged

    forces = rattled_atoms.get_forces()
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    ref_energy = reference_atoms.get_potential_energy()
    e_opt = (
        rattled_atoms.get_potential_energy() * len(reference_atoms) / len(rattled_atoms)
    )
    e_err = abs(e_opt - ref_energy)
    print()
    print(
        "{:>20}: fmax={:.05f} eopt={:.06f}, err={:06e}".format(
            optcls.__name__, final_fmax, e_opt, e_err
        )
    )

    assert final_fmax < fmax
    assert e_err < 1.75e-5  # (This tolerance is arbitrary)
