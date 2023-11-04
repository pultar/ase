from io import StringIO
from typing import IO, Any, Dict, Type

import pytest

from ase import Atoms
from ase.optimize import (
    BFGS,
    FIRE,
    LBFGS,
    BFGSLineSearch,
    GoodOldQuasiNewton,
    GPMin,
    LBFGSLineSearch,
    MDMin,
    ODE12r,
)
from ase.optimize.optimize import Optimizer
from ase.optimize.precon import PreconFIRE, PreconLBFGS, PreconODE12r
from ase.optimize.sciopt import (
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
    ODE12r,
    PreconODE12r,
]


FMAX = 1e-1
ENERGY_TOLERANCE = 1e-5


@pytest.fixture(name="optcls", scope="module", params=optclasses)
def fixture_optcls(request) -> Type[Optimizer]:
    optcls: Type[Optimizer] = request.param
    return optcls


@pytest.fixture(name="kwargs", scope="module")
def fixture_kwargs(optcls):
    kwargs = {}
    if optcls is PreconLBFGS:
        kwargs["precon"] = None
    yield kwargs
    kwargs = {}


@pytest.fixture(name="logfile")
def fixture_logfile() -> IO:
    return StringIO()


@pytest.fixture(name="optimizer", params=optclasses)
def fixture_optimizer(
    request: pytest.FixtureRequest,
    rattled_atoms: Atoms,
    logfile: IO,
    kwargs: Dict[str, Any],
) -> Optimizer:
    optclass: Type[Optimizer] = request.param
    return optclass(atoms=rattled_atoms, logfile=logfile, **kwargs)


@pytest.fixture(name="run_optimizer")
def fixture_run_optimizer(optimizer: Optimizer) -> bool:
    return optimizer.run(fmax=FMAX)


@pytest.fixture(name="final_energy")
def fixture_final_energy(rattled_atoms: Atoms, run_optimizer: bool) -> float:
    final_energy: float = rattled_atoms.get_potential_energy()
    return final_energy


@pytest.fixture(name="final_max_force")
def fixture_final_max_force(rattled_atoms: Atoms, run_optimizer: bool) -> float:
    final_max_force = max((rattled_atoms.get_forces() ** 2).sum(axis=1) ** 0.5)
    return final_max_force


@pytest.fixture(name="log_optimizer_run")
def fixture_log_optimizer_run(
    optimizer: Optimizer,
    reference_energy: float,
    final_max_force: float,
    final_energy: float,
) -> None:
    print()
    e_err = abs(final_energy - reference_energy)
    print(
        "{:>20}: fmax={:.05f} eopt={:.06f}, err={:06e}".format(
            optimizer.__class__.__name__, final_max_force, final_energy, e_err
        )
    )


def test_should_obtain_final_state_with_energy_within_tolerance(
    reference_energy: float, final_energy: float, run_optimizer: bool
) -> None:
    assert abs(final_energy - reference_energy) < ENERGY_TOLERANCE


def test_should_reduce_forces_below_tolerance(
    run_optimizer: bool, final_max_force: float
) -> None:
    assert final_max_force < FMAX
