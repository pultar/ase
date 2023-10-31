from functools import partial

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.cluster import Icosahedron
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


@pytest.fixture(name="ref_atoms", scope="module")
def fixture_ref_atoms(optcls):
    if optcls is Berny:
        ref_atoms = Icosahedron("Ag", 2, 3.82975)
        ref_atoms.calc = EMT()
        return ref_atoms

    ref_atoms = bulk("Au")
    ref_atoms.calc = EMT()
    ref_atoms.get_potential_energy()
    return ref_atoms


@pytest.fixture(name="atoms", scope="module")
def fixture_atoms(ref_atoms, optcls):
    if optcls is Berny:
        atoms = ref_atoms.copy()
        floor = 7
        optcls = partial(optcls, dihedral=False)
        optcls.__name__ = Berny.__name__
    else:
        atoms = ref_atoms * (2, 2, 2)
        floor = 0.45

    atoms.calc = EMT()
    atoms.rattle(stdev=0.1, seed=7)
    e_unopt = atoms.get_potential_energy()
    assert e_unopt > floor
    assert e_unopt > floor
    return atoms


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
def test_optimize(optcls, atoms, ref_atoms, testdir, kwargs):
    fmax = 0.01
    with optcls(atoms, logfile=testdir / "opt.log", **kwargs) as opt:
        is_converged = opt.run(fmax=fmax)
    assert is_converged  # check if opt.run() returns True when converged

    forces = atoms.get_forces()
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    final_fmax = max((forces**2).sum(axis=1) ** 0.5)
    ref_energy = ref_atoms.get_potential_energy()
    e_opt = atoms.get_potential_energy() * len(ref_atoms) / len(atoms)
    e_err = abs(e_opt - ref_energy)
    print()
    print(
        "{:>20}: fmax={:.05f} eopt={:.06f}, err={:06e}".format(
            optcls.__name__, final_fmax, e_opt, e_err
        )
    )

    assert final_fmax < fmax
    assert e_err < 1.75e-5  # (This tolerance is arbitrary)


@pytest.mark.optimize
def test_unconverged(optcls, atoms, kwargs):
    """Test if things work properly when forces are not converged."""
    fmax = 1e-9  # small value to not get converged
    with optcls(atoms, **kwargs) as opt:
        opt.run(fmax=fmax, steps=1)  # only one step to not get converged
    assert not opt.converged()


def test_run_twice(optcls, atoms, kwargs):
    """Test if `steps` increments `max_steps` when `run` is called twice."""
    fmax = 1e-9  # small value to not get converged
    steps = 5
    with optcls(atoms, **kwargs) as opt:
        opt.run(fmax=fmax, steps=steps)
        opt.run(fmax=fmax, steps=steps)
    assert opt.nsteps == 2 * steps
    assert opt.max_steps == 2 * steps


@pytest.mark.optimize
@pytest.mark.parametrize('optcls,max_steps', product(optclasses, range(-1,2)))
def test_should_respect_steps_limit(
    optcls, max_steps, atoms, testdir, kwargs, to_catch
):
    fmax = 0.01
    with optcls(atoms, logfile=testdir / "opt.log", **kwargs) as opt:
        try:
            opt.run(fmax=fmax, steps=max_steps)
        except to_catch:
            pass

    with open(testdir / "opt.log", encoding="utf-8") as file:
        lines = file.readlines()

    steps_ran = int(lines[-1].split()[1].strip("[]"))

    if optcls in (ODE12r, PreconODE12r):
        steps_ran -= 1  # ODE12r and PreconODE12r repeat the first step

    max_steps = max_steps if max_steps >= 0 else 0

    assert steps_ran <= max_steps
