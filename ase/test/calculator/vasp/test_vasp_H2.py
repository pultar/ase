"""Test module using a H2 molecule, for fast calculations"""

import pytest

from ase.build import molecule

calc = pytest.mark.calculator


@pytest.fixture
def atoms():
    return molecule('H2', vacuum=5)


@pytest.fixture
def calc_settings():
    """Some simple fast calculation settings"""
    return dict(xc='lda',
                prec='Low',
                algo='Fast',
                setups='minimal',
                ismear=0,
                nelm=1,
                sigma=1.,
                istart=0,
                lwave=False,
                lcharg=False)


@calc('vasp')
@pytest.mark.parametrize('pbc', [
    3 * [False],
    [True, False, True],
    [False, True, False],
    3 * [True],
])
def test_vasp_pbc(factory, atoms, calc_settings, pbc):
    calc = factory.calc(**calc_settings)

    # Set up, with specific PBC input
    atoms.pbc = pbc
    atoms.calc = calc

    # Run
    atoms.get_potential_energy()

    # We should now have fully PBC atoms object
    assert atoms.pbc.all()