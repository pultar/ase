import pytest

REFERENCE_ENERGY = -48044.56719492441
REFERENCE_DIPOLE = [-2.03999981, -2.03999981, -2.0399998]

@pytest.mark.parametrize('define_handler', [None, 'interactive'])
def test_turbomole_au13(define_handler):
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.calculators.turbomole import Turbomole

    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [1, 2, 1]
    atoms = FaceCenteredCubic('Au', surfaces, layers, latticeconstant=4.08)

    params = {
        'title': 'Au13-',
        'task': 'energy',
        'basis set name': 'def2-SV(P)',
        'total charge': -1,
        'multiplicity': 1,
        'use dft': True,
        'density functional': 'pbe',
        'use resolution of identity': True,
        'ri memory': 1000,
        'use fermi smearing': True,
        'fermi initial temperature': 500,
        'fermi final temperature': 100,
        'fermi annealing factor': 0.9,
        'fermi homo-lumo gap criterion': 0.09,
        'fermi stopping criterion': 0.002,
        'scf energy convergence': 1.e-4,
        'scf iterations': 250
    }

    calc = Turbomole(**params, define_handler=define_handler)
    atoms.calc = calc
    calc.calculate(atoms)

    # use the get_property() method
    assert calc.get_property('energy') == pytest.approx(REFERENCE_ENERGY)
    assert calc.get_property('dipole') == pytest.approx(REFERENCE_DIPOLE)

    # test restart

    params = {
        'task': 'gradient',
        'scf energy convergence': 1.e-6
    }

    calc = Turbomole(restart=True, **params, define_handler=define_handler)
    assert calc.converged
    calc.calculate()

    assert calc.get_property('energy') == pytest.approx(REFERENCE_ENERGY)
    assert calc.get_property('dipole') == pytest.approx(REFERENCE_DIPOLE)
    # just check that they are present
    print(calc.get_property('forces'))
