import pytest

REFERENCE_ENERGY = -28.205659279578953

@pytest.mark.parametrize('define_handler', [ None, 'interactive' ])
def test_turbomole_H2(define_handler):
    from ase import Atoms
    from ase.calculators.turbomole import Turbomole

    atoms = Atoms('H2', positions=[(0, 0, 0), (0, 0, 1.1)])

    # Write all commands for the define command in a string
    define_str = '\n\na coord\n*\nno\nb all sto-3g hondo\n*\neht\n\n\n\n*'

    atoms.calc = Turbomole(define_str=define_str,
                           define_handler=define_handler)

    # Run turbomole
    assert atoms.get_potential_energy() == pytest.approx(REFERENCE_ENERGY)
