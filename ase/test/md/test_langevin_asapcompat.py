from ase import units
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (maxwell_boltzmann_distribution,
                                         stationary)


def test_langevin_asapcompat():
    """Check that the Langevin object has the attributes that Asap needs."""
    # parameters
    size = 2
    T = 300
    dt = 0.01

    # setup
    atoms = bulk('CuAg', 'rocksalt', a=4.0).repeat(size)
    atoms.pbc = False
    atoms.calc = EMT()

    maxwell_boltzmann_distribution(atoms, temperature=T)
    stationary(atoms)
    with Langevin(atoms, dt * units.fs, temperature=T, friction=0.02) as dyn:
        dyn.run(1)

    for attrib in ('temp', 'fr', 'c1', 'c2', 'c3', 'c4',
                   'c5', 'v', 'rnd_pos', 'rnd_vel'):
        assert hasattr(dyn, attrib), (
            f'Langevin object must have a "{attrib}" attribute or Asap fails.')
