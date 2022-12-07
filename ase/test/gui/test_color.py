import numpy as np
import pytest

from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator


@pytest.fixture
def c10():
    atoms = Atoms('C10', magmoms=np.linspace(1, -1, 10))
    atoms.positions[:] = np.linspace(0, 9, 10)[:, None]
    atoms.calc = SinglePointCalculator(atoms, forces=atoms.positions)
    return atoms


def test_color(gui, c10):
    a = c10
    che = np.linspace(100, 110, 10)
    mask = [0] * 10
    mask[5] = 1
    a.set_array('corehole_energies', np.ma.array(che, mask=mask))
    gui.new_atoms(a)
    c = gui.colors_window()
    c.toggle('force')
    c.toggle('magmom')
    activebuttons = [button.active for button in c.radio.buttons]
    assert activebuttons == [1, 0, 1, 0, 0, 1, 1, 1], activebuttons
    c.toggle('corehole_energies')
    c.change_mnmx(101, 120)
