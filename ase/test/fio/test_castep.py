import filecmp

from ase.build import bulk
from ase.io import write


def test_write_cell_velocities():
    atoms = bulk("Si")
    atoms.set_velocities([[1, 2, 3], [4, 5, 6]])

    # recognise the keyword
    write("1.cell", atoms, velocities=False)
    write("2.cell", atoms, velocities=True)
    assert not filecmp.cmp("1.cell", "2.cell", shallow=False), "No difference in files"
