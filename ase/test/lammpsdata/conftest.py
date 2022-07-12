import pytest
import numpy as np
from ase import Atoms


@pytest.fixture
def lammpsdata_file_path(datadir):
    return datadir / "lammpsdata_input.data"


@pytest.fixture
def atoms():
    # Some constants used to initialize things
    CELL_LENGTH = 102.3776
    NATOMS = 3
    MAX_VEL = 0.1
    MAX_MASS = 32

    atoms_attrs = {
        "cell": [CELL_LENGTH] * 3,
        "positions": np.random.RandomState(17).rand(NATOMS, 3) * CELL_LENGTH,
        "charges": [0.0] * NATOMS,
        "velocities": np.random.RandomState(41).rand(NATOMS, 3) * MAX_VEL,
        "numbers": [1] * NATOMS,
        "masses": np.random.RandomState(23).rand(NATOMS) * MAX_MASS,
        "pbc": [True] * 3,
    }

    return Atoms(**atoms_attrs)
