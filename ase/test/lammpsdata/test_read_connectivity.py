"""Testing lammpsdata reader."""

import pytest
import numpy as np

from io import StringIO
from ase.io import read


CONTENTS = """
3 atoms
1 atom types
2 bonds
2 bond types
1 angles
1 angle types
1 dihedrals
1 dihedral types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 1

Atoms # full

3 1 1 0 2 0 0
1 1 1 0 0 0 0
2 1 1 0 1 0 0

Bonds

1 1 1 2
2 2 2 3

Angles

1 1 1 2 3

Dihedrals

1 1 1 2 3 1
"""

SORTED = {
    True: np.array([0, 1, 2]),
    False: np.array([2, 0, 1]),
}

REFERENCE = {
    'positions': np.array([
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ]),
    'cell': np.eye(3) * 10,
    'bonds': {
        'atoms': np.array([
            [0, 1],
            [1, 2],
        ]),
        'types': np.array([1, 2]),
    },
    'angles': {
        'atoms': np.array([
            [0, 1, 2],
        ]),
        'types': np.array([1]),
    },
    'dihedrals': {
        'atoms': np.array([
            [0, 1, 2, 0],
        ]),
        'types': np.array([1]),
    },
}


@pytest.fixture
def fmt():
    return 'lammps-data'


@pytest.fixture(params=[True, False])
def sort_by_id(request):
    return request.param


@pytest.fixture
def lammpsdata(fmt, sort_by_id):
    fd = StringIO(CONTENTS)
    return read(fd, format=fmt, sort_by_id=sort_by_id), SORTED[sort_by_id]


def parse_dicts(atoms, permutation, label):
    """Parse connectivity strings stored in atoms."""
    all_tuples = np.zeros((0, len(permutation)), int)
    types = np.array([], int)

    dicts = atoms.arrays[label]
    bonded = np.where(dicts != {})[0]

    for i, per_atom in zip(bonded, dicts[bonded]):
        for type_n, per_type in per_atom.items():
            if label == 'bonds':
                per_type = np.array([per_type], int)
            else:
                per_type = np.array(per_type, int)
            new_tuples = [
                np.full(per_type.shape[0], i, int),
                *(per_type.T)
            ]
            new_tuples = np.array(new_tuples)

            all_tuples = np.append(all_tuples,
                                    new_tuples[permutation, :].T,
                                    axis=0)
            types = np.append(types, np.full(per_type.shape[0], type_n))

    return all_tuples, types


def test_positions(lammpsdata):
    atoms, sorted = lammpsdata
    assert pytest.approx(atoms.positions) == REFERENCE['positions'][sorted]


def test_cell(lammpsdata):
    atoms, _ = lammpsdata
    assert pytest.approx(atoms.cell.array) == REFERENCE['cell']


def test_connectivity(lammpsdata):
    atoms, sorted_atoms = lammpsdata

    parser_data = {
        'bonds': (0, 1),
        'angles': (1, 0, 2),
        'dihedrals': (0, 1, 2, 3),
    }

    for label, permutation in parser_data.items():
        tuples, types = parse_dicts(atoms, permutation, label)
        tuples = sorted_atoms[tuples.flatten()].reshape(tuples.shape)
        assert np.all(tuples == REFERENCE[label]['atoms'])
        assert np.all(types == REFERENCE[label]['types'])
