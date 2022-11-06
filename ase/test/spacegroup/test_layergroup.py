import pytest
import numpy as np
from ase.spacegroup import get_layergroup, get_spacegroup
import ase
from itertools import permutations

def systems():
    yield 78, ase.build.mx2(formula='MoS2', kind='2H', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=10.0) # https://doi.org/10.1016/j.micron.2021.103071
    yield 72, ase.build.mx2(formula='MoS2', kind='1T', a=3.18, thickness=3.19, size=(1, 1, 1), vacuum=10.0)

def variations():
    for system in systems():
        ref_no, atoms = system
        for p in permutations([0,1,2]):
            for rot in range(10):
                p = list(p)
                q, r = np.linalg.qr(np.random.rand(3,3))
                varatoms = atoms.copy()
                spos = atoms.get_scaled_positions()
                varatoms.set_scaled_positions(spos[:, p])
                varatoms.set_cell(atoms.get_cell()[p, :] @ q, scale_atoms=True)
                varatoms.set_pbc(atoms.pbc[p])
                yield ref_no, varatoms


def test_get_layergroup():
    for system in variations():
        ref_no, atoms = system
        lg = get_layergroup(atoms)
        print(ref_no, lg['number'])
        assert ref_no == lg['number']
