import numpy as np
from ase import Atoms
from ase.io import read, write

atoms = read('solvated.traj')
qmidx = list(range(13))

# add counterions
l = atoms.cell[0, 0]
r = l / 4.
R = 3 * r
ct_pos = np.array([[r, r, r],
                   [R, R, R],
                   [R, r, r],
                   [r, R, R]])

ctions = Atoms('K4', positions=ct_pos)

atoms = atoms[qmidx] + ctions + atoms[qmidx[-1] + 1:]

ct_idx = list(range(qmidx[-1] + 1, qmidx[-1] + 5))

mask = np.zeros(len(atoms), bool)
mask[qmidx + ct_idx] = True

# delete overlapping waters
radius = 2.5

atoms_to_delete = []
for atm in ct_idx:
    idx = np.where((np.linalg.norm(atoms[atm].position -
                                   atoms[~mask][::3].positions, axis=1)) < radius)
    idx = idx[0] * 3 + len(qmidx) + len(ct_idx) 
    for i in idx:
        for a in range(3):
            atoms_to_delete.append(i + a)

atoms_to_delete = np.unique(atoms_to_delete)
del atoms[atoms_to_delete]

write('neutralized.traj', atoms)
