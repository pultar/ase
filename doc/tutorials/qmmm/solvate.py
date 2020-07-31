import numpy as np
from sys import argv
from ase.io import read, write
from ase.visualize import view
from ase.geometry.geometry import wrap_positions as wrap

qmfile = argv[1]  # Name of QM structure file
mmfile = argv[2]  # Name of MM structure file
apm = int(argv[3])  # Number of atoms per solvent molecule
radius = float(argv[4])  # Cutout radius around each solute atom
enforce_wrap=bool(argv[5])  # (1,0): Wrap MM regardless of previous MM PBC choices

def molwrap(atoms, n, idx=0):
    ''' Wrap to cell without breaking molecules
        atoms: Atoms object
        n:     Number of atoms per solvent molecule
        idx:   Which atom in the solvent molecule
               to determine molecular distances from
               e.g. for OHHOHH... idx=0.  '''

    center = atoms.cell.diagonal() / 2
    positions = atoms.positions.reshape((-1, n, 3))
    distances = positions[:, idx] - center
    old_distances = distances.copy()
    distances = wrap(distances, atoms.cell, atoms.pbc, center=(0, 0, 0))
    offsets = distances - old_distances
    positions += offsets[:, None]
    atoms.set_positions(positions.reshape((-1, 3)))
    return atoms

# Read in MM solvent and wrap:
mm = read(mmfile)

# Check PBCs and box dimensions of MM box
assert not (mm.cell.diagonal() == 0).any(),\
        'mm atoms have no cell'
assert (mm.cell == np.diag(mm.cell.diagonal())).all(),\
        'mm cell not orthorhombic'
if not mm.pbc.all() and enforce_wrap:
    print('Warning: All MM PBCs not on,',
          'be sure you can wrap this box.')
if enforce_wrap:
    mm.pbc = True
mm = molwrap(mm, apm)

qm = read(qmfile)
qm.set_cell(mm.cell)
qm.center()

atoms = qm + mm
atoms.pbc = True

# cut away molecules within radius
qmidx = range(len(qm))
mask = np.zeros(len(atoms), bool)
mask[qmidx] = True

atoms_to_delete = []
for atm in qmidx:
    # Assumes solvent is ordered as mol1,mol2,mol3, ...
    # and starts from the first atom in each mol
    mol_dists = atoms[atm].position - atoms[~mask][::apm].positions
    idx = np.where((np.linalg.norm(mol_dists, axis=1)) < radius)
    idx = idx[0] * apm + len(qmidx)
    for i in idx:
        for a in range(apm):
            atoms_to_delete.append(i + a)

atoms_to_delete = np.unique(atoms_to_delete)
del atoms[atoms_to_delete]

view(atoms)
write('solvated.traj', atoms)
