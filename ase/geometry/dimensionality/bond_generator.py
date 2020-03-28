import numpy as np
from ase.neighborlist import NeighborList
from ase.data import covalent_radii


def get_bond_list(atoms, nl, rs):
    num_atoms = len(atoms)
    bonds = []
    for i in range(num_atoms):
        p = atoms.positions[i]
        indices, offsets = nl.get_neighbors(i)

        for j, offset in zip(indices, offsets):
            q = atoms.positions[j] + np.dot(offset, atoms.get_cell())
            d = np.linalg.norm(p - q)
            k = d / (rs[i] + rs[j])
            bonds.append((k, i, j, tuple(offset)))
    return sorted(bonds)


def next_bond(atoms):
    """
    Generates bonds (lazily) one at a time, sorted by k-value (low to high).
    Here, k = d_ij / (r_i + r_j), where d_ij is the bond length and r_i and r_j
    are the covalent radii of atoms i and j.

    Parameters:

    atoms: ASE atoms object

    Returns:
        k:       float   k-value
        i:       float   index of first atom
        j:       float   index of second atom
        offset:  tuple   cell offset of second atom
    """
    kmax = 0
    rs = covalent_radii[atoms.get_atomic_numbers()]
    seen = set()
    while 1:
        # Expand the scope of the neighbor list.
        kmax += 2
        nl = NeighborList(kmax * rs, skin=0, self_interaction=False)
        nl.update(atoms)

        # Get a list of bonds, sorted by k-value.
        bonds = get_bond_list(atoms, nl, rs)

        # Yield the bonds which we have not previously generated.
        for b in bonds:
            if b not in seen:
                seen.add(b)
                yield b
