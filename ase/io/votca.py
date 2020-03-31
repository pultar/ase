"""Interface to call orca For Votca.

API
---
.. autofunction:: write_votca

"""

from ..io import xyz
from ..atoms import Atoms


def write_votca(atoms: Atoms, **params) -> None:
    """Write VOTCA input file(s)."""
    # geometry to votca.xyz
    xyz.write_xyz('votca.xyz', atoms)

    inp = f"! engrad {params['orcasimpleinput']} \n"
    inp += f"{params['orcablocks']} \n"
    inp += f"*xyz {params['charge']} {params['mult']}\n{format_atoms(atoms)}\n*"

    with open(f"{params['label']}.inp", 'w') as f:
        f.write(inp)


def format_atoms(atoms: Atoms) -> str:
    """Print the atoms in xyz format."""
    inp = ""
    for atom in atoms:
        x, y, z = atom.position
        # 71 is ascii G (Ghost)
        ghost = ":" if atom.tag == 71 else " "
        inp += f"{atom.symbol} {ghost} {x} {y} {z}\n"

    return inp
