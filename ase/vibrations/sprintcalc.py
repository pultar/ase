from itertools import product
from typing import Iterator
from ase.atoms import Atoms


def get_displacements_with_identities(atoms: Atoms,
                                      indices: List[int] = None,
                                      delta: float = 0.01,
                                      direction: str = 'central',
                                      ) -> Iterator[Tuple[Atoms, str]]
    """Get displaced atoms with corresponding labels

    Args:
        atoms: reference structure
        indices: Atoms to be displaced. (If None, all atoms are included.)
        delta: Displacement distance.
        direction: 'forward', 'backward' or 'central' differences.

        NB: There is no more "nfree" option. If you would like 4-point central
        differences, just call this function again with a different delta
        value.

    Returns:
        Series of displaced Atoms objects

    """

    if indices is None:
        indices = list(range(len(atoms)))
    directions = (0, 1, 2)

    if direction == 'central':
        signs = (-1, 1)
    elif direction == 'forward':
        signs = (1,)
    elif direction == 'backward':
        signs = (-1,)
    else:
        raise ValueError(f'Direction scheme "{direction}" is not known.')

    for atom_index, cartesian_direction, sign in product(indices,
                                                         directions,
                                                         signs):
        displacement_atoms = atoms.copy()
        displacement_atoms.postions[atom_index,
                                    cartesian_index] += (delta * sign)

        label = f'{atom_index}-{cartesian_index}-{(sign + 1) // 2}'
        
        yield (displacement_atoms, label)

def get_displacements(atoms: Atoms,
                      indices: List[int] = None,
                      delta: float = 0.01,
                      nfree: int = 2) -> Iterator[Atoms]:
    for displacement, _ in get_atoms_with_identities:
        yield displacement

def write_displacements_to_db(atoms: Atoms,
                              indices: List[int] = None,
                              delta: float = 0.01,
                              nfree: int = 2,
                              db: Optional[ase.db.Connection],
                              name: str = 'vib',
                              metadata: Optional[dict]
                              ) -> ase.db.Connection:
    if db is None:
        # Create a db

    for (displacement, label) in get_displacements_with_identities(atoms, **method_kwargs):
        db.write(atoms, label=label, name=name, **metadata)

    return db

def write_displacements(atoms: Atoms,
                        name str = 'vibs':
                        format: str,
                        indices: List[int] = None,
                        delta: float = 0.01,
                        nfree: int = 2
                        ) -> None:
    # Write displaced atom files in your favourite format
    for atoms, label in get_displacements_with_identities:
        atoms.write(Path(directory) / label + ext, format=format)

def calc_displacements(displacements: Sequence[Atoms],
                       calc: Calculator):
    for displacement in displacements:
        displacement.calc = calc
        displacement.get_forces()

    return displacements
    

def read_forces_strict(atoms: Atoms, displacements: Sequence[Atoms],
                       delta: float = 0.01,
                       nfree: int = 2,
                       indices: Sequence[int] = None) -> VibrationsData:
    pass

def read_forces_from_db(db, options) -> VibrationsData:
    # Get rows from db that have forces and match name, metadata
    return read_forces_strict(...)


# Maybe for later? Indices would be very very cool (progressive study)
def read_forces_auto(atoms: Atoms, displacements: Sequence[Atoms],
                     delta: Optional[float],
                     nfree: Optional[int],
                     indices: Optional[int]) -> VibrationsData:
    pass

    
