from typing import Iterator

from ase.atoms import Atoms


def get_displacements_with_identities(atoms: Atoms,
                                      indices: List[int] = None,
                                      delta: float = 0.01,
                                      nfree: int = 2,
                                      ) -> Iterator[Tuple[Atoms, str]]
    # Actually implement stuff here
    yield (displacement, label)

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
        atoms.write(Path(directory) / label + ext)

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

    
