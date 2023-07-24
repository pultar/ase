from itertools import product
from typing import Dict, Iterator, List, Tuple, Union, Sequence
import warnings

import numpy as np
from numpy.linalg import lstsq

from ase.atoms import Atoms
import ase.db
from ase.db.core import Database
from ase.vibrations import VibrationsData
from ase.calculators.calculator import (PropertyNotImplementedError,
                                        PropertyNotPresent)


def _get_displacements_with_identities(atoms: Atoms,
                                       indices: List[int] = None,
                                       delta: float = 0.01,
                                       direction: str = 'central',
                                       ) -> Iterator[Tuple[Atoms, dict]]:
    if indices is None:
        indices = list(range(len(atoms)))
    directions = (0, 1, 2)

    if direction == 'central':
        signs = [-1, 1]
    elif direction == 'forward':
        signs = [1, ]
    elif direction == 'backward':
        signs = [-1, ]
    else:
        raise ValueError(f'Direction scheme "{direction}" is not known.')

    for atom_index, cartesian_index, sign in product(indices,
                                                     directions,
                                                     signs):
        displacement_atoms = atoms.copy()
        displacement_atoms.positions[atom_index,
                                     cartesian_index] += (delta * sign)

        identity = {'atom_index': atom_index,
                    'cartesian_axis': cartesian_index,
                    'sign': sign,
                    'delta': delta}

        yield(displacement_atoms, identity)


def get_displacements_with_identities(atoms: Atoms,
                                      indices: List[int] = None,
                                      delta: float = 0.01,
                                      direction: str = 'central',
                                      ) -> List[Tuple[Atoms, str]]:
    """Get displaced atoms with corresponding labels

    Args:
        atoms: reference structure
        indices: Atoms to be displaced. (If None, all atoms without relevant
            constraints are included.)
        delta: Displacement distance.
        direction: 'forward', 'backward' or 'central' differences.

        NB: Compared to the legacy Vibrations object, there is no "nfree"
        option. If you would like 4-point central differences, just call this
        function again with a different delta value.

    Returns:
        Series of pairs: displaced Atoms object with an identifying string

    """
    if indices is None:
        indices = VibrationsData.indices_from_constraints(atoms)

    displacements_labels = []
    for displacement, identity in _get_displacements_with_identities(
            atoms, indices=indices, direction=direction, delta=delta):

        identity['direction_index'] = (identity['sign'] + 1) // 2
        label = ('{atom_index}-{cartesian_axis}-{direction_index}'
                 .format(**identity))
        displacements_labels.append((displacement, label))

    return displacements_labels


def get_displacements(atoms: Atoms,
                      indices: List[int] = None,
                      direction: str = 'central',
                      delta: float = 0.01) -> List[Atoms]:
    """Get displaced atoms with corresponding labels

    Args:
        atoms: reference structure
        indices: Atoms to be displaced. (If None, all atoms without relevant
            constraints are included.)
        delta: Displacement distance.
        direction: 'forward', 'backward' or 'central' differences.

        NB: Compared to the legacy Vibrations object, there is no "nfree"
        option. If you would like 4-point central differences, just call this
        function again with a different delta value.

    Returns:
        Series of displaced Atoms object

    """

    return [displacement for displacement, _ in
            get_displacements_with_identities(atoms,
                                              indices=indices,
                                              direction=direction,
                                              delta=delta)]


def write_displacements_to_db(atoms: Atoms,
                              indices: List[int] = None,
                              direction: str = 'central',
                              delta: float = 0.01,
                              db: Union[Database, str] = 'displacements.db',
                              metadata: dict = None
                              ) -> Database:
    """Get displaced atoms as an ASE database

    This may be convenient for sharing or task-farming displacement
    calculations.

    Args:
        atoms: reference structure
        indices: Atoms to be displaced. (If None, all atoms without relevant
            constraints are included.)
        direction: 'forward', 'backward' or 'central' differences.
        delta: Displacement distance.
        db: An active ASE database, or string for a new database file. If the
            database already exists, displacements will be appended.
        metadata:  Additional keys/values to include with the database entries.
            For example, this might be the name of the conformation in a
            database containing several species to be calculated in a separate
            step.

    Returns:
        ASE Database object including displaced structures for vibrations
        calculation. These require forces to be calculated before further
        analysis.

    """

    metadata = {} if metadata is None else metadata

    if isinstance(db, str):
        db = ase.db.connect(db, append=True)
        assert isinstance(db, Database)

    for (displacement, label) in get_displacements_with_identities(
            atoms, indices=indices, direction=direction, delta=delta):
        db.write(displacement, label=label, **metadata)

    return db


def read_axis_aligned_db(db: Union[Database, str] = 'displacements.db',
                         ref_atoms: Atoms = None,
                         metadata: Dict[str, str] = None,
                         **kwargs) -> VibrationsData:
    """Read axis-aligned displacements from ASE DB and get VibrationsData

    This is intended to be used with a set of displacements from
    ``write_displacements_to_db`` or ``get_displacements``, but it is possible
    to combine multiple sets of displacements in order to perform e.g. 4-point
    finite differences.

    Args:
        db: ASE database containing displaced structures with computed forces.
            (Entries without forces may be present and will be ignored.)
        ref_atoms: reference structure for displacements. If not provided, a
            reference geometry will be inferred from the displaced structures.
        metadata: Dict of DB keys/values identifying rows to consider. This may
            be used to e.g. store multiple molecules in the same database as
            part of a high-throughput workflow.

        kwargs: Remaining arguments will be passed to read_axis_aligned_forces.

    Returns:
        VibrationsData

    """
    if metadata is None:
        metadata = {}

    with ase.db.connect(db) as db_connection:
        displacement_rows = db_connection.select('fmax>0', **metadata)
        displacements = [row.toatoms() for row in displacement_rows]

    return read_axis_aligned_forces(displacements,
                                    ref_atoms=ref_atoms,
                                    **kwargs)


def read_axis_aligned_forces(displacements: Sequence[Atoms],
                             ref_atoms: Atoms = None,
                             use_equilibrium_forces: bool = None,
                             indices: Sequence[int] = None,
                             direction: str = None,
                             threshold: float = 1e-12) -> VibrationsData:
    """Convert a set of atoms objects with displacements to VibrationsData

    Displacements are relative to the reference structure, and should consist
    of a change in the position of one atom along one Cartesian direction.
    Any number of displacements can be used for each included degree of freedom
    but a warning will be provided if this is not always consistent.

    The displacements should have forces available (i.e.
    "displacement.get_forces()" should return an array.) If forces are also
    available on the equilibrium atoms, they may be subtracted from
    displacement forces to compensate for imperfect geometry optimisation (see
    "use_equilibrium_forces").


    Args:
        displacements:
            Structures with axis-aligned displacements and available forces
        ref_atoms: reference structure for displacements. If not provided, a
            reference geometry will be inferred from the displaced structures.
        use_equilibrium_forces:
            Subtract forces on central atoms from displacement forces. If None,
            detect whether forces are available and use if possible. This is
            only expected to benefit asymmetric displacements (e.g. 'forward').
        indices:
            Indices of atoms that should be included in harmonic vibrations.
            If None, all atoms with available displacements will be used.
            If no displacement data is available for an atom in Indices, an
            error will be raised. If an item from ``displacements`` would
            be used to calculate the Hessian row for an atom that is missing
            from ``indices``, it is skipped. Indices will be sorted in the
            output data; this feature cannot be used to reorder the rows of the
            Hessian relative to the atom order.
        direction: If 'forward', 'backward' or 'central' is specified, check
            that displacements have the expected directions.
        threshold: Difference tolerated between ref_atoms positions and
            positions in displaced structure, when identifying which atom has
            been moved. A higher threshold may be needed to accommodate
            structure files that were written to limited precision.

    """
    if ref_atoms is None:
        ref_atoms = guess_ref_atoms(displacements)

    # Create a container: List[List[Dict[str, Union[float, np.ndarray]]]]
    # Mypy doesn't handle this well: https://github.com/python/mypy/issues/6463
    arranged_displacements = [[[], [], []] for _ in ref_atoms]  # type: ignore

    if use_equilibrium_forces:
        try:
            if ref_atoms.calc is None:
                raise PropertyNotPresent()
            eq_forces = ref_atoms.get_forces()
        except (PropertyNotImplementedError, PropertyNotPresent):
            raise ValueError("Could not read equilibrium forces, but "
                             "use_equilibrium_forces is True.")

    elif (use_equilibrium_forces is None) and (ref_atoms.calc is not None):
        try:
            eq_forces = ref_atoms.get_forces()
        except (PropertyNotImplementedError, PropertyNotPresent):
            eq_forces = None
    else:
        eq_forces = None

    for displacement in displacements:
        delta_position = displacement.positions - ref_atoms.positions
        disp_index = np.argwhere(np.abs(delta_position) > threshold)

        # argwhere can return multiple results, there should only be one
        if disp_index.shape[0] != 1:
            raise ValueError("Could not assign all structures to x,y,z "
                             "displacements")

        atom_index, cartesian_direction = disp_index[0]

        h = delta_position[atom_index, cartesian_direction]

        forces = displacement.get_forces()
        if eq_forces is not None:
            forces -= eq_forces

        arranged_displacements[atom_index][cartesian_direction].append(
            {'h': h, 'forces': forces})

    n_atoms = len(ref_atoms)
    hessian = np.empty([n_atoms * 3, n_atoms * 3])

    displaced_atoms = set(i for i, row in enumerate(arranged_displacements)
                          if all(row))

    if indices is None:
        if displaced_atoms != set(range(len(ref_atoms))):
            warnings.warn('Displacements not found for all atoms in reference '
                          'structure. As expected indices were not provided, '
                          'assume that this is intentional and create '
                          'VibrationsData with some frozen atoms.')
            indices = sorted(displaced_atoms)
        else:
            indices = list(range(len(ref_atoms)))

    elif not displaced_atoms.issuperset(indices):
        raise ValueError(f"Atoms {set(indices) - displaced_atoms} were "
                         "included in requested indices but corresponding "
                         "displacements were not found.")
    else:
        unused_displacements = (displaced_atoms - set(indices))
        if unused_displacements:
            warnings.warn("Found some displacements outside of requested "
                          f"indices: {unused_displacements}")
        indices = sorted(indices)

    for atom_index in indices:
        atom_data = arranged_displacements[atom_index]
        if not any(atom_data):
            continue
        elif not all(atom_data):
            raise ValueError(f"Displacement data is not complete "
                             f"for atom {atom_index}.")

        for cartesian_index, disp_data in enumerate(atom_data):
            f_array = np.stack([disp['forces'].flatten()
                                for disp in disp_data])
            disp_array = np.array([disp['h'] for disp in disp_data])

            # lstsq solves Ax = b for x
            # To fit the Hessian elements for this row (k)
            # we solve F = -k disp
            hessian_element = lstsq(-disp_array[:, np.newaxis], f_array,
                                    rcond=None)[0]

            hessian[atom_index * 3 + cartesian_index] = hessian_element

    # Convert Hessian to 4-D format for easier indexing
    hessian = hessian.reshape(n_atoms, 3, n_atoms, 3)

    # Cut down to active atom indices
    hessian = hessian.take(indices, axis=0).take(indices, axis=2)

    return VibrationsData(ref_atoms, hessian, indices=indices)


def guess_ref_atoms(displacements: Sequence[Atoms]) -> Atoms:
    """Obtain the presumed reference structure from axis-aligned displacements

    Returns an Atoms object with the modal (i.e. most common) position of each
    atom in the displacement set along each axis.

    This could get memory-intensive for a very large set of displacements.

    """

    from scipy.stats import mode
    import scipy.version

    # Scipy 1.9 raises deprecation warnings if (newly-introduced) keepdims
    # parameter is not set.
    scipy_major, scipy_minor, _ = scipy.version.short_version.split('.')
    if scipy_major == '1' and int(scipy_minor) < 9:
        mode_kwargs = {}
    else:
        mode_kwargs = {'keepdims': True}

    all_positions = np.stack([displacement.positions
                              for displacement in displacements])

    atoms = displacements[0].copy()
    atoms.positions, counts = mode(all_positions, axis=0, **mode_kwargs)

    # Check a mode was found for each degree of freedom
    if counts.min() == 1:
        raise ValueError(
            "Not all positions could be determined; repeated value not found.")

    return atoms
