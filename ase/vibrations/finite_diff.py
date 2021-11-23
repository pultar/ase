from itertools import product
from typing import Iterator, List, Tuple

import numpy as np
from numpy.linalg import lstsq

from ase.atoms import Atoms
from ase.vibrations import VibrationsData
from ase.calculators.calculator import (PropertyNotImplementedError,
                                        PropertyNotPresent)

def get_displacements_with_identities(atoms: Atoms,
                                      indices: List[int] = None,
                                      delta: float = 0.01,
                                      direction: str = 'central',
                                      ) -> List[Tuple[Atoms, str]]:
    """Get displaced atoms with corresponding labels

    Args:
        atoms: reference structure
        indices: Atoms to be displaced. (If None, all atoms are included.)
        delta: Displacement distance.
        direction: 'forward', 'backward' or 'central' differences.

        NB: Compared to the legacy Vibrations object, there is no "nfree"
        option. If you would like 4-point central differences, just call this
        function again with a different delta value.

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

    displacements_labels = []

    for atom_index, cartesian_index, sign in product(indices,
                                                     directions,
                                                     signs):
        displacement_atoms = atoms.copy()
        displacement_atoms.positions[atom_index,
                                    cartesian_index] += (delta * sign)

        label = f'{atom_index}-{cartesian_index}-{(sign + 1) // 2}'
        displacements_labels.append((displacement_atoms, label))

    return displacements_labels



def get_displacements(atoms: Atoms,
                      indices: List[int] = None,
                      direction: str = 'central',
                      use_equilibrium_forces: bool = None,
                      delta: float = 0.01) -> Iterator[Atoms]:
    return [displacement for displacement, _ in
            get_displacements_with_identities(atoms,
                                              indices=indices,
                                              direction=direction,
                                              delta=delta)]


def read_axis_aligned_forces(ref_atoms: Atoms,
                             displacements: Sequence[Atoms],
                             method: str = 'standard',
                             use_equilibrium_forces: bool = None,
                             indices: Sequence[int] = None
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
        ref_atoms:
        use_equilibrium_forces:
            Subtract forces on central atoms from displacement forces. If None,
            detect whether forces are available and use if possible. This is
            only expected to benefit asymmetric displacements (e.g. 'forward').
        direction: If 'forward', 'backward' or 'central' is specified, check
            that displacements have the expected directions.

    """

    arranged_displacements = [[[], [], []] for _ in ref_atoms]
    assert method.lower() in ('standard', 'frederiksen')

    if use_equilibrium_forces:
        if atoms.calc is None:
            raise ValueError("Could not read equilibrium forces, but "
                             "use_equilibrium_forces is True.")
        try:
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

    # We could inspect arranged_displacements to determine if abs(h) is consistent...

    n_atoms = len(ref_atoms)
    hessian = np.empty([n_atoms * 3, n_atoms * 3])
    indices = []
    for atom_index, atom_data in enumerate(arranged_displacements):
        if not any(atom_data):
            continue
        elif not all(atom_data):
            raise ValueError(f"Displacement data is not complete "
                             f"for atom {atom_index}.")
        indices.append(atom_index)

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

    if method.lower() == 'frederiksen':
        for row_index, row in enumerate(hessian):
            hessian[row_index] -= hessian.sum(axis=0)

    return VibrationsData.from_2d(ref_atoms, hessian, indices=indices)
