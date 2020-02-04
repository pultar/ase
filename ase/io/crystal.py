from typing import Dict, List, Union, Tuple
import numbers
import numpy as np
from ase.atoms import Atoms
from ase.cell import Cell
from ase.io.jsonio import write_json
import ase.spacegroup

SymmetryDataset = Dict[str, Union[int, str, np.narray, List[str]]]


def write_crystal(filename: str,
                  atoms: Atoms,
                  symmetry: Union[bool, int] = None,
                  idealize: bool = False,
                  tolerance: float = 1e-6,
                  asymmetric: bool = False,
                  write_transformation: bool = True) -> None:
    """Method to write atom structure in crystal format
       (fort.34 format)

    Parameters
    ----------

    filename: output filename. Typically for CRYSTAL the extension will be .gui
        or the file is named "fort.34".

    atoms: structure

    symmetry:

        The 'symmetry' option may be used to include symmetry operations. If
        True, the spacegroup is detected using spglib according to 'tolerance'
        option.  Alternatives, a spacegroup may be specified
        (e.g. symmetry=255) or disable symmetry with symmetry=False (equivalent
        to symmetry=1; a P1 spacegoup will be established.) The default `None`
        is equivalent to `True` for systems with 3D periodicity, and `False`
        for others. (Wallpaper and frieze groups are not yet supported.)

        If symmetry is applied, the structure written to file will be
        rotated/translated as appropriate to a high-symmetry setting consistent
        with the symmetry operations. See write_transformation to write data
        allowing this operation to be reversed when reading output structures.

    tolerance:

        The 'tolerance' option is passed to spglib as a distance threshold (in
        Angstrom) to determine the symmetry of the system.

    idealize:
        The 'idealize' option may be used to symmetrise the structure using
        spglib. This will always target the spacegroup specified with
        *symmetry* or detected with *tolerance*.  It is strongly recommended to
        use this when creating a structure from other inputs as CRYSTAL only
        looks a small distance for redundant atoms when interpreting the
        fort.34 file.

    asymmetric:

        write only a minimal basis of atoms (the "asymmetric unit cell"), to be
        expanded to the full set of atoms by the symmetry operations. Recent
        versions of CRYSTAL will accept either form. Spglib will be used to a)
        identify the spacegroup b) identify equivalent sites in the unit cell.

    write_transformation:

        If True, the rotation and translation operation to a standard
        high-symmetry cell will be written to the file FILENAME.transform.json
        to facilitate "undoing" the rotation when reading results, obtaining
        an object consistent with the initial Atoms.

    """

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than '
                               'one image to CRYSTAL input')
        else:
            atoms = atoms[0]

    pbc = list(atoms.get_pbc())
    ndim = pbc.count(True)

    if ndim == 0:
        if symmetry is not None and symmetry:
            raise ValueError("ASE cannot currently set symmetry groups for 0D "
                             "cluster in CRYSTAL fort.34 file.")
        else:
            spg = ase.spacegroup.Spacegroup(1)

    elif ndim == 1:
        if symmetry is not None and symmetry:
            raise ValueError("ASE cannot currently set symmetry groups for 1D "
                             "rod in CRYSTAL fort.34 file.")
        elif (pbc == [True, False, False]
              and abs(np.dot(atoms.cell[0], [0, 1, 1])) < tolerance
              and abs(np.dot(atoms.cell[1], [1, 0, 0])) < tolerance
              and abs(np.dot(atoms.cell[2], [1, 0, 0])) < tolerance):
            spg = ase.spacegroup.Spacegroup(1)
        else:
            raise ValueError("If using 1D periodic boundary with CRYSTAL, "
                             "non-periodic directions must be Cartesian y, z "
                             "and vector 'a' must be orthogonal to y, z")

    elif ndim == 2:
        if symmetry is not None and symmetry:
            raise ValueError("ASE cannot currently set symmetry groups for 2D "
                             "slab in CRYSTAL fort.34 file.")
        elif (atoms.get_pbc() == [True, True, False]
              and abs(np.dot(atoms.cell[0], [0, 0, 1])) < tolerance
              and abs(np.dot(atoms.cell[1], [0, 0, 1])) < tolerance
              and abs(np.dot(atoms.cell[2], [1, 1, 0])) < tolerance):
            spg = ase.spacegroup.Spacegroup(1)
        else:
            raise ValueError("If using 2D periodic boundary with CRYSTAL, "
                             "non-periodic direction must be Cartesian z, "
                             "a, b vectors must be orthogonal to c")
    else:
        if isinstance(symmetry, ase.spacegroup.Spacegroup):
            spg = symmetry
        elif symmetry is None or (isinstance(symmetry, bool) and symmetry):
            spg = ase.spacegroup.get_spacegroup(atoms, symprec=tolerance)
        else:
            spg = ase.spacegroup.Spacegroup(1)

    if spg.no > 1:
        atoms, symmetry_data = _get_standard_atoms(
            atoms, spg, write_transformation=write_transformation,
            tolerance=tolerance, asymmetric=asymmetric, idealize=idealize)
        rotations = symmetry_data['rotations']
        translations = symmetry_data['translations']
    else:
        rotations = [np.eye(3)]
        translations = [np.zeros(3)]

    # We have already asserted that the non-periodic direction are z
    # in 2D case, z and y in the 1D case. These are marked for CRYSTAL by
    # setting the length equal to 500.

    with open(filename, 'w') as fd:

        fd.write('{dimensions:4d} {centring:4d} {crystaltype:4d}\n'.format(
            dimensions=ndim,
            centring=1,
            crystaltype=1))

        for vector, identity, periodic in zip(atoms.cell, np.eye(3), pbc):
            if periodic:
                row = list(vector)
            else:
                row = list(identity * 500.)
            fd.write('{0:-20.12E} {1:-20.12E} {2:-20.12E}\n'.format(*row))

        # Convert symmetry operations to Cartesian coordinates before writing
        if atoms.cell.T.any():
            cart_vectors = atoms.cell.T
            inv_cart_vectors = np.linalg.inv(cart_vectors)
        else:
            # use identity matrix if no unit cell
            cart_vectors = inv_cart_vectors = np.eye(3)

        fd.write('{0:5d}\n'.format(len(rotations)))
        for rotation, translation in zip(rotations, translations):
            rotation = cart_vectors.dot(rotation.dot(inv_cart_vectors))
            translation = cart_vectors.dot(translation)

            for row in rotation:
                fd.write('{0:-20.12E} {1:-20.12E} {2:-20.12E}\n'.format(*row))
            fd.write(
                '{0:-20.12E} {1:-20.12E} {2:-20.12E}\n'.format(*translation))

        fd.write('{0:5d}\n'.format(len(atoms)))
        for z, tag, position in zip(atoms.get_atomic_numbers(),
                                    atoms.get_tags(),
                                    atoms.get_positions()):
            if not isinstance(tag, numbers.Integral):
                raise ValueError("Non-integer tag encountered. Accepted values"
                                 " are 100, 200, 300...")
            fd.write('{0:5d} {1:-17.12f} {2:-17.12f} '
                     '{3:-17.12f}\n'.format(z + tag, *position))


def read_crystal(filename):
    """Method to read coordinates form 'fort.34' files

    Additionally read information about periodic boundary condition.
    """
    with open(filename, 'r') as myfile:
        lines = myfile.readlines()

    atoms_pos = []
    anumber_list = []
    tag_list = []
    my_pbc = [False, False, False]
    mycell = []

    if float(lines[4]) != 1:
        raise ValueError('High symmetry geometry is not allowed.')

    if float(lines[1].split()[0]) < 500.0:
        cell = [float(c) for c in lines[1].split()]
        mycell.append(cell)
        my_pbc[0] = True
    else:
        mycell.append([1, 0, 0])

    if float(lines[2].split()[1]) < 500.0:
        cell = [float(c) for c in lines[2].split()]
        mycell.append(cell)
        my_pbc[1] = True
    else:
        mycell.append([0, 1, 0])

    if float(lines[3].split()[2]) < 500.0:
        cell = [float(c) for c in lines[3].split()]
        mycell.append(cell)
        my_pbc[2] = True
    else:
        mycell.append([0, 0, 1])

    natoms = int(lines[9].split()[0])
    for i in range(natoms):
        index = 10 + i
        anum_plus_tag = int(lines[index].split()[0])
        anum = anum_plus_tag % 100
        anumber_list.append(anum)
        tag_list.append(anum_plus_tag - anum)

        position = [float(p) for p in lines[index].split()[1:]]
        atoms_pos.append(position)

    atoms = Atoms(positions=atoms_pos, numbers=anumber_list,
                  tags=tag_list, cell=mycell, pbc=my_pbc)
    return atoms


def _get_standard_atoms(atoms: Atoms,
                        spg: ase.spacegroup.Spacegroup,
                        tolerance: float,
                        filename: str,
                        write_transformation: bool = True,
                        asymmetric: bool = False,
                        idealize: bool = False
                        ) -> Tuple[Atoms, SymmetryDataset]:
    """Set atoms into standard position/orientation for spacegroup

    Optionally write a file with transformation data

    Args:
        atoms: input structure
        spg: spacegroup with desired symmetry operations
        tolerance: distance tolerance used when mapping spacegroup to structure
        filename: name (without symmetry.json extension) for transformation
            data. Conventionally this is equal to the output structure file for
            CRYSTAL, e.g. fort.34
        write_transformation: write the symmetry data to json file so that an
            Atoms object can be recovered in the original orientation.
        asymmetric: Omit equivalent atoms, leaving a minimal asymmetric unit.
        idealize: Snap atoms to high-symmetry positions in standard cell

    Returns:
        transformed copy of atoms

    """
    try:
        import spglib
    except ImportError:
        raise ImportError("The spglib library is required for symmetry "
                          "analysis when writing CRYSTAL input files")

    spglib_cell = (atoms.cell, atoms.get_scaled_positions(), atoms.numbers)
    # spglib prefers Hall number to int tables number; Hall numbers are a
    # larger set that specifies setting more exactly than int table numbers.
    # However, we expect ASE users to be more familiar with int table numbers
    # so this is what users are allowed to input.  Here we ask spglib for a
    # Hall number compatible with the symmetry data from our ASE spacegroup.
    # This might lead to unnecessary changes in orientation/setting, but
    # those can be undone with the other transformation to standard setting
    hall_number = spglib.get_hall_number_from_symmetry(*zip(*spg.get_symop()))
    symmetry_data = spglib.get_symmetry_dataset(spglib_cell,
                                                symprec=tolerance,
                                                hall_number=hall_number)
    if symmetry_data is None:
        raise Exception("Could not obtain symmetry data for spacegroup {}"
                        "(Hall number {}) with tolerance {}.".format(
                            spg.no, hall_number, tolerance))
    if _is_supercell(symmetry_data):
        raise ValueError("Cannot write CRYSTAL symmetry operations for a "
                         "supercell")

    if idealize:
        assert symmetry_data['numbers'] == atoms.numbers
        new_cell = symmetry_data['std_lattice'],
        new_scaled_positions = symmetry_data['std_positions']

    else:
        # From spglib docs: (a_s b_s c_s) = (a b c) P^-1
        # We store cell with lattice vectors as rows, so use transpose to get
        # in and out of column format
        p_matrix = symmetry_data['transformation_matrix']
        new_cell = atoms.cell.T.dot(np.linalg.inv(p_matrix)).T
        # x_s = P_x + p (mod 1)
        new_scaled_positions = ([p_matrix.dot(scaled_position[:, np.newaxis])
                                 for scaled_position
                                 in atoms.get_scaled_positions()]
                                + symmetry_data['origin_shift'])

    new_atoms = atoms.copy()
    new_atoms.cell(new_cell)
    new_atoms.set_scaled_positions(new_scaled_positions)

    if asymmetric:
        new_atoms = new_atoms[list(set(symmetry_data['equivalent_atoms']))]

    if write_transformation:
        transformation = {key: symmetry_data[key] for key in
                          ('transformation_matrix', 'origin_shift')}
        write_json(filename + '.transform.json', transformation)

    return new_atoms


def _is_supercell(symmetry_data, tol=1e-3):
    return Cell(symmetry_data['transformation_matrix']).volume > (1 + tol)
