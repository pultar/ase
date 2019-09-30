import numbers
import numpy as np
from ase.utils import basestring
from ase.atoms import Atoms
import ase.spacegroup


def write_crystal(filename, atoms, symmetry=None,
                  refine=False, tolerance=1e-6):
    """Method to write atom structure in crystal format
       (fort.34 format)

    The 'symmetry' option may be used to include symmetry operations. If True,
    the spacegroup is detected using spglib according to 'tolerance' option.
    Alternatives, a spacegroup may be specified (e.g. symmetry=255) or disable
    symmetry with symmetry=False (equivalent to symmetry=1; a P1 spacegoup will
    be established.) The default `None` is equivalent to `True` for systems
    with 3D periodicity, and `False` for others. (Wallpaper and frieze groups
    are not yet supported.)

    The 'refine' option may be used to symmetrise the structure using
    spglib. This will always target the spacegroup detected using 'tolerance'.
    It is strongly recommended to use this when creating a structure from other
    inputs as CRYSTAL only looks a small distance for redundant atoms when
    interpreting the fort.34 file.

    The 'tolerance' option is passed to spglib as a distance threshold (in
    Angstrom) to determine the symmetry of the system.

    """

    if isinstance(atoms, (list, tuple)):
        if len(atoms) > 1:
            raise RuntimeError('Don\'t know how to save more than '
                               'one image to CRYSTAL input')
        else:
            atoms = atoms[0]

    myfile = open(filename, 'w')

    pbc = list(atoms.get_pbc())
    ndim = pbc.count(True)

    if refine:
        atoms = _refine_cell(atoms, tolerance)

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
        if symmetry is None or (isinstance(symmetry, bool) and symmetry):
            spg = ase.spacegroup.get_spacegroup(atoms, symprec=tolerance)
        else:
            spg = ase.spacegroup.Spacegroup(1)

    # We have already asserted that the non-periodic direction are z
    # in 2D case, z and y in the 1D case. These are marked for CRYSTAL by
    # setting the length equal to 500.

    myfile.write('{dimensions:4d} {centring:4d} {crystaltype:4d}\n'.format(
        dimensions=ndim,
        centring=1,
        crystaltype=1))

    for vector, identity, periodic in zip(atoms.cell, np.eye(3), pbc):
        if periodic:
            row = list(vector)
        else:
            row = list(identity * 500.)
        myfile.write('{0:-20.12E} {1:-20.12E} {2:-20.12E}\n'.format(*row))

    # Convert symmetry operations to Cartesian coordinates before writing
    cart_vectors = atoms.cell.T
    inv_cart_vectors = np.linalg.inv(cart_vectors)

    myfile.write('{0:5d}\n'.format(len(spg.get_symop())))
    for rotation, translation in spg.get_symop():
        rotation = cart_vectors.dot(rotation.dot(inv_cart_vectors))
        translation = cart_vectors.dot(translation)

        for row in rotation:
            myfile.write('{0:-20.12E} {1:-20.12E} {2:-20.12E}\n'.format(*row))
        myfile.write(
            '{0:-20.12E} {1:-20.12E} {2:-20.12E}\n'.format(*translation))

    myfile.write('{0:5d}\n'.format(len(atoms)))
    for z, tag, position in zip(atoms.get_atomic_numbers(),
                                atoms.get_tags(),
                                atoms.get_positions()):
        if not isinstance(tag, numbers.Integral):
            raise ValueError("Non-integer tag encountered. Accepted values are"
                             " 100, 200, 300...")
        myfile.write(
            '{0:5d} {1:-17.12f} {2:-17.12f} {3:-17.12f}\n'.format(z + tag,
                                                                  *position))

    if isinstance(filename, basestring):
        myfile.close()


def read_crystal(filename):
    """Method to read coordinates form 'fort.34' files

    Additionally read information about periodic boundary condition.
    """
    with open(filename, 'r') as myfile:
        lines = myfile.readlines()

    atoms_pos = []
    anumber_list = []
    tags = []
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

def _refine_cell(atoms, tolerance):
    try:
        import spglib
    except ImportError:
        raise ImportError('Could not import spglib; required if writing '
                          'CRYSTAL fort.34 file with "refine=True".')

    cell, positions, numbers = spglib.refine_cell(atoms, symprec=tolerance)
    return Atoms(numbers, scaled_positions=positions, cell=cell)
