import re
import numpy as np

from ase.atoms import Atoms
from ase.calculators.lammps import Prism, convert
from ase.utils import reader, writer

ASTYLE_ATOMIC = 'atomic'
ASTYLE_CHARGE = 'charge'
ASTYLE_BOND = 'bond'
ASTYLE_ANGLE = 'angle'
ASTYLE_MOLECULAR = 'molecular'
ASTYLE_FULL = 'full'


def _store_bonds(bonds_in, ind_of_id, N):
    """Store the bonds as read in an array to be added to an Atoms object.

    The entry for atom i in the array is a dict with the bond type k as key
    and a list of all atoms bonded to it with bond type k.

    So the following entry in a LAMMPS datafile Bonds section:

        1 3 17 29

    Will result in:

        bonds[16] = {3: [28]}
        [...]
        bonds[28] = {}

    Caveat: Only bonds explicitly defined in the datafile are added and indexed
    to the first atom of the pair. This means bonds[j] is *not* an exhaustive
    list of atoms bonded to atom j.
    """
    bonds = [{} for _ in range(N)]
    for bond_type, a1, a2 in bonds_in:
        ind_a1 = ind_of_id[a1]
        ind_a2 = ind_of_id[a2]
        if bond_type in bonds[ind_a1]:
            bonds[ind_a1][bond_type].append(ind_a2)
        else:
            bonds[ind_a1][bond_type] = [ind_a2]
    return np.array(bonds)


def _store_angles(angles_in, ind_of_id, N):
    """Store the angles as read in an array to be added to an Atoms object.

    The entry for atom j in the array is a dict with the angle type k as key
    and a list of all atom pairs which form an angle of type k centered on it.

    So the following entry in a LAMMPS datafile Angles section:

        2 2 17 29 430

    Will result in:

        angles[28] = {2: [(17,430)]}
    """
    angles = [{} for _ in range(N)]
    for angle_type, a1, a2, a3 in angles_in:
        ind_a1 = ind_of_id[a1]
        ind_a2 = ind_of_id[a2]
        ind_a3 = ind_of_id[a3]
        if angle_type in angles[ind_a2]:
            angles[ind_a2][angle_type].append((ind_a1, ind_a3))
        else:
            angles[ind_a2][angle_type] = [(ind_a1, ind_a3)]
    return np.array(angles)


def _store_dihedrals(dih_in, ind_of_id, N):
    """Store the dihedrals as read in an array to be added to an Atoms object.

    The entry for atom j in the array is a dict with the dihedrals type k as
    key and a list of all atom trios which form a dihedral of type k, with atom
    j as second atom.

    So the following entry in a LAMMPS datafile Dihedrals section:

        12 4 17 29 30 21

    Will result in:

        dihedrals[28] = {4: [(16,29,20)]}

    The entry for the i-j-k-l dihedral is stored in dihedrals[j] as (i, k, l).
    """
    dihedrals = [{} for _ in range(N)]
    for dih_type, a1, a2, a3, a4 in dih_in:
        ind_a1 = ind_of_id[a1]
        ind_a2 = ind_of_id[a2]
        ind_a3 = ind_of_id[a3]
        ind_a4 = ind_of_id[a4]
        if dih_type in dihedrals[ind_a2]:
            dihedrals[ind_a2][dih_type].append((ind_a1, ind_a3, ind_a4))
        else:
            dihedrals[ind_a2][dih_type] = [(ind_a1, ind_a3, ind_a4)]
    return np.array(dihedrals)


def _store_impropers(imp_in, ind_of_id, N):
    """Store the impropers as read in an array to be added to an Atoms object.

    The entry for atom i in the array is a dict with the impropers type k as
    key and a list of all atom trios which form a improper of type k, centered
    on atom i

    So the following entry in a LAMMPS datafile Dihedrals section:

        12 3 17 29 13 100

    Will result in:

        impropers[16] = {3: [(28,12,99)]}

    The entry for the i-j-k-l improper is stored in impropers[i] as (j, k, l),
    since usually (except for improper_style class2) atom i is the central
    atom and its deviation from the j-k-l plane is the improper angle.
    """
    impropers = [{} for _ in range(N)]
    for imp_type, a1, a2, a3, a4 in imp_in:
        ind_a1 = ind_of_id[a1]
        ind_a2 = ind_of_id[a2]
        ind_a3 = ind_of_id[a3]
        ind_a4 = ind_of_id[a4]
        if imp_type in impropers[ind_a1]:
            impropers[ind_a1][imp_type].append((ind_a2, ind_a3, ind_a4))
        else:
            impropers[ind_a1][imp_type] = [(ind_a2, ind_a3, ind_a4)]
    return np.array(impropers)


@reader
def read_lammps_data(fileobj, Z_of_type=None, atom_style=ASTYLE_FULL,
                     sort_by_id=False, units="metal"):
    """Method which reads a LAMMPS data file.

    sort_by_id: Order the particles according to their id. Might be faster to
    switch it off.
    Units are set by default to the units=metal setting in LAMMPS.
    """
    # load everything into memory
    lines = fileobj.readlines()

    # begin read_lammps_data
    comment = None
    N = None
    # N_types = None
    xlo = None
    xhi = None
    ylo = None
    yhi = None
    zlo = None
    zhi = None
    xy = None
    xz = None
    yz = None
    pos_in = {}
    travel_in = {}
    mol_id_in = {}
    charge_in = {}
    mass_in = {}
    vel_in = {}
    bonds_in = []
    angles_in = []
    dihedrals_in = []
    impropers_in = []
    bond_types = 0
    angle_types = 0
    dihedral_types = 0
    improper_types = 0
    coeffs = {}

    sections = [
        "Atoms",
        "Velocities",
        "Masses",
        "Charges",
        "Ellipsoids",
        "Lines",
        "Triangles",
        "Bodies",
        "Bonds",
        "Angles",
        "Dihedrals",
        "Impropers",
        "Impropers Pair Coeffs",
        "PairIJ Coeffs",
        "Pair Coeffs",
        "Bond Coeffs",
        "Angle Coeffs",
        "Dihedral Coeffs",
        "Improper Coeffs",
        "BondBond Coeffs",
        "BondAngle Coeffs",
        "MiddleBondTorsion Coeffs",
        "EndBondTorsion Coeffs",
        "AngleTorsion Coeffs",
        "AngleAngleTorsion Coeffs",
        "BondBond13 Coeffs",
        "AngleAngle Coeffs",
    ]
    header_fields = [
        "atoms",
        "bonds",
        "angles",
        "dihedrals",
        "impropers",
        "atom types",
        "bond types",
        "angle types",
        "dihedral types",
        "improper types",
        "extra bond per atom",
        "extra angle per atom",
        "extra dihedral per atom",
        "extra improper per atom",
        "extra special per atom",
        "ellipsoids",
        "lines",
        "triangles",
        "bodies",
        "xlo xhi",
        "ylo yhi",
        "zlo zhi",
        "xy xz yz",
    ]
    sections_re = "(" + "|".join(sections).replace(" ", "\\s+") + ")"
    header_fields_re = "(" + "|".join(header_fields).replace(" ", "\\s+") + ")"

    section = None
    header = True
    for line in lines:
        if comment is None:
            comment = line.rstrip()
        else:
            line = re.sub("#.*", "", line).rstrip().lstrip()
            if re.match("^\\s*$", line):  # skip blank lines
                continue

        # check for known section names
        m = re.match(sections_re, line)
        if m is not None:
            section = m.group(0).rstrip().lstrip()
            header = False
            continue

        if header:
            field = None
            val = None
            # m = re.match(header_fields_re+"\s+=\s*(.*)", line)
            # if m is not None: # got a header line
            #   field=m.group(1).lstrip().rstrip()
            #   val=m.group(2).lstrip().rstrip()
            # else: # try other format
            #   m = re.match("(.*)\s+"+header_fields_re, line)
            #   if m is not None:
            #       field = m.group(2).lstrip().rstrip()
            #       val = m.group(1).lstrip().rstrip()
            m = re.match("(.*)\\s+" + header_fields_re, line)
            if m is not None:
                field = m.group(2).lstrip().rstrip()
                val = m.group(1).lstrip().rstrip()
            if field is not None and val is not None:
                if field == "atoms":
                    N = int(val)
                # elif field == "atom types":
                #     N_types = int(val)

                elif field == 'bond types':
                    bond_types = int(val)
                elif field == 'angle types':
                    angle_types = int(val)
                elif field == 'dihedral types':
                    dihedral_types = int(val)
                elif field == 'improper types':
                    improper_types = int(val)

                elif field == "xlo xhi":
                    (xlo, xhi) = [float(x) for x in val.split()]
                elif field == "ylo yhi":
                    (ylo, yhi) = [float(x) for x in val.split()]
                elif field == "zlo zhi":
                    (zlo, zhi) = [float(x) for x in val.split()]
                elif field == "xy xz yz":
                    (xy, xz, yz) = [float(x) for x in val.split()]

        if section is not None:
            fields = line.split()
            if section == "Atoms":  # id *
                atom_id = int(fields[0])
                if atom_style == ASTYLE_FULL and (
                        len(fields) == 7 or len(fields) == 10):
                    # id mol-id type q x y z [tx ty tz]
                    pos_in[atom_id] = (
                        int(fields[2]),
                        float(fields[4]),
                        float(fields[5]),
                        float(fields[6]),
                    )
                    mol_id_in[atom_id] = int(fields[1])
                    charge_in[atom_id] = float(fields[3])
                    if len(fields) == 10:
                        travel_in[atom_id] = (
                            int(fields[7]),
                            int(fields[8]),
                            int(fields[9]),
                        )
                elif atom_style == ASTYLE_ATOMIC and (
                        len(fields) == 5 or len(fields) == 8
                ):
                    # id type x y z [tx ty tz]
                    pos_in[atom_id] = (
                        int(fields[1]),
                        float(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                    )
                    if len(fields) == 8:
                        travel_in[atom_id] = (
                            int(fields[5]),
                            int(fields[6]),
                            int(fields[7]),
                        )
                elif (atom_style in (ASTYLE_ANGLE, ASTYLE_BOND,
                                     ASTYLE_MOLECULAR)
                      and (len(fields) == 6 or len(fields) == 9)):
                    # id mol-id type x y z [tx ty tz]
                    pos_in[atom_id] = (
                        int(fields[2]),
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                    )
                    mol_id_in[atom_id] = int(fields[1])
                    if len(fields) == 9:
                        travel_in[atom_id] = (
                            int(fields[6]),
                            int(fields[7]),
                            int(fields[8]),
                        )
                elif (atom_style == ASTYLE_CHARGE
                      and (len(fields) == 6 or len(fields) == 9)):
                    # id type q x y z [tx ty tz]
                    pos_in[atom_id] = (
                        int(fields[1]),
                        float(fields[3]),
                        float(fields[4]),
                        float(fields[5]),
                    )
                    charge_in[atom_id] = float(fields[2])
                    if len(fields) == 9:
                        travel_in[atom_id] = (
                            int(fields[6]),
                            int(fields[7]),
                            int(fields[8]),
                        )
                else:
                    raise RuntimeError(
                        "Style '{}' not supported or invalid "
                        "number of fields {}"
                        "".format(atom_style, len(fields))
                    )
            elif section == "Velocities":  # id vx vy vz
                vel_in[int(fields[0])] = (
                    float(fields[1]),
                    float(fields[2]),
                    float(fields[3]),
                )
            elif section == "Masses":
                mass_in[int(fields[0])] = float(fields[1])
            elif section == "Bonds":  # id type atom1 atom2
                bonds_in.append(
                    (int(fields[1]), int(fields[2]), int(fields[3]))
                )
            elif section == "Angles":  # id type atom1 atom2 atom3
                angles_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                    )
                )
            elif section == "Dihedrals":  # id type atom1 atom2 atom3 atom4
                dihedrals_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                        int(fields[5]),
                    )
                )
            elif section == "Impropers":  # id type atom1 atom2 atom3 atom4
                impropers_in.append(
                    (
                        int(fields[1]),
                        int(fields[2]),
                        int(fields[3]),
                        int(fields[4]),
                        int(fields[5]),
                    )
                )
            elif section == "Pair Coeffs":
                if 'Pair' in coeffs:
                    coeffs['Pair'].append(fields)
                else:
                    coeffs['Pair'] = [fields]
                    # All Coeffs are stored as strings because lengths and
                    # datatypes depend on the corresponding style and asterisks
                    # and ranges are valid inputs.
            elif section == "Bond Coeffs":
                if 'Bond' in coeffs:
                    coeffs['Bond'].append(fields)
                else:
                    coeffs['Bond'] = [fields]
            elif section == "Angle Coeffs":
                if 'Angle' in coeffs:
                    coeffs['Angle'].append(fields)
                else:
                    coeffs['Angle'] = [fields]
            elif section == "Dihedral Coeffs":
                if 'Dihedral' in coeffs:
                    coeffs['Dihedral'].append(fields)
                else:
                    coeffs['Dihedral'] = [fields]
            elif section == "Improper Coeffs":
                if 'Improper' in coeffs:
                    coeffs['Improper'].append(fields)
                else:
                    coeffs['Improper'] = [fields]

    # set cell
    cell = np.zeros((3, 3))
    cell[0, 0] = xhi - xlo
    cell[1, 1] = yhi - ylo
    cell[2, 2] = zhi - zlo
    if xy is not None:
        cell[1, 0] = xy
    if xz is not None:
        cell[2, 0] = xz
    if yz is not None:
        cell[2, 1] = yz

    # initialize arrays for per-atom quantities
    positions = np.zeros((N, 3))
    numbers = np.zeros((N), int)
    ids = np.zeros((N), int)
    types = np.zeros((N), int)
    if len(vel_in) > 0:
        velocities = np.zeros((N, 3))
    else:
        velocities = None
    if len(mass_in) > 0:
        masses = np.zeros((N))
    else:
        masses = None
    if len(mol_id_in) > 0:
        mol_id = np.zeros((N), int)
    else:
        mol_id = None
    if len(charge_in) > 0:
        charge = np.zeros((N), float)
    else:
        charge = None
    if len(travel_in) > 0:
        travel = np.zeros((N, 3), int)
    else:
        travel = None

    ind_of_id = {}
    # copy per-atom quantities from read-in values
    for (i, atom_id) in enumerate(pos_in.keys()):
        # by id
        ind_of_id[atom_id] = i
        if sort_by_id:
            ind = atom_id - 1
        else:
            ind = i
        atom_type = pos_in[atom_id][0]
        positions[ind, :] = [pos_in[atom_id][1],
                             pos_in[atom_id][2],
                             pos_in[atom_id][3]]
        if velocities is not None:
            velocities[ind, :] = [vel_in[atom_id][0],
                                  vel_in[atom_id][1],
                                  vel_in[atom_id][2]]
        if travel is not None:
            travel[ind] = travel_in[atom_id]
        if mol_id is not None:
            mol_id[ind] = mol_id_in[atom_id]
        if charge is not None:
            charge[ind] = charge_in[atom_id]
        ids[ind] = atom_id
        # by type
        types[ind] = atom_type
        if Z_of_type is None:
            numbers[ind] = atom_type
        else:
            numbers[ind] = Z_of_type[atom_type]
        if masses is not None:
            masses[ind] = mass_in[atom_type]
    # convert units
    positions = convert(positions, "distance", units, "ASE")
    cell = convert(cell, "distance", units, "ASE")
    if masses is not None:
        masses = convert(masses, "mass", units, "ASE")
    if velocities is not None:
        velocities = convert(velocities, "velocity", units, "ASE")

    # create ase.Atoms
    at = Atoms(
        positions=positions,
        numbers=numbers,
        masses=masses,
        cell=cell,
        pbc=[True, True, True],
    )
    # set velocities (can't do it via constructor)
    if velocities is not None:
        at.set_velocities(velocities)
    at.arrays["id"] = ids
    at.arrays["type"] = types
    if travel is not None:
        at.arrays["travel"] = travel
    if mol_id is not None:
        at.arrays["mol-id"] = mol_id
    if charge is not None:
        at.arrays["initial_charges"] = charge
        at.arrays["mmcharges"] = charge.copy()

    if (bond_types + angle_types + dihedral_types + improper_types != 0
            and atom_style in (ASTYLE_BOND, ASTYLE_ANGLE, ASTYLE_MOLECULAR,
                               ASTYLE_FULL)):
        at.info['types'] = {}
        at.info['coeffs'] = coeffs

        if len(bonds_in) > 0:
            at.new_array('bonds', _store_bonds(bonds_in, ind_of_id, N))
            at.info['types']['bond'] = bond_types
        if atom_style != ASTYLE_BOND:
            if len(angles_in) > 0:
                at.new_array('angles', _store_angles(angles_in, ind_of_id, N))
                at.info['types']['angle'] = angle_types
            if atom_style != ASTYLE_ANGLE:
                if len(dihedrals_in) > 0:
                    at.new_array('dihedrals', _store_dihedrals(dihedrals_in,
                                                               ind_of_id, N))
                    at.info['types']['dihedral'] = dihedral_types
                if len(impropers_in) > 0:
                    at.new_array('impropers', _store_impropers(impropers_in,
                                                               ind_of_id, N))
                    at.info['types']['improper'] = improper_types

    at.info['comment'] = comment

    return at


def _prepare_bonds(atoms):
    """Read bonds in Atoms.arrays[] and prepare for writing to datafile."""
    if 'bonds' not in atoms.arrays:
        return None

    bonds = []
    for at1, atom_bonds in enumerate(atoms.arrays['bonds']):
        for bond_type in atom_bonds:
            for at2 in atom_bonds[bond_type]:
                # ID type at1 at2
                # ID is list index +1
                # FIXME: bond IDs are not preserved.
                bonds.append([bond_type, at1 + 1, at2 + 1])
    return bonds


def _print_bonded_section(fd, bonds, section_title):
    """Output a properly formatted bonded section to fd."""
    if section_title not in ['Bonds', 'Angles', 'Dihedrals', 'Impropers']:
        raise NotImplementedError(f'Unknown section {section_title}.')

    if bonds is None:
        return

    fd.write(f'\n\n{section_title}\n\n')
    for bond_id, bond in enumerate(bonds):
        fd.write(f'{bond_id + 1:6d}')
        for atom in bond:
            fd.write(f' {atom:6d}')
        fd.write('\n')


def _print_coeff_section(fd, coeffs, section_title):
    """Output a Coeffs section to fd."""
    if section_title not in ['Pair', 'Bond', 'Angle', 'Dihedral', 'Improper']:
        raise NotImplementedError(f'Unknown section {section_title} Coeffs.')

    fd.write(f'\n{section_title} Coeffs\n\n')
    # TODO: support {Pair,Bond,Angle,Dihedral,Improper}_style comment:
    # https://docs.lammps.org/read_data.html#format-of-the-body-of-a-data-file

    for coeff in coeffs:
        fd.write(' '.join(coeff))
        fd.write('\n')

    fd.write('\n')


def _prepare_angles(atoms):
    """Read angles in Atoms.arrays[] and prepare for writing to datafile."""
    if 'angles' not in atoms.arrays:
        return None

    angles = []
    for at2, atom_angles in enumerate(atoms.arrays['angles']):
        for angle_type in atom_angles:
            for at1, at3 in atom_angles[angle_type]:
                # ID type at1 at2 at3
                # ID is list index +1
                # FIXME: angle IDs are not preserved.
                angles.append([angle_type, at1 + 1, at2 + 1, at3 + 1])
    return angles


def _prepare_dihedrals(atoms):
    """Read dihedrals in Atoms.arrays[] and prepare for writing to datafile."""
    if 'dihedrals' not in atoms.arrays:
        return None

    dihedrals = []
    for at2, atom_dihedrals in enumerate(atoms.arrays['dihedrals']):
        for dihedral_type in atom_dihedrals:
            for at1, at3, at4 in atom_dihedrals[dihedral_type]:
                # ID type at1 at2 at3
                # ID is list index +1
                # FIXME: dihedral IDs are not preserved.
                dihedrals.append([dihedral_type,
                                  at1 + 1, at2 + 1, at3 + 1, at4 + 1])
    return dihedrals


def _prepare_impropers(atoms):
    """Read impropers in Atoms.arrays[] and prepare for writing to datafile."""
    if 'impropers' not in atoms.arrays:
        return None

    impropers = []
    for at1, atom_impropers in enumerate(atoms.arrays['impropers']):
        for improper_type in atom_impropers:
            for at2, at3, at4 in atom_impropers[improper_type]:
                # ID type at1 at2 at3
                # ID is list index +1
                # FIXME: improper IDs are not preserved.
                impropers.append([improper_type,
                                  at1 + 1, at2 + 1, at3 + 1, at4 + 1])
    return impropers


@writer
def write_lammps_data(fd, atoms, specorder=None, force_skew=False,
                      prismobj=None, velocities=False, units='metal',
                      atom_style=ASTYLE_ATOMIC):
    """Write atomic structure data to a LAMMPS data file."""

    # FIXME: We should add a check here that the encoding of the file object
    #        is actually ascii once the 'encoding' attribute of IOFormat
    #        objects starts functioning in implementation (currently it
    #        doesn't do  anything).

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                "Can only write one configuration to a lammps data file!"
            )
        atoms = atoms[0]

    if hasattr(fd, "name"):
        fd.write("{0} (written by ASE)\n\n".format(fd.name))
    else:
        fd.write("(written by ASE)\n\n")

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write("{0} atoms\n".format(n_atoms))

    bonds = _prepare_bonds(atoms)
    angles = _prepare_angles(atoms)
    dihedrals = _prepare_dihedrals(atoms)
    impropers = _prepare_impropers(atoms)

    if atom_style in [ASTYLE_BOND, ASTYLE_ANGLE, ASTYLE_MOLECULAR,
                      ASTYLE_FULL]:
        if bonds is not None:
            fd.write(f"{len(bonds)} bonds\n")
        if atom_style != ASTYLE_BOND:
            if angles is not None:
                fd.write(f"{len(angles)} angles\n")
            if atom_style != ASTYLE_ANGLE:
                if dihedrals is not None:
                    fd.write(f"{len(dihedrals)} dihedrals\n")
                if impropers is not None:
                    fd.write(f"{len(impropers)} impropers\n")

    if specorder is None:
        # This way it is assured that LAMMPS atom types are always
        # assigned predictably according to the alphabetic order
        species = sorted(set(symbols))
    else:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        species = specorder
    n_atom_types = len(species)
    fd.write("{0}  atom types\n".format(n_atom_types))

    if 'types' in atoms.info and atom_style in [ASTYLE_BOND, ASTYLE_ANGLE,
                                                ASTYLE_MOLECULAR, ASTYLE_FULL]:
        for item in atoms.info['types']:
            fd.write(f"{atoms.info['types'][item]} {item} types\n")

    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj

    # Get cell parameters and convert from ASE units to LAMMPS units
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), "distance",
                                        "ASE", units)

    fd.write("0.0 {0:23.17g}  xlo xhi\n".format(xhi))
    fd.write("0.0 {0:23.17g}  ylo yhi\n".format(yhi))
    fd.write("0.0 {0:23.17g}  zlo zhi\n".format(zhi))

    if force_skew or p.is_skewed():
        fd.write(
            "{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n".format(
                xy, xz, yz
            )
        )
    fd.write("\n\n")

    # Print {Bond,Angle,Dihedrals,Impropers}Coeffs sections
    if 'coeffs' in atoms.info:
        if 'Pair' in atoms.info['coeffs']:
            _print_coeff_section(fd, atoms.info['coeffs']['Pair'], 'Pair')

        if atom_style in [ASTYLE_BOND, ASTYLE_ANGLE, ASTYLE_MOLECULAR,
                          ASTYLE_FULL]:
            if 'Bond' in atoms.info['coeffs']:
                _print_coeff_section(fd, atoms.info['coeffs']['Bond'], 'Bond')
            if atom_style != ASTYLE_BOND:
                if 'Angle' in atoms.info['coeffs']:
                    _print_coeff_section(fd, atoms.info['coeffs']['Angle'],
                                         'Angle')
                if atom_style != ASTYLE_ANGLE:
                    if 'Dihedral' in atoms.info['coeffs']:
                        _print_coeff_section(fd,
                                             atoms.info['coeffs']['Dihedral'],
                                             'Dihedral')
                    if 'Improper' in atoms.info['coeffs']:
                        _print_coeff_section(fd,
                                             atoms.info['coeffs']['Improper'],
                                             'Improper')

    # Print Masses section
    masses = convert(atoms.get_masses(), "mass", "ASE", units)
    fd.write("Masses\n\n")
    for (atom_type, symbol) in enumerate(species):
        fd.write(f"{atom_type + 1} {masses[symbols.index(symbol)]:g}\n")
    fd.write("\n\n")

    # Write (unwrapped) atomic positions.  If wrapping of atoms back into the
    # cell along periodic directions is desired, this should be done manually
    # on the Atoms object itself beforehand.
    fd.write("Atoms\n\n")

    # TODO: support atom_style comment after Atoms section header.
    # https://docs.lammps.org/read_data.html#format-of-the-body-of-a-data-file
    # Currently the test fails because the regexp in parse_lammps_data_file.py
    # won't match a section header with a comment.
    # fd.write(f"Atoms # {atom_style} \n\n")

    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)

    if atom_style == ASTYLE_ATOMIC:
        for i, r in enumerate(pos):
            # Convert position from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write(
                "{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n".format(
                    *(i + 1, s) + tuple(r)
                )
            )
    elif atom_style == ASTYLE_CHARGE:
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            # Convert position and charge from ASE units to LAMMPS units
            r = convert(r, "distance", "ASE", units)
            q = convert(q, "charge", "ASE", units)
            s = species.index(symbols[i]) + 1
            fd.write("{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n"
                     .format(*(i + 1, s, q) + tuple(r)))
    elif atom_style in (ASTYLE_ANGLE, ASTYLE_BOND, ASTYLE_MOLECULAR,
                        ASTYLE_FULL):
        # The label 'mol-id' has apparently been introduced in read earlier,
        # but so far not implemented here. Wouldn't a 'underscored' label
        # be better, i.e. 'mol_id' or 'molecule_id'?
        if atoms.has('mol-id'):
            molecules = atoms.get_array('mol-id')
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " mol-id dtype must be subtype of np.integer, and"
                    " not {:s}.").format(str(molecules.dtype)))
            if (len(molecules) != len(atoms)) or (molecules.ndim != 1):
                raise TypeError((
                    "If 'atoms' object has 'mol-id' array, then"
                    " each atom must have exactly one mol-id."))
        else:
            # Assigning each atom to a distinct molecule id would seem
            # preferable above assigning all atoms to a single molecule id per
            # default, as done within ase <= v 3.19.1. I.e., molecules =
            # = np.arange(start=1, stop=len(atoms)+1, step=1, dtype=int)
            # However, according to LAMMPS default behavior,
            molecules = np.zeros(len(atoms), dtype=int)
            # which is what happens if one creates new atoms within LAMMPS
            # without explicitly taking care of the molecule id.
            # Quote from docs at https://lammps.sandia.gov/doc/read_data.html:
            #    The molecule ID is a 2nd identifier attached to an atom.
            #    Normally, it is a number from 1 to N, identifying which
            #    molecule the atom belongs to. It can be 0 if it is a
            #    non-bonded atom or if you don't care to keep track of
            #    molecule assignments.

        if atom_style == ASTYLE_FULL:
            charges = atoms.get_initial_charges()
            for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
                # Convert position and charge from ASE units to LAMMPS units
                r = convert(r, "distance", "ASE", units)
                q = convert(q, "charge", "ASE", units)
                s = species.index(symbols[i]) + 1
                fd.write("{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} "
                         "{6:23.17g}\n".format(*(i + 1, m, s, q) + tuple(r)))
        else:
            for i, (m, r) in enumerate(zip(molecules, pos)):
                # Convert position from ASE units to LAMMPS units
                r = convert(r, "distance", "ASE", units)
                s = species.index(symbols[i]) + 1
                fd.write("{0:>6} {1:>3} {2:>3} {3:23.17g} {4:23.17g} "
                         "{5:23.17g}\n".format(*(i + 1, m, s) + tuple(r)))
    else:
        raise NotImplementedError

    if velocities and atoms.get_velocities() is not None:
        fd.write("\n\nVelocities \n\n")
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            # Convert velocity from ASE units to LAMMPS units
            v = convert(v, "velocity", "ASE", units)
            fd.write(
                "{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n".format(
                    *(i + 1,) + tuple(v)
                )
            )

    # Print Bonds, Angles, Dihedrals, impropers sections
    if atom_style in [ASTYLE_BOND, ASTYLE_ANGLE, ASTYLE_MOLECULAR,
                      ASTYLE_FULL]:
        _print_bonded_section(fd, bonds, 'Bonds')
        if atom_style != ASTYLE_BOND:
            _print_bonded_section(fd, angles, 'Angles')
            if atom_style != ASTYLE_ANGLE:
                _print_bonded_section(fd, dihedrals, 'Dihedrals')
                _print_bonded_section(fd, impropers, 'Impropers')

    fd.flush()
