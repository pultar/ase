import re
import numpy as np

import decimal as dec
from ase.lammpsatoms import LammpsAtoms
from ase.lammpsquaternions import LammpsQuaternions
from ase.quaternions import Quaternions
from ase.parallel import paropen
from ase.calculators.lammpslib import unit_convert
from ase.utils import basestring


def read_lammps_data(fileobj, Z_of_type=None, style='full', sort_by_id=False,
                     units="metal"):
    """Method which reads a LAMMPS data file.

    sort_by_id: Order the particles according to their id. Might be faster to
    switch it off.
    Units are set by default to the style=metal setting in LAMMPS.
    """
    if isinstance(fileobj, basestring):
        f = paropen(fileobj)
    else:
        f = fileobj

    # load everything into memory
    lines = f.readlines()

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
    mmcharge_in = {}
    mass_in = {}
    vel_in = {}
    bonds_in = []
    angles_in = []
    dihedrals_in = []
    impropers_in = []

    sections = ["Atoms",
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
                "AngleAngle Coeffs"]
    header_fields = ["atoms",
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
                     "xy xz yz"]
    sections_re = '(' + '|'.join(sections).replace(' ', '\\s+') + ')'
    header_fields_re = '(' + '|'.join(header_fields).replace(' ', '\\s+') + ')'

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
                id = int(fields[0])
                if style == 'full' and (len(fields) == 7 or len(fields) == 10):
                    # id mol-id type q x y z [tx ty tz]
                    pos_in[id] = (int(fields[2]), float(fields[4]),
                                  float(fields[5]), float(fields[6]))
                    mol_id_in[id] = int(fields[1])
                    mmcharge_in[id] = float(fields[3])
                    if len(fields) == 10:
                        travel_in[id] = (int(fields[7]),
                                         int(fields[8]),
                                         int(fields[9]))
                elif (style == 'atomic' and
                      (len(fields) == 5 or len(fields) == 8)):
                    # id type x y z [tx ty tz]
                    pos_in[id] = (int(fields[1]), float(fields[2]),
                                  float(fields[3]), float(fields[4]))
                    if len(fields) == 8:
                        travel_in[id] = (int(fields[5]),
                                         int(fields[6]),
                                         int(fields[7]))
                elif ((style == 'angle' or style == 'bond' or
                       style == 'molecular') and
                      (len(fields) == 6 or len(fields) == 9)):
                    # id mol-id type x y z [tx ty tz]
                    pos_in[id] = (int(fields[2]), float(fields[3]),
                                  float(fields[4]), float(fields[5]))
                    mol_id_in[id] = int(fields[1])
                    if len(fields) == 9:
                        travel_in[id] = (int(fields[6]),
                                         int(fields[7]),
                                         int(fields[8]))
                else:
                    raise RuntimeError("Style '{}' not supported or invalid "
                                       "number of fields {}"
                                       "".format(style, len(fields)))
            elif section == "Velocities":  # id vx vy vz
                vel_in[int(fields[0])] = (float(fields[1]),
                                          float(fields[2]),
                                          float(fields[3]))
            elif section == "Masses":
                mass_in[int(fields[0])] = float(fields[1])
            elif section == "Bonds":  # id type atom1 atom2
                bonds_in.append((int(fields[1]),
                                 int(fields[2]),
                                 int(fields[3])))
            elif section == "Angles":  # id type atom1 atom2 atom3
                angles_in.append((int(fields[1]),
                                  int(fields[2]),
                                  int(fields[3]),
                                  int(fields[4])))
            elif section == "Dihedrals":  # id type atom1 atom2 atom3 atom4
                dihedrals_in.append((int(fields[1]),
                                     int(fields[2]),
                                     int(fields[3]),
                                     int(fields[4]),
                                     int(fields[5])))
            elif section == "Impropers":  # id type atom1 atom2 atom3 atom4
                impropers_in.append((int(fields[1]),
                                     int(fields[2]),
                                     int(fields[3]),
                                     int(fields[4]),
                                     int(fields[5])))

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
    if len(mmcharge_in) > 0:
        mmcharge = np.zeros((N), float)
    else:
        mmcharge = None
    if len(travel_in) > 0:
        travel = np.zeros((N, 3), int)
    else:
        travel = None
    if len(bonds_in) > 0:
        bonds = [{} for _ in range(N)]
    else:
        bonds = None
    if len(angles_in) > 0:
        angles = [{} for _ in range(N)]
    else:
        angles = None
    if len(dihedrals_in) > 0:
        dihedrals = [{} for _ in range(N)]
    else:
        dihedrals = None
    if len(impropers_in) > 0:
        impropers = [{} for _ in range(N)]
    else:
        impropers = None

    ind_of_id = {}
    # copy per-atom quantities from read-in values
    for (i, id) in enumerate(pos_in.keys()):
        # by id
        ind_of_id[id] = i
        if sort_by_id:
            ind = id - 1
        else:
            ind = i
        type = pos_in[id][0]
        positions[ind, :] = [pos_in[id][1], pos_in[id][2], pos_in[id][3]]
        if velocities is not None:
            velocities[ind, :] = [vel_in[id][0], vel_in[id][1], vel_in[id][2]]
        if travel is not None:
            travel[ind] = travel_in[id]
        if mol_id is not None:
            mol_id[i] = mol_id_in[id]
        if mmcharge is not None:
            mmcharge[i] = mmcharge_in[id]
        ids[i] = id
        # by type
        types[ind] = type
        if Z_of_type is None:
            numbers[ind] = type
        else:
            numbers[ind] = Z_of_type[type]
        if masses is not None:
            masses[ind] = mass_in[type]
    # convert units
    positions *= unit_convert("distance", units)
    cell *= unit_convert("distance", units)
    if masses is not None:
        masses *= unit_convert("mass", units)
    if velocities is not None:
        velocities *= unit_convert("velocity", units)

    # create ase.Atoms
    at = LammpsAtoms(positions=positions,
               numbers=numbers,
               masses=masses,
               cell=cell,
               pbc=[True, True, True])
    # set velocities (can't do it via constructor)
    if velocities is not None:
        at.set_velocities(velocities)
    at.set_array('id', ids, int)
    at.set_array('type', types, int)
    if travel is not None:
        at.set_array('travel', travel, int)
    if mol_id is not None:
        at.set_array('mol-id', mol_id, int)
    if mmcharge is not None:
        at.set_array('mmcharge', mmcharge, float)

    if bonds is not None:
        for (type, a1, a2) in bonds_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            # Double list for bonds to make it consistent with other sections
            bonds[i_a1][type] = bonds[i_a1].get(type, []) + [[i_a2]]
        at.set_array('bonds', bonds, 'object')

    if angles is not None:
        for (type, a1, a2, a3) in angles_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            angles[i_a2][type] = angles[i_a2].get(type, []) + [[i_a1, i_a3]]
            at.set_array('angles', angles, 'object')

    if dihedrals is not None:
        for (type, a1, a2, a3, a4) in dihedrals_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            i_a4 = ind_of_id[a4]
            dihedrals[i_a1][type] = dihedrals[i_a1].get(type, []) \
                + [[i_a2, i_a3, i_a4]]
            at.set_array('dihedrals', dihedrals, 'object')

    if impropers is not None:
        for (type, a1, a2, a3, a4) in impropers_in:
            i_a1 = ind_of_id[a1]
            i_a2 = ind_of_id[a2]
            i_a3 = ind_of_id[a3]
            i_a4 = ind_of_id[a4]
            impropers[i_a1][type] = impropers[i_a1].get(type, []) \
                + [[i_a2, i_a3, i_a4]]
            at.set_array('impropers', impropers, 'object')

    at.info['comment'] = comment

    return at

def write_lammps_data(fileobj, atoms, specorder=None, force_skew=False,
                      prismobj=None, velocities=False, style='full'):
    """Write atomic structure data to a LAMMPS data_ file."""
    if isinstance(fileobj, basestring):
        f = paropen(fileobj, 'wb')
        close_file = True
    else:
        # Presume fileobj acts like a fileobj
        f = fileobj
        close_file = False

    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError(
                'Can only write one configuration to a lammps data file!')
        atoms = atoms[0]

    # TODO: add quarternions printing
    if type(atoms) == Quaternions:
        atoms = LammpsQuaternions(atoms)
    elif not isinstance(atoms, LammpsAtoms):
        atoms = LammpsAtoms(atoms)

    if specorder is not None:
        # To index elements in the LAMMPS data file
        # (indices must correspond to order in the potential file)
        atoms.update(specorder)

    f.write('{0} (written by ASE) \n\n'.format(f.name))

    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    f.write('{0} \t atoms \n'.format(n_atoms))
    for prop in ['bonds', 'angles', 'dihedrals', 'impropers']:
        f.write('{0} \t '
                '{1} \n'.format(atoms.get_num_prop(prop),
                                prop))

    # LammpsAtoms always assigns type
    types = atoms.get_array('type')

    n_types = atoms.get_num_types('type')
    f.write('{0:8} \t '
            'atoms types\n'.format(n_types))
    for prop in ['bond types', 'angle types', 'dihedral types',
                 'improper types']:
        f.write('{0:8} \t '
                '{1} \n'.format(atoms.get_num_types(prop.split()[0] + 's'),
                                prop))
    if prismobj is None:
        p = Prism(atoms.get_cell(), digits=6)
    else:
        p = prismobj
    xhi, yhi, zhi, xy, xz, yz = p.get_lammps_prism_str()

    f.write('\t 0.0 {0}  xlo xhi\n'.format(xhi))
    f.write('\t 0.0 {0}  ylo yhi\n'.format(yhi))
    f.write('\t 0.0 {0}  zlo zhi\n'.format(zhi))

    if force_skew or p.is_skewed():
        f.write('{0} {1} {2}  xy xz yz\n'.format(xy, xz, yz))
    f.write('\n\n')

    f.write('Masses \n\n')
    for i in atoms.get_types('type'):
        indx = np.where(i == types)[0][0]
        mass = atoms.get_masses()[indx]
        sym = atoms.get_chemical_symbols()[indx]
        f.write('{0:>6} {1:8.4f} # {2}\n'.format(i, mass, sym))
    f.write('\n\n')

    f.write('Atoms \n\n')
    if atoms.has('travel'):
        travel = atoms.get_array('travel')
    else:
        travel = None
    id = atoms.get_array('id')
    mol_id = atoms.get_array('mol-id')
    pos = p.pos_to_lammps_fold_str(atoms.get_positions())
    if style == 'full':
        # id mol-id type q x y z [tx ty tz]
        if atoms.has('mmcharge'):
            mmcharge = atoms.get_array('mmcharge')
            for i, r in enumerate(pos):
                f.write('{0:6} {1:4} {2:4} {3:8.6f}'
                        ' {4} {5} {6}'.format(id[i],
                                              mol_id[i],
                                              types[i],
                                              mmcharge[i],
                                              *r))
                if travel:
                    f.write(' {0:8.4f} {1:8.4f} '
                            '{2:8.4f}\n'.format(*travel[i]))
                else:
                    f.write('\n')
        else:
            raise RuntimeError('style "full" not possible. '
                               'The system does not have mmcharge')
    elif style == 'atomic':
        # id type x y z [tx ty tz]
        for i, r in enumerate(pos):
            f.write('{0:6} {1:4}'
                    ' {2} {3} {4}'.format(id[i],
                                          types[i],
                                          *r))
            if travel:
                f.write(' {0:8.4f} {1:8.4f} '
                        '{2:8.4f}\n'.format(*travel[i]))
            else:
                f.write('\n')
    elif (style == 'angle' or style == 'bond' or
                       style == 'molecular'):
        # id mol-id type x y z [tx ty tz]
        for i, r in enumerate(pos):
            f.write('{0:6} {1:4} {2:4}'
                    ' {3} {4} {5}'.format(id[i],
                                          mol_id[i],
                                          types[i],
                                          *r))
            if travel:
                f.write(' {0:8.4f} {1:8.4f} '
                        '{2:8.4f}\n'.format(*travel[i]))
            else:
                f.write('\n')
    else:
        raise RuntimeError('style {0} not supported. '
                           'Use: full, atomic, angle, bond, or'
                           'molecular'.format(style))
    f.write('\n\n')

    if atoms.has('bonds'):
        count = 1
        f.write('Bonds \n\n')
        for indx, item in enumerate(atoms.get_array('bonds')):
            for key, values in item.items():
                for value in values:
                    f.write('{0:6} {1:6} {2:6} '
                            '{3:6}\n'.format(count,
                                              key,
                                              indx + 1,
                                              value[0]))
                    count += 1

    if atoms.has('angles'):
        count = 1
        f.write('Angles \n\n')
        for indx, item in enumerate(atoms.get_array('angles')):
            for key, values in item.items():
                for value in values:
                    f.write('{0:6} {1:6} {2:6}'
                            '{3:6} {4:6}\n'.format(count,
                                                   key,
                                                   value[0],
                                                   indx + 1,
                                                   value[1]))
                    count += 1

    if atoms.has('dihedrals'):
        count = 1
        f.write('Dihedrals \n\n')
        for indx, item in enumerate(atoms.get_array('dihedrals')):
            for key, values in item.items():
                for value in values:
                    _ = [i + 1 for i in value]
                    f.write('{0:6} {1:6} {2:6} {3:6} {4:6} '
                            '{5:6}\n'.format(count,
                                             key,
                                             indx + 1,
                                             *_))
                    count += 1


    if atoms.has('impropers'):
        count = 1
        f.write('Impropers \n\n')
        for indx, item in enumerate(atoms.get_array('impropers')):
            for key, values in item.items():
                for value in values:
                    _ = [i + 1 for i in value]
                    f.write('{0:6} {1:6} {2:6} {3:6} {4:6} '
                            '{5:6}\n'.format(count,
                                             key,
                                             indx + 1,
                                             *_))
                    count += 1

    if velocities and atoms.get_velocities() is not None:
        f.write('\n\nVelocities \n\n')
        for i, v in enumerate(atoms.get_velocities() / (Ang/(fs*1000.))):
            f.write('{0:>6} {1} {2} {3}\n'.format(
                    *(i + 1,) + tuple(v)))

    f.flush()
    if close_file:
        f.close()


class Prism(object):

    def __init__(self, cell, pbc=(True, True, True), digits=10):
        """Create a lammps-style triclinic prism object from a cell

        The main purpose of the prism-object is to create suitable
        string representations of prism limits and atom positions
        within the prism.
        When creating the object, the digits parameter (default set to 10)
        specify the precision to use.
        lammps is picky about stuff being within semi-open intervals,
        e.g. for atom positions (when using create_atom in the in-file),
        x must be within [xlo, xhi).
        """
        a, b, c = cell
        an, bn, cn = [np.linalg.norm(v) for v in cell]

        alpha = np.arccos(np.dot(b, c) / (bn * cn))
        beta = np.arccos(np.dot(a, c) / (an * cn))
        gamma = np.arccos(np.dot(a, b) / (an * bn))

        xhi = an
        xyp = np.cos(gamma) * bn
        yhi = np.sin(gamma) * bn
        xzp = np.cos(beta) * cn
        yzp = (bn * cn * np.cos(alpha) - xyp * xzp) / yhi
        zhi = np.sqrt(cn**2 - xzp**2 - yzp**2)

        # Set precision
        self.car_prec = dec.Decimal('10.0') ** \
            int(np.floor(np.log10(max((xhi, yhi, zhi)))) - digits)
        self.dir_prec = dec.Decimal('10.0') ** (-digits)
        self.acc = float(self.car_prec)
        self.eps = np.finfo(xhi).eps

        # For rotating positions from ase to lammps
        Apre = np.array(((xhi, 0, 0),
                         (xyp, yhi, 0),
                         (xzp, yzp, zhi)))
        self.R = np.dot(np.linalg.inv(cell), Apre)

        # Actual lammps cell may be different from what is used to create R
        def fold(vec, pvec, i):
            p = pvec[i]
            x = vec[i] + 0.5 * p
            n = (np.mod(x, p) - x) / p
            return [float(self.f2qdec(a)) for a in (vec + n * pvec)]

        Apre[1, :] = fold(Apre[1, :], Apre[0, :], 0)
        Apre[2, :] = fold(Apre[2, :], Apre[1, :], 1)
        Apre[2, :] = fold(Apre[2, :], Apre[0, :], 0)

        self.A = Apre
        self.Ainv = np.linalg.inv(self.A)

        if self.is_skewed() and \
                (not (pbc[0] and pbc[1] and pbc[2])):
            raise RuntimeError('Skewed lammps cells MUST have '
                               'PBC == True in all directions!')

    def f2qdec(self, f):
        return dec.Decimal(repr(f)).quantize(self.car_prec, dec.ROUND_DOWN)

    def f2qs(self, f):
        return str(self.f2qdec(f))

    def f2s(self, f):
        return str(dec.Decimal(repr(f)).quantize(self.car_prec,
                                                 dec.ROUND_HALF_EVEN))

    def dir2car(self, v):
        """Direct to cartesian coordinates"""
        return np.dot(v, self.A)

    def car2dir(self, v):
        """Cartesian to direct coordinates"""
        return np.dot(v, self.Ainv)

    def fold_to_str(self, v):
        """Fold a position into the lammps cell (semi open)

        Returns tuple of str.
        """
        # Two-stage fold, first into box, then into semi-open interval
        # (within the given precision).
        d = [x % (1 - self.dir_prec) for x in
             map(dec.Decimal,
                 map(repr, np.mod(self.car2dir(v) + self.eps, 1.0)))]
        return tuple([self.f2qs(x) for x in
                      self.dir2car(list(map(float, d)))])

    def get_lammps_prism(self):
        A = self.A
        return A[0, 0], A[1, 1], A[2, 2], A[1, 0], A[2, 0], A[2, 1]

    def get_lammps_prism_str(self):
        """Return a tuple of strings"""
        p = self.get_lammps_prism()
        return tuple([self.f2s(x) for x in p])

    def pos_to_lammps_strs(self, positions):
        """Rotate an ase-cell position to the lammps cell orientation

        Returns tuple of str.
        """
        rot_positions = np.dot(positions, self.R)
        return [tuple([self.f2s(x) for x in position])
                for position in rot_positions]

    def pos_to_lammps_fold_str(self, positions):
        """Rotate and fold an ase-cell position into the lammps cell

        Returns tuple of str.
        """
        return [self.fold_to_str(np.dot(position, self.R))
                for position in positions]

    def is_skewed(self):
        acc = self.acc
        prism = self.get_lammps_prism()
        axy, axz, ayz = [np.abs(x) for x in prism[3:]]
        return (axy >= acc) or (axz >= acc) or (ayz >= acc)
