"""Module to read and write atoms in PDB file format.

See::

    http://www.wwpdb.org/documentation/file-format

Note: The PDB format saves cell lengths and angles; hence the absolute
orientation is lost when saving.  Saving and loading a file will
conserve the scaled positions, not the absolute ones.
"""

import warnings

import numpy as np

from ase.atoms import Atoms
from ase.parallel import paropen
from ase.geometry import cellpar_to_cell
from ase.utils import basestring
from ase.io.espresso import label_to_symbol

def read_atom_line(line_full):
    """
    Read atom line from pdb format
    HETATM    1  H14 ORTE    0       6.301   0.693   1.919  1.00  0.00           H
    """

    line = line_full.rstrip('\n')
    type_atm = line[0:6]
    if type_atm == "ATOM  " or type_atm == "HETATM":

        name = line[12:16].strip()

        altloc = line[16]
        # resname and chainid are merged
        #  they are space separated
        resname = line[17:22]
        # chainid = line[21]

        resseq = int(line[22:26].split()[0])  # sequence identifier
        # icode = line[26]          # insertion code, not used

        # atomic coordinates
        try:
            coord = np.array([float(line[30:38]),
                              float(line[38:46]),
                              float(line[46:54])], dtype=np.float64)
        except ValueError:
            raise ValueError("Invalid or missing coordinate(s)")

        # occupancy & B factor
        try:
            occupancy = float(line[54:60])
        except ValueError:
            occupancy = None  # Rather than arbitrary zero or one

        if occupancy is not None and occupancy < 0:
            warnings.warn("Negative occupancy in one or more atoms")

        try:
            bfactor = float(line[60:66])
        except ValueError:
            bfactor = 0.0  # The PDB use a default of zero if the data is missing

        # segid = line[72:76] # not used
        symbol = line[76:78].strip().capitalize()

    else:
        raise ValueError("Only ATOM and HETATM supported")

    return symbol, name, altloc, resname, coord, occupancy, bfactor, resseq

def read_proteindatabank(fileobj, index=-1, read_arrays=True):
    """Read PDB files."""

    if isinstance(fileobj, basestring):
        fileobj = open(fileobj)

    images = []
    orig = np.identity(3)
    trans = np.zeros(3)
    occ = []
    bfactor = []
    residuenames = []
    residuenumbers = []
    atomtypes = []
    bonds = []

    symbols = []
    positions = []
    cell = None
    pbc = None

    def build_atoms():
        atoms = Atoms(symbols=symbols,
                      cell=cell, pbc=pbc,
                      positions=positions)

        if not read_arrays:
            return atoms

        info = {'occupancy': occ,
                'bfactor': bfactor,
                'resnames': residuenames,
                'names': atomtypes,
                'mol-ids': residuenumbers}
        for name, array in info.items():
            if len(array) == 0:
                pass
            elif len(array) != len(atoms):
                warnings.warn('Length of {} array, {}, '
                              'different from number of atoms {}'.
                              format(name, len(array), len(atoms)))
            else:
                atoms.set_array(name, np.array(array))
        atoms.set_topology()
        atoms.topology.generate_with_indices({'bonds': bonds})
        return atoms

    for line in fileobj.readlines():
        if line.startswith('CRYST1'):
            cellpar = [float(line[6:15]),  # a
                       float(line[15:24]),  # b
                       float(line[24:33]),  # c
                       float(line[33:40]),  # alpha
                       float(line[40:47]),  # beta
                       float(line[47:54])]  # gamma
            cell = cellpar_to_cell(cellpar)
            pbc = True
        for c in range(3):
            if line.startswith('ORIGX' + '123'[c]):
                orig[c] = [float(line[10:20]),
                           float(line[20:30]),
                           float(line[30:40])]
                trans[c] = float(line[45:55])

        if line.startswith('ATOM') or line.startswith('HETATM'):
            # Atom name is arbitrary and does not necessarily
            # contain the element symbol.  The specification
            # requires the element symbol to be in columns 77+78.
            # Fall back to Atom name for files that do not follow
            # the spec, e.g. packmol.

            # line_info = symbol, name, altloc, resname, coord, occupancy,
            #             bfactor, resseq
            line_info = read_atom_line(line)

            try:
                symbol = label_to_symbol(line_info[0])
            except (KeyError, IndexError):
                symbol = label_to_symbol(line_info[1])

            position = np.dot(orig, line_info[4]) + trans
            atomtypes.append(line_info[1])
            residuenames.append(line_info[3])
            if line_info[5] is not None:
                occ.append(line_info[5])
            bfactor.append(line_info[6])
            residuenumbers.append(line_info[7])

            symbols.append(symbol)
            positions.append(position)

        if line.startswith('CONECT'):
            indices = np.array(line.split()[1:], dtype=int) - 1
            for j in indices[1:]:
                if j > indices[0]:
                    # removes duplicate entries
                    bonds.append([indices[0], j])

        if line.startswith('END'):
            # End of configuration reached
            # According to the latest PDB file format (v3.30),
            # this line should start with 'ENDMDL' (not 'END'),
            # but in this way PDB trajectories from e.g. CP2K
            # are supported (also VMD supports this format).
            atoms = build_atoms()
            images.append(atoms)
            occ = []
            bfactor = []
            residuenames = []
            atomtypes = []
            symbols = []
            positions = []
            cell = None
            pbc = None

    if len(images) == 0:
        atoms = build_atoms()
        images.append(atoms)
    return images[index]


def write_proteindatabank(fileobj, images, write_arrays=True):
    """Write images to PDB-file."""
    if isinstance(fileobj, basestring):
        fileobj = paropen(fileobj, 'w')

    if hasattr(images, 'get_positions'):
        images = [images]

    fileobj.write('TITLE     {} Generated by ASE'
                  '\n'.format(images[0].get_chemical_formula().upper()))


    rotation = None
    if images[0].get_pbc().any():
        from ase.geometry import cell_to_cellpar, cellpar_to_cell

        currentcell = images[0].get_cell()
        cellpar = cell_to_cellpar(currentcell)
        exportedcell = cellpar_to_cell(cellpar)
        rotation = np.linalg.solve(currentcell, exportedcell)
        # ignoring Z-value, using P1 since we have all atoms defined explicitly
        format = 'CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1\n'
        fileobj.write(format % (cellpar[0], cellpar[1], cellpar[2],
                                cellpar[3], cellpar[4], cellpar[5]))

    #     1234567 123 6789012345678901   89   67   456789012345678901234567 890
    format = ('ATOM  {0:>5d} {1:4s} {2:>3s} {3:1s}{4:>4d}    {5:8.3f}'
              '{6:8.3f}{7:8.3f}{8:6.2f}{9:6.2f}          {10:>2s}  \n')

    # RasMol complains if the atom index exceeds 100000. There might
    # be a limit of 5 digit numbers in this field.
    MAXNUM = 100000

    symbols = images[0].get_chemical_symbols()
    natoms = len(symbols)

    for n, atoms in enumerate(images):
        fileobj.write('MODEL     ' + str(n + 1) + '\n')
        if atoms.has('ids'):
            # topology exists
            names = atoms.get_array('names')
            resnames = atoms.get_array('resnames')
            resnames[resnames == ''] = 'MOL'
            resnumbers = atoms.get_array('mol-ids')
            # making sure that atoms with same mol-id has same resname
            # a resname can have atoms of different mol-ids
            restypes = {}
            change = {}
            max_resnumber = np.max(resnumbers)
            for i, resnumber in enumerate(resnumbers):
                if resnumber not in restypes:
                    restypes[resnumber] = resnames[i]
                if restypes[resnumber] != resnames[i]:
                    if resnames[i] not in change:
                        change[resnames[i]] = {}
                    if resnumber in change[resnames[i]]:
                        resnumbers[i] = change[resnames[i]][resnumber]
                    else:
                        resnumbers[i] = max_resnumber + 1
                        restypes[resnumbers[i]] = resnames[i]
                        max_resnumber += 1
                        change[resnames[i]][resnumber] = resnumbers[i]

        else:
            names = symbols
            resnames = ['MOL'] * natoms
            resnumbers = [1] * natoms
        p = atoms.get_positions()
        occupancy = np.ones(len(atoms))
        bfactor = np.zeros(len(atoms))
        if write_arrays:
            if 'occupancy' in atoms.arrays:
                occupancy = atoms.get_array('occupancy')
            if 'bfactor' in atoms.arrays:
                bfactor = atoms.get_array('bfactor')
        if rotation is not None:
            p = p.dot(rotation)
        for a in range(natoms):
            x, y, z = p[a]
            occ = occupancy[a]
            bf = bfactor[a]
            resname = resnames[a].split()[0]
            chain_id = ('A' if len(resnames[a].split()) == 1
                        else resnames[a].split()[1])
            fileobj.write(format.format(a % MAXNUM + 1, names[a], resname,
                                        chain_id, resnumbers[a],
                                        x, y, z, occ, bf, symbols[a].upper()))
        fileobj.write('ENDMDL\n')
        if atoms.has('bonds'):
            conect_mat = np.zeros((len(atoms), len(atoms)))
            for y in atoms.topology.bonds().values():
                for x in y:
                    conect_mat[x[0], x[1]] = 1
            conect_mat = np.asarray(conect_mat + conect_mat.T, dtype=bool)
            for i in range(len(atoms)):
                count = 0
                for j in np.where(conect_mat[i])[0]:
                    if count % 4 == 0:
                        fileobj.write('\nCONECT{:5d}'.format(i + 1))
                    fileobj.write('{:5d}'.format(j + 1))
                    count += 1
            fileobj.write('\n')
        fileobj.write('END\n')
