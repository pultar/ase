"""Module to read and write atoms in cif file format.

See http://www.iucr.org/resources/cif/spec/version1.1/cifsyntax for a
description of the file format.  STAR extensions as save frames,
global blocks, nested loops and multi-data values are not supported.
The "latin-1" encoding is required by the IUCR specification.
"""

import shlex
import warnings

from ase.utils import basestring
from ase.io.cif import tags2atoms

from pycodcif import parse_cif as pycodcif_parse_cif


def pycodcif2blocks(datablocks):
    blocks = []
    for datablock in datablocks:
        tags = datablock['values']
        for tag in tags.keys():
            if len(tags[tag]) == 1:
                tags[tag] = tags[tag][0]
        blocks.append(datablock['name'], tags)
    return blocks


def parse_cif(fileobj):
    """Parse a CIF file. Returns a list of blockname and tag
    pairs. All tag names are converted to lower case."""

    if not isinstance(fileobj, basestring):
        fileobj = fileobj.name()

    data = pycodcif_parse_cif(fileobj)
    return pycodcif2blocks(data)


def read_cif(fileobj, index, store_tags=False, primitive_cell=False,
             subtrans_included=True, fractional_occupancies=True):
    """Read Atoms object from CIF file. *index* specifies the data
    block number or name (if string) to return.

    If *index* is None or a slice object, a list of atoms objects will
    be returned. In the case of *index* is *None* or *slice(None)*,
    only blocks with valid crystal data will be included.

    If *store_tags* is true, the *info* attribute of the returned
    Atoms object will be populated with all tags in the corresponding
    cif data block.

    If *primitive_cell* is true, the primitive cell will be built instead
    of the conventional cell.

    If *subtrans_included* is true, sublattice translations are
    assumed to be included among the symmetry operations listed in the
    CIF file (seems to be the common behaviour of CIF files).
    Otherwise the sublattice translations are determined from setting
    1 of the extracted space group.  A result of setting this flag to
    true, is that it will not be possible to determine the primitive
    cell.

    If *fractional_occupancies* is true, the resulting atoms object will be
    tagged equipped with an array `occupancy`. Also, in case of mixed
    occupancies, the atom's chemical symbol will be that of the most dominant
    species.
    """
    blocks = parse_cif(fileobj)
    # Find all CIF blocks with valid crystal data
    images = []
    for name, tags in blocks:
        try:
            atoms = tags2atoms(tags, store_tags, primitive_cell,
                               subtrans_included,
                               fractional_occupancies=fractional_occupancies)
            images.append(atoms)
        except KeyError:
            pass
    for atoms in images[index]:
        yield atoms
