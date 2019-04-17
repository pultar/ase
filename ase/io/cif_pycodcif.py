"""Module to read and write atoms in cif file format.

See http://www.iucr.org/resources/cif/spec/version1.1/cifsyntax for a
description of the file format.  STAR extensions as save frames,
global blocks, nested loops and multi-data values are not supported.
The "latin-1" encoding is required by the IUCR specification.
"""

import re
import shlex
import warnings

import numpy as np

from ase import Atoms
from ase.parallel import paropen
from ase.spacegroup import crystal
from ase.spacegroup.spacegroup import spacegroup_from_data, Spacegroup
from ase.utils import basestring
from ase.data import atomic_numbers, atomic_masses
from ase.io.cif_unicode import format_unicode

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


def tags2atoms(tags, store_tags=False, primitive_cell=False,
               subtrans_included=True, fractional_occupancies=True):
    """Returns an Atoms object from a cif tags dictionary.  See read_cif()
    for a description of the arguments."""
    if primitive_cell and subtrans_included:
        raise RuntimeError(
            'Primitive cell cannot be determined when sublattice translations '
            'are included in the symmetry operations listed in the CIF file, '
            'i.e. when `subtrans_included` is True.')

    cell_tags = ['_cell_length_a', '_cell_length_b', '_cell_length_c',
                 '_cell_angle_alpha', '_cell_angle_beta', '_cell_angle_gamma']

    # If any value is missing, ditch periodic boundary conditions
    has_pbc = True
    try:
        cell_values = [tags[ct] for ct in cell_tags]
        a, b, c, alpha, beta, gamma = cell_values
    except KeyError:
        has_pbc = False

    # Now get positions
    try:
        scaled_positions = np.array([tags['_atom_site_fract_x'],
                                     tags['_atom_site_fract_y'],
                                     tags['_atom_site_fract_z']]).T
    except KeyError:
        scaled_positions = None

    try:
        positions = np.array([tags['_atom_site_cartn_x'],
                              tags['_atom_site_cartn_y'],
                              tags['_atom_site_cartn_z']]).T
    except KeyError:
        positions = None

    if (positions is None) and (scaled_positions is None):
        raise RuntimeError('No positions found in structure')
    elif scaled_positions is not None and not has_pbc:
        raise RuntimeError('Structure has fractional coordinates but not '
                           'lattice parameters')

    symbols = []
    if '_atom_site_type_symbol' in tags:
        labels = tags['_atom_site_type_symbol']
    else:
        labels = tags['_atom_site_label']
    for s in labels:
        # Strip off additional labeling on chemical symbols
        m = re.search(r'([A-Z][a-z]?)', s)
        symbol = m.group(0)
        symbols.append(symbol)

    # Symmetry specification, see
    # http://www.iucr.org/resources/cif/dictionaries/cif_sym for a
    # complete list of official keys.  In addition we also try to
    # support some commonly used depricated notations
    no = None
    if '_space_group.it_number' in tags:
        no = tags['_space_group.it_number']
    elif '_space_group_it_number' in tags:
        no = tags['_space_group_it_number']
    elif '_symmetry_int_tables_number' in tags:
        no = tags['_symmetry_int_tables_number']

    symbolHM = None
    if '_space_group.Patterson_name_h-m' in tags:
        symbolHM = tags['_space_group.patterson_name_h-m']
    elif '_symmetry_space_group_name_h-m' in tags:
        symbolHM = tags['_symmetry_space_group_name_h-m']
    elif '_space_group_name_h-m_alt' in tags:
        symbolHM = tags['_space_group_name_h-m_alt']

    if symbolHM is not None:
        symbolHM = old_spacegroup_names.get(symbolHM.strip(), symbolHM)

    for name in ['_space_group_symop_operation_xyz',
                 '_space_group_symop.operation_xyz',
                 '_symmetry_equiv_pos_as_xyz']:
        if name in tags:
            sitesym = tags[name]
            break
    else:
        sitesym = None

    # The setting needs to be passed as either 1 or two, not None (default)
    setting = 1
    spacegroup = 1
    if sitesym is not None:
        subtrans = [(0.0, 0.0, 0.0)] if subtrans_included else None
        spacegroup = spacegroup_from_data(
            no=no, symbol=symbolHM, sitesym=sitesym, subtrans=subtrans,
            setting=setting)
    elif no is not None:
        spacegroup = no
    elif symbolHM is not None:
        spacegroup = symbolHM
    else:
        spacegroup = 1

    kwargs = {}
    if store_tags:
        kwargs['info'] = tags.copy()

    if 'D' in symbols:
        deuterium = [symbol == 'D' for symbol in symbols]
        symbols = [symbol if symbol != 'D' else 'H' for symbol in symbols]
    else:
        deuterium = False

    setting_name = None
    if '_space_group_crystal_system' in tags:
        setting_name = tags['_space_group_crystal_system']
    elif '_symmetry_cell_setting' in tags:
        setting_name = tags['_symmetry_cell_setting']
    if setting_name:
        no = Spacegroup(spacegroup).no
        # rhombohedral systems
        if no in (146, 148, 155, 160, 161, 166, 167):
            if setting_name == 'hexagonal':
                setting = 1
            elif setting_name in ('trigonal', 'rhombohedral'):
                setting = 2
            else:
                warnings.warn(
                    'unexpected crystal system %r for space group %r' % (
                        setting_name, spacegroup))
        # FIXME - check for more crystal systems...
        else:
            warnings.warn(
                'crystal system %r is not interpreated for space group %r. '
                'This may result in wrong setting!' % (
                    setting_name, spacegroup))

    occupancies = None
    if fractional_occupancies:
        try:
            occupancies = tags['_atom_site_occupancy']
            # no warnings in this case
            kwargs['onduplicates'] = 'keep'
        except KeyError:
            pass
    else:
        try:
            if not np.allclose(tags['_atom_site_occupancy'], 1.):
                warnings.warn(
                    'Cif file containes mixed/fractional occupancies. '
                    'Consider using `fractional_occupancies=True`')
                kwargs['onduplicates'] = 'keep'
        except KeyError:
            pass

    if has_pbc:
        if scaled_positions is None:
            _ = Atoms(symbols, positions=positions,
                      cell=[a, b, c, alpha, beta, gamma])
            scaled_positions = _.get_scaled_positions()

        if deuterium:
            numbers = np.array([atomic_numbers[s] for s in symbols])
            masses = atomic_masses[numbers]
            masses[deuterium] = 2.01355
            kwargs['masses'] = masses

        atoms = crystal(symbols, basis=scaled_positions,
                        cellpar=[a, b, c, alpha, beta, gamma],
                        spacegroup=spacegroup,
                        occupancies=occupancies,
                        setting=setting,
                        primitive_cell=primitive_cell,
                        **kwargs)
    else:
        atoms = Atoms(symbols, positions=positions,
                      info=kwargs.get('info', None))
        if occupancies is not None:
            # Compile an occupancies dictionary
            occ_dict = {}
            for i, sym in enumerate(symbols):
                occ_dict[i] = {sym: occupancies[i]}
            atoms.info['occupancy'] = occ_dict

        if deuterium:
            masses = atoms.get_masses()
            masses[atoms.numbers == 1] = 1.00783
            masses[deuterium] = 2.01355
            atoms.set_masses(masses)

    return atoms


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
