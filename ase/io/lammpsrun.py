import gzip
import struct
from os.path import splitext

import numpy as np
import re
import io
import multiprocessing

from ase.atoms import Atoms
from ase.calculators.lammps import convert
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.quaternions import Quaternions


def read_lammps_dump(infileobj, **kwargs):
    """Method which reads a LAMMPS dump file.

       LAMMPS chooses output method depending on the given suffix:
        - .bin  : binary file
        - .gz   : output piped through gzip
        - .mpiio: using mpiio (should be like cleartext,
                  with different ordering)
        - else  : normal clear-text format

    :param infileobj: string to file, opened file or file-like stream

    """
    # !TODO: add support for lammps-regex naming schemes (output per
    # processor and timestep wildcards)

    opened = False
    if isinstance(infileobj, str):
        opened = True
        suffix = splitext(infileobj)[-1]
        if suffix == ".bin":
            fileobj = paropen(infileobj, "rb")
        elif suffix == ".gz":
            # !TODO: save for parallel execution?
            fileobj = gzip.open(infileobj, "rb")
        else:
            fileobj = paropen(infileobj)
    else:
        suffix = splitext(infileobj.name)[-1]
        fileobj = infileobj

    if suffix == ".bin":
        out = read_lammps_dump_binary(fileobj, **kwargs)
        if opened:
            fileobj.close()
        return out

    out = read_lammps_dump_text(fileobj, **kwargs)

    if opened:
        fileobj.close()

    return out


def lammps_data_to_ase_atoms(
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="metal",
):
    """Extract positions and other per-atom parameters and create Atoms

    :param data: per atom data
    :param colnames: index for data
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms

    """
    if len(data.shape) == 1:
        data = data[np.newaxis, :]

    # read IDs if given and order if needed
    if "id" in colnames:
        ids = data[:, colnames.index("id")].astype(int)
        if order:
            sort_order = np.argsort(ids)
            data = data[sort_order, :]

    # determine the elements
    if "element" in colnames:
        # priority to elements written in file
        elements = data[:, colnames.index("element")]
    elif "type" in colnames:
        # fall back to `types` otherwise
        elements = data[:, colnames.index("type")].astype(int)

        # reconstruct types from given specorder
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        # todo: what if specorder give but no types?
        # in principle the masses could work for atoms, but that needs
        # lots of cases and new code I guess
        raise ValueError("Cannot determine atom types form LAMMPS dump file")

    def get_quantity(labels, quantity=None):
        try:
            cols = [colnames.index(label) for label in labels]
            if quantity:
                return convert(data[:, cols].astype(float), quantity,
                               units, "ASE")

            return data[:, cols].astype(float)
        except ValueError:
            return None

    # Positions
    positions = None
    scaled_positions = None
    if "x" in colnames:
        # doc: x, y, z = unscaled atom coordinates
        positions = get_quantity(["x", "y", "z"], "distance")
    elif "xs" in colnames:
        # doc: xs,ys,zs = scaled atom coordinates
        scaled_positions = get_quantity(["xs", "ys", "zs"])
    elif "xu" in colnames:
        # doc: xu,yu,zu = unwrapped atom coordinates
        positions = get_quantity(["xu", "yu", "zu"], "distance")
    elif "xsu" in colnames:
        # xsu,ysu,zsu = scaled unwrapped atom coordinates
        scaled_positions = get_quantity(["xsu", "ysu", "zsu"])
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    velocities = get_quantity(["vx", "vy", "vz"], "velocity")
    charges = get_quantity(["q"], "charge")
    forces = get_quantity(["fx", "fy", "fz"], "force")
    # !TODO: how need quaternions be converted?
    quaternions = get_quantity(["c_q[1]", "c_q[2]", "c_q[3]", "c_q[4]"])

    # convert cell
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")
    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)

    if quaternions is not None:
        out_atoms = Quaternions(
            symbols=elements,
            positions=positions,
            cell=cell,
            celldisp=celldisp,
            pbc=pbc,
            quaternions=quaternions,
        )
    elif positions is not None:
        # reverse coordinations transform to lammps system
        # (for all vectors = pos, vel, force)
        if prismobj:
            positions = prismobj.vector_to_ase(positions, wrap=True)

        out_atoms = atomsobj(
            symbols=elements,
            positions=positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell
        )
    elif scaled_positions is not None:
        out_atoms = atomsobj(
            symbols=elements,
            scaled_positions=scaled_positions,
            pbc=pbc,
            celldisp=celldisp,
            cell=cell,
        )

    if velocities is not None:
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        out_atoms.set_velocities(velocities)
    if charges is not None:
        out_atoms.set_initial_charges([charge[0] for charge in charges])
    if forces is not None:
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs)
        calculator = SinglePointCalculator(out_atoms, energy=0.0,
                                           forces=forces)
        out_atoms.calc = calculator

    # process the extra columns of fixes, variables and computes
    #    that can be dumped, add as additional arrays to atoms object
    for colname in colnames:
        # determine if it is a compute or fix (but not the quaternian)
        if (colname.startswith('f_') or colname.startswith('v_') or
                (colname.startswith('c_') and not colname.startswith('c_q['))):
            out_atoms.new_array(colname, get_quantity([colname]),
                                dtype='float')

    return out_atoms


def lammps_data_to_ase_atoms_typed(
    data,
    colnames,
    cell,
    celldisp,
    pbc=False,
    atomsobj=Atoms,
    order=True,
    specorder=None,
    prismobj=None,
    units="metal",
):
    """Extract positions and other per-atom parameters and create Atoms.

    Unlike `lammps_data_to_ase_atoms`, this function assumes the data it is
    given contains the type of each column. This avoids a large number of type
    checks and conversions, but requires that the data is already in the
    correct format.

    :param data: per atom data (numpy structured array)
    :param colnames: list of column names
    :param cell: cell dimensions
    :param celldisp: origin shift
    :param pbc: periodic boundaries
    :param atomsobj: function to create ase-Atoms object
    :param order: sort atoms by id. Might be faster to turn off.
    Disregarded in case `id` column is not given in file.
    :param specorder: list of species to map lammps types to ase-species
    (usually .dump files to not contain type to species mapping)
    :param prismobj: Coordinate transformation between lammps and ase
    :type prismobj: Prism
    :param units: lammps units for unit transformation between lammps and ase
    :returns: Atoms object
    :rtype: Atoms
    """

    if data.ndim == 0:
        data = data[np.newaxis]

    # read IDs if given and order if needed
    if "id" in colnames and order:
        sort_order = np.argsort(data["id"])
        data = data[sort_order]

    # Determine the elements
    if "element" in colnames:
        elements = data["element"]
    elif "type" in colnames:
        elements = data["type"]
        if specorder:
            elements = [specorder[t - 1] for t in elements]
    else:
        raise ValueError("Cannot determine atom types from LAMMPS dump file")

    # Create a dictionary to store the atom properties
    atoms_dict = {
        "symbols": elements
    }

    # Add positions or scaled positions
    if "x" in colnames:
        atoms_dict["positions"] = np.column_stack((data["x"],
                                                   data["y"],
                                                   data["z"]))
    elif "xs" in colnames:
        atoms_dict["scaled_positions"] = np.column_stack((data["xs"],
                                                          data["ys"],
                                                          data["zs"]))
    elif "xu" in colnames:
        atoms_dict["positions"] = np.column_stack((data["xu"],
                                                   data["yu"],
                                                   data["zu"]))
    elif "xsu" in colnames:
        atoms_dict["scaled_positions"] = np.column_stack((data["xsu"],
                                                          data["ysu"],
                                                          data["zsu"]))
    else:
        raise ValueError("No atomic positions found in LAMMPS output")

    # Convert cell and celldisp
    cell = convert(cell, "distance", units, "ASE")
    celldisp = convert(celldisp, "distance", units, "ASE")

    if prismobj:
        celldisp = prismobj.vector_to_ase(celldisp)
        cell = prismobj.update_cell(cell)
        if "positions" in atoms_dict:
            atoms_dict["positions"] = prismobj.vector_to_ase(
                atoms_dict["positions"], wrap=True)
        if "scaled_positions" in atoms_dict:
            atoms_dict["scaled_positions"] = prismobj.vector_to_ase(
                atoms_dict["scaled_positions"], wrap=True)

    # Create the Atoms object
    atoms = atomsobj(cell=cell, celldisp=celldisp, pbc=pbc, **atoms_dict)
    # Add velocities if they exist
    if all(col in colnames for col in ["vx", "vy", "vz"]):
        velocities = np.column_stack((data["vx"], data["vy"], data["vz"]))
        if prismobj:
            velocities = prismobj.vector_to_ase(velocities)
        atoms.set_velocities(velocities)

    # Add charges if they exist
    if "q" in colnames:
        charges = data["q"]
        atoms.set_initial_charges(charges)

    # Add forces if they exist
    if all(col in colnames for col in ["fx", "fy", "fz"]):
        forces = np.column_stack((data["fx"], data["fy"], data["fz"]))
        if prismobj:
            forces = prismobj.vector_to_ase(forces)
        # !TODO: use another calculator if available (or move forces
        #        to atoms.property) (other problem: synchronizing
        #        parallel runs)
        calculator = SinglePointCalculator(atoms, energy=0.0, forces=forces)
        atoms.calc = calculator

    # Process the extra columns of fixes, variables, and computes
    for colname in colnames:
        if colname.startswith(("f_", "v_")) or\
           (colname.startswith("c_") and not colname.startswith("c_q[")):
            atoms.new_array(colname, data[colname])
    return atoms


def construct_cell(diagdisp, offdiag):
    """Help function to create an ASE-cell with displacement vector from
    the lammps coordination system parameters.

    :param diagdisp: cell dimension convoluted with the displacement vector
    :param offdiag: off-diagonal cell elements
    :returns: cell and cell displacement vector
    :rtype: tuple
    """
    xlo, xhi, ylo, yhi, zlo, zhi = diagdisp
    xy, xz, yz = offdiag

    # create ase-cell from lammps-box
    xhilo = (xhi - xlo) - abs(xy) - abs(xz)
    yhilo = (yhi - ylo) - abs(yz)
    zhilo = zhi - zlo
    celldispx = xlo - min(0, xy) - min(0, xz)
    celldispy = ylo - min(0, yz)
    celldispz = zlo
    cell = np.array([[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]])
    celldisp = np.array([celldispx, celldispy, celldispz])

    return cell, celldisp


def get_max_index(index):
    if np.isscalar(index):
        return index
    elif isinstance(index, slice):
        return index.stop if (index.stop is not None) else float("inf")


def _process_timestep(args):
    data_bytes, kwargs = args
    print(data_bytes)
    bytes_stream = io.BytesIO(data_bytes) if isinstance(data_bytes,
                                                        bytes) else data_bytes

    timestep_header = bytes_stream.readline().strip()
    if not timestep_header.startswith(b'ITEM: TIMESTEP'):
        raise ValueError("Expected 'ITEM: TIMESTEP' line is missing or invalid")

    # The actual timestep
    bytes_stream.readline()

    # Read the number of atoms
    natoms_header = bytes_stream.readline().strip()
    if not natoms_header.startswith(b'ITEM: NUMBER OF ATOMS'):
        raise ValueError(
            "Expected 'ITEM: NUMBER OF ATOMS' line is missing or invalid")
    bytes_stream.readline()

    # Read the box bounds
    bounds_header = bytes_stream.readline().strip()
    if not bounds_header.startswith(b'ITEM: BOX BOUNDS'):
        raise ValueError(
            "Expected 'ITEM: BOX BOUNDS' line is missing or invalid")

    bounds_data = [list(
        map(float, bytes_stream.readline().strip().split())) for _ in range(3)]

    # Read the atom data header
    atoms_header = bytes_stream.readline().strip()
    if not atoms_header.startswith(b'ITEM: ATOMS'):
        raise ValueError("Expected 'ITEM: ATOMS' line is missing or invalid")
    colnames = atoms_header.split()[2:]

    # Determine the data types for each column
    dtype_list = []
    decoded_colnames = []
    for colname in colnames:
        decoded_colname = colname.decode()
        decoded_colnames.append(decoded_colname)
        if decoded_colname == 'id' or decoded_colname == 'type':
            dtype_list.append((decoded_colname, int))
        elif decoded_colname == 'element':
            # 'U10' for string with maximum length 10
            dtype_list.append((decoded_colname, 'U10'))
        else:
            dtype_list.append((decoded_colname, float))

    # Read the data directly into the numpy array using numpy.loadtxt
    data = np.loadtxt(bytes_stream, dtype=dtype_list)

    pbc = bounds_header.split(b' ')[3:]
    pbc = [bc.decode() for bc in pbc]
    pbc = [bc == 'pp' for bc in pbc]

    # Construct the cell and celldisp
    celldata = np.array(bounds_data)
    diagdisp = celldata[:, :2].reshape(6, 1).flatten()
    offdiag = celldata[:, 2] if celldata.shape[1] > 2 else (0.0,) * 3
    cell, celldisp = construct_cell(diagdisp, offdiag)

    # Convert data to Atoms object
    atoms = lammps_data_to_ase_atoms_typed(
        data, decoded_colnames, cell, celldisp, pbc, atomsobj=Atoms, **kwargs)
    print(atoms.get_atomic_numbers())
    return atoms


def read_lammps_dump_text(fileobj, index=-1, **kwargs):
    """Process cleartext lammps dumpfiles

    :param fileobj: filestream providing the trajectory data
    :param index: integer or slice object (default: get the last timestep)
    :returns: list of Atoms objects
    :rtype: list
    """
    # If string.IO or file-like object
    if isinstance(fileobj, str):
        with open(fileobj, "rb") as f:
            return read_lammps_dump_text(f, index, **kwargs)

    # Convert the input stream to bytes if necessary
    if isinstance(fileobj, io.TextIOWrapper):
        data = fileobj.buffer.read()
    elif isinstance(fileobj, io.StringIO):
        data = fileobj.read().encode()
    else:
        data = fileobj.read()
    # Find the positions of all timesteps in the data
    pattern = re.compile(rb'ITEM: TIMESTEP\n')
    timestep_positions = [m.start() for m in pattern.finditer(data)]
    num_timesteps = len(timestep_positions)

    indices = _get_indices(index, num_timesteps)

    # Create a multiprocessing pool
    pool = multiprocessing.Pool()

    # Read the data for each timestep and send it to the process pool
    timestep_data = []
    for i in indices:
        current_start = timestep_positions[i]
        if i < num_timesteps - 1:
            current_end = timestep_positions[i + 1]
        else:
            current_end = len(data)
        timestep_data.append((data[current_start:current_end], kwargs))

    # Process the timesteps in parallel
    results = pool.map(_process_timestep, timestep_data)

    # Close the pool
    pool.close()
    pool.join()
    print(results)
    if isinstance(index, slice):
        return results
    else:
        return results[0]


def _get_indices(index, num_timesteps):
    if isinstance(index, slice):
        start, stop, step = index.indices(num_timesteps)
        return np.arange(start, stop, step)
    elif index < 0:
        return [num_timesteps - 1]
    elif index is None:
        return np.arange(num_timesteps)
    else:
        return [index]


def read_lammps_dump_binary(
    fileobj, index=-1, colnames=None, intformat="SMALLBIG", **kwargs
):
    """Read binary dump-files (after binary2txt.cpp from lammps/tools)

    :param fileobj: file-stream containing the binary lammps data
    :param index: integer or slice object (default: get the last timestep)
    :param colnames: data is columns and identified by a header
    :param intformat: lammps support different integer size.  Parameter set \
    at compile-time and can unfortunately not derived from data file
    :returns: list of Atoms-objects
    :rtype: list
    """
    # depending on the chosen compilation flag lammps uses either normal
    # integers or long long for its id or timestep numbering
    # !TODO: tags are cast to double -> missing/double ids (add check?)
    tagformat, bigformat = dict(
        SMALLSMALL=("i", "i"), SMALLBIG=("i", "q"), BIGBIG=("q", "q")
    )[intformat]

    index_end = get_max_index(index)

    # Standard columns layout from lammpsrun
    if not colnames:
        colnames = ["id", "type", "x", "y", "z",
                    "vx", "vy", "vz", "fx", "fy", "fz"]

    images = []

    # wrap struct.unpack to raise EOFError
    def read_variables(string):
        obj_len = struct.calcsize(string)
        data_obj = fileobj.read(obj_len)
        if obj_len != len(data_obj):
            raise EOFError
        return struct.unpack(string, data_obj)

    while True:
        try:
            # Assume that the binary dump file is in the old (pre-29Oct2020)
            # format
            magic_string = None

            # read header
            ntimestep, = read_variables("=" + bigformat)

            # In the new LAMMPS binary dump format (version 29Oct2020 and
            # onward), a negative timestep is used to indicate that the next
            # few bytes will contain certain metadata
            if ntimestep < 0:
                # First bigint was actually encoding the negative of the format
                # name string length (we call this 'magic_string' to
                magic_string_len = -ntimestep

                # The next `magic_string_len` bytes will hold a string
                # indicating the format of the dump file
                magic_string = b''.join(read_variables(
                    "=" + str(magic_string_len) + "c"))

                # Read endianness (integer). For now, we'll disregard the value
                # and simply use the host machine's endianness (via '='
                # character used with struct.calcsize).
                #
                # TODO: Use the endianness of the dump file in subsequent
                #       read_variables rather than just assuming it will match
                #       that of the host
                endian, = read_variables("=i")

                # Read revision number (integer)
                revision, = read_variables("=i")

                # Finally, read the actual timestep (bigint)
                ntimestep, = read_variables("=" + bigformat)

            n_atoms, triclinic = read_variables("=" + bigformat + "i")
            boundary = read_variables("=6i")
            diagdisp = read_variables("=6d")
            if triclinic != 0:
                offdiag = read_variables("=3d")
            else:
                offdiag = (0.0,) * 3
            size_one, = read_variables("=i")

            if len(colnames) != size_one:
                raise ValueError("Provided columns do not match binary file")

            if magic_string and revision > 1:
                # New binary dump format includes units string,
                # columns string, and time
                units_str_len, = read_variables("=i")

                if units_str_len > 0:
                    # Read lammps units style
                    _ = b''.join(
                        read_variables("=" + str(units_str_len) + "c"))

                flag, = read_variables("=c")
                if flag != b'\x00':
                    # Flag was non-empty string
                    time, = read_variables("=d")

                # Length of column string
                columns_str_len, = read_variables("=i")

                # Read column string (e.g., "id type x y z vx vy vz fx fy fz")
                _ = b''.join(read_variables("=" + str(columns_str_len) + "c"))

            nchunk, = read_variables("=i")

            # lammps cells/boxes can have different boundary conditions on each
            # sides (makes mainly sense for different non-periodic conditions
            # (e.g. [f]ixed and [s]hrink for a irradiation simulation))
            # periodic case: b 0 = 'p'
            # non-peridic cases 1: 'f', 2 : 's', 3: 'm'
            pbc = np.sum(np.array(boundary).reshape((3, 2)), axis=1) == 0

            cell, celldisp = construct_cell(diagdisp, offdiag)

            data = []
            for _ in range(nchunk):
                # number-of-data-entries
                n_data, = read_variables("=i")
                # retrieve per atom data
                data += read_variables("=" + str(n_data) + "d")
            data = np.array(data).reshape((-1, size_one))

            # map data-chunk to ase atoms
            out_atoms = lammps_data_to_ase_atoms(
                data=data,
                colnames=colnames,
                cell=cell,
                celldisp=celldisp,
                pbc=pbc,
                **kwargs
            )

            images.append(out_atoms)

            # stop if requested index has been found
            if len(images) > index_end >= 0:
                break

        except EOFError:
            break

    return images[index]
