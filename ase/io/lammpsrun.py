from ase.lammpsatoms import LammpsAtoms
from ase.lammpsquaternions import LammpsQuaternions
from ase.calculators.singlepoint import SinglePointCalculator
from ase.parallel import paropen
from ase.utils import basestring
from collections import deque


def read_lammps_dump(fileobj, index=-1, order=True, Z_of_type=None,
                    atomsobj=LammpsAtoms, lammps_data=None):
    """Method which reads a LAMMPS dump file.

    order: Order the particles according to their id. Might be faster to
    switch it off.
    lammps_data: lammps data file which was used to start the Lammps
    simulation
    """
    lammps_props = ['id',
                    'type',
                    'mol-id',
                    'masses',
                    'mmcharge',
                    'bonds',
                    'angles',
                    'dihedrals',
                    'impropers']

    if isinstance(fileobj, basestring):
        f = paropen(fileobj)
    else:
        f = fileobj

    # load everything into memory
    lines = deque(f.readlines())

    natoms = 0
    images = []

    while len(lines) > natoms:
        line = lines.popleft()

        if 'ITEM: TIMESTEP' in line:
            lo = []
            hi = []
            tilt = []
            id = []
            types = []
            pbc = [False, False, False]
            positions = []
            scaled_positions = []
            velocities = []
            forces = []
            quaternions = []

        if 'ITEM: NUMBER OF ATOMS' in line:
            line = lines.popleft()
            natoms = int(line.split()[0])
            
        if 'ITEM: BOX BOUNDS' in line:
            # save labels behind "ITEM: BOX BOUNDS" in
            # triclinic case (>=lammps-7Jul09)
            tilt_items = line.split()[3:]
            for i in range(3):
                line = lines.popleft()
                fields = line.split()
                lo.append(float(fields[0]))
                hi.append(float(fields[1]))
                if (len(fields) >= 3):
                    tilt.append(float(fields[2]))

            #check pbc
            pbc_items = tilt_items[-3:]
            for i in range(3):
                pbc[i] = pbc_items[i] == 'pp' or pbc_items[i] == 'p'

            # determine cell tilt (triclinic case!)
            if (len(tilt) >= 3):
                # for >=lammps-7Jul09 use labels behind
                # "ITEM: BOX BOUNDS" to assign tilt (vector) elements ...
                if (len(tilt_items) >= 3):
                    xy = tilt[tilt_items.index('xy')]
                    xz = tilt[tilt_items.index('xz')]
                    yz = tilt[tilt_items.index('yz')]
                # ... otherwise assume default order in 3rd column
                # (if the latter was present)
                else:
                    xy = tilt[0]
                    xz = tilt[1]
                    yz = tilt[2]
            else:
                xy = xz = yz = 0
            xhilo = (hi[0] - lo[0]) - (xy**2)**0.5 - (xz**2)**0.5
            yhilo = (hi[1] - lo[1]) - (yz**2)**0.5
            zhilo = (hi[2] - lo[2])
            if xy < 0:
                if xz < 0:
                    celldispx = lo[0] - xy - xz
                else:
                    celldispx = lo[0] - xy
            else:
                celldispx = lo[0]
            celldispy = lo[1]
            celldispz = lo[2]

            cell = [[xhilo, 0, 0], [xy, yhilo, 0], [xz, yz, zhilo]]
            celldisp = [[celldispx, celldispy, celldispz]]

        def add_quantity(fields, var, labels):
            for label in labels:
                if label not in atom_attributes:
                    return
            var.append([float(fields[atom_attributes[label]])
                        for label in labels])
                
        if 'ITEM: ATOMS' in line:
            # (reliably) identify values by labels behind
            # "ITEM: ATOMS" - requires >=lammps-7Jul09
            # create corresponding index dictionary before
            # iterating over atoms to (hopefully) speed up lookups...
            atom_attributes = {}
            for (i, x) in enumerate(line.split()[2:]):
                atom_attributes[x] = i
            for n in range(natoms):
                line = lines.popleft()
                fields = line.split()
                id.append(int(fields[atom_attributes['id']]))
                types.append(int(fields[atom_attributes['type']]))
                add_quantity(fields, positions, ['x', 'y', 'z'])
                add_quantity(fields, scaled_positions, ['xs', 'ys', 'zs'])
                add_quantity(fields, velocities, ['vx', 'vy', 'vz'])
                add_quantity(fields, forces, ['fx', 'fy', 'fz'])
                add_quantity(fields, quaternions, ['c_q[1]', 'c_q[2]',
                                                   'c_q[3]', 'c_q[4]'])

            if order:
                def reorder(inlist):
                    if not len(inlist):
                        return inlist
                    outlist = [None] * len(id)
                    for i, v in zip(id, inlist):
                        outlist[i - 1] = v
                    return outlist
                types = reorder(types)
                positions = reorder(positions)
                scaled_positions = reorder(scaled_positions)
                velocities = reorder(velocities)
                forces = reorder(forces)
                quaternions = reorder(quaternions)

            if Z_of_type is None:
                numbers = types
            else:
                numbers = [Z_of_type[i] for i in types]

            if len(quaternions):
                images.append(LammpsQuaternions(symbols=numbers,
                                          positions=positions, pbc=pbc,
                                          cell=cell, celldisp=celldisp,
                                          quaternions=quaternions))
            elif len(positions):
                images.append(atomsobj(
                    symbols=numbers, positions=positions, pbc=pbc,
                    celldisp=celldisp, cell=cell))
            elif len(scaled_positions):
                images.append(atomsobj(
                    symbols=numbers, scaled_positions=scaled_positions, pbc=pbc,
                    celldisp=celldisp, cell=cell))

            images[-1].set_array('type', types, int)
            if lammps_data:
                '''it assumes lammps_data has id = np.arange(len(self)) + 1'''
                for prop in lammps_props:
                    if lammps_data.has(prop) and not images[-1].has(prop):
                        _ = lammps_data.get_array(prop)
                        images[-1].set_array(prop,
                                             [_[i - 1] for i in id],
                                             _.dtype)

            if len(velocities):
                images[-1].set_velocities(velocities)
            if len(forces):
                calculator = SinglePointCalculator(images[-1],
                                                   energy=0.0, forces=forces)
                images[-1].set_calculator(calculator)

    return images[index]
