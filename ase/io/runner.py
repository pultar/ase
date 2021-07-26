"""Implementation of reader and writer for RuNNer input.data files.

The RuNNer Neural Network Energy Representation is a framework for the 
construction of high-dimensional neural network potentials developed in the 
group of Prof. Dr. Jörg Behler at Georg-August-Universität Göttingen.
input.data files contain all structural information required for training a
neural network potential.   

See https://runner.pages.gwdg.de/runner/reference/files/#inputdata

"""

import numpy as np
from ase.atoms import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.calculators.calculator import PropertyNotImplementedError


class FileFormatError(Exception):
    """Raised if a format mistake is encountered in the input.data file."""
    pass


def reset_structure():
    r"""Reset per-structure arrays and variables while reading input.data."""
    
    symbols = []
    positions = []
    cell = []     
    charges = []
    magmoms = []
    forces = []
    periodic = np.array([False, False, False])
    totalenergy = []
    totalcharge = []
    latticecount = 0
    
    return (symbols, positions, cell, charges, magmoms, forces, periodic, 
            totalenergy, totalcharge, latticecount)


def read_runner(fileobj, index):
    r"""Parse all structures within a RuNNer input.data file.
    
    Parameters
    ----------
    fileobj : fileobj
        Python file object with the target input.data file.
    index : int
        The slice of structures which should be returned.
        
    Returns
    --------
    Atoms: ase.atoms.Atoms object
        All information about the structures in input.data (within `index`), 
        including symbols, positions, atomic charges, and cell lattice. Every
        `Atoms` object has a calculator `calc` attached to it with additional
        information on the total energy, atomic forces, and **total charge,
        saved as the magnetic moment**.  
    
    Raises
    -------
    FileFormatError : exception
        Raised when a format error in the `fileobj` is encountered.
    
    References
    ----------
    Detailed information about the RuNNer input.data file format can be found
    in the program's 
    [documentation](https://runner.pages.gwdg.de/runner/reference/files/#inputdata)

    """
    
    # Set a list of all valid keywords in RuNNer input.data files.
    RUNNERDATA_KEYWORDS = [
        'begin',               
        'comment',
        'lattice',
        'atom',
        'charge',
        'energy',
        'end'
    ]

    # Read in the file.
    lines = fileobj.readlines()
    
    # Container for all images in the file.
    images = []
    
    # Set all per-structure containers and variables.
    (symbols, positions, cell, charges, magmoms, forces, periodic, 
     totalenergy, totalcharge, latticecount) = reset_structure()

    for lineidx, line in enumerate(lines):
        # Jump over blank lines.
        if line.strip() == "":
            continue

        # First word of each line must be a valid keyword.
        keyword = line.split()[0]

        if keyword not in RUNNERDATA_KEYWORDS:
            raise FileFormatError(
                f"File {fileobj.name} is not a valid input.data file. "
                f"Illegal keyword '{keyword}' in line {lineidx+1}."
            )
        
        # 'begin' marks the start of a new structure.
        if keyword == 'begin':
            # Check if anything appeared in between this new structure and the
            # previous one, e.g. a poorly formatted third structure.
            if any(symbols):
                raise FileFormatError(
                    f"Structure {len(images)} beginning in line {lineidx+1}" 
                    f"appears to be preceded by a poorly formatted structure."
                )
                
            # Set all per-structure containers and variables.
            (symbols, positions, cell, charges, magmoms, forces, periodic, 
             totalenergy, totalcharge, latticecount) = reset_structure()
        
        # Read one atom.
        elif keyword == 'atom':
            x, y, z, symbol, charge, magmom, fx, fy, fz = line.split()[1:10]
            
            # Convert and process.
            symbol = symbol.lower().capitalize()
            
            # Append to related arrays.
            symbols.append(symbol)
            positions.append([float(x), float(y), float(z)])
            charges.append(float(charge))
            forces.append([float(fx), float(fy), float(fz)])
            magmoms.append(float(magmom))
        
        # Read one cell lattice vector.
        elif keyword == 'lattice':
            lx, ly, lz = line.split()[1:4]
            cell.append([float(lx), float(ly), float(lz)])
            
            periodic[latticecount] = True
            latticecount += 1
        
        # Read the total energy of the structure.
        elif keyword == 'energy':
            energy = float(line.split()[1])
            totalenergy.append(energy)
        
        # Read the total charge of the structure.
        elif keyword == 'charge':
            charge = float(line.split()[1])
            totalcharge.append(charge)
            
        # 'end' statement marks the end of a structure.
        elif keyword == 'end':            
            # Check if there is at least one atom in the structure.
            if len(symbols) == 0:
                raise FileFormatError(
                    f"Structure {len(images)} ending in line {lineidx+1} does" 
                    f"not contain any atoms."
                )

            # Check if a charge has been specified for the structure.
            if len(totalcharge) != 1:
                raise FileFormatError(
                    f"Structure {len(images)} ending in line {lineidx+1} does"
                    f"not have exactly one total charge."
                )

            # Check if an energy has been specified for the structure.
            if len(totalenergy) != 1:
                raise FileFormatError(
                    f"Structure {len(images)} ending in line {lineidx+1} does"
                    f"not have exactly one total energy."
                )

            # If all checks clear, set the atoms object.
            atoms = Atoms(
                symbols=symbols, 
                positions=positions
            )
            
            # Optional: set periodic structure properties.
            if periodic.any():
                # Check if there is exactly three lattice vectors.
                if len(cell) != 3:
                    raise FileFormatError(
                        f"Structure {len(images)} ending in line {lineidx+1}" 
                        f"does not contain any atoms."
                    )
            
                atoms.set_cell(cell)
                atoms.set_pbc(periodic)
            
            # Optional: set magnetic moments for each atom.
            if any(magmoms):
                atoms.set_initial_magnetic_moments(magmoms)
            
            # Optional: set atomic charges.
            if any(charges):
                atoms.set_initial_charges(charges)

            # CAUTION: The calculator has to be attached at the very end, 
            # otherwise, it is overwritten by `set_cell()`, `set_pbc()`, ...
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=totalenergy[0],
                forces=forces,
                # The total charge can only be stored in a SinglePointCalculator
                # by modifying ase.calculators.calculator.all_properties
                # so this line does not work on its own.
                #totalcharge=totalcharge[0]
            )

            # Finally, append the structure to the list of all structures.
            images.append(atoms)

            # Set all per-structure containers and variables.
            (symbols, positions, cell, charges, magmoms, forces, periodic, 
             totalenergy, totalcharge, latticecount) = reset_structure()

    # Check whether there are any structures in the file.
    if len(images) == 0:
        raise FileFormatError(
            f"File {fileobj.name} does not contain any structures."
        )
    
    for atoms in images[index]:
        yield atoms


def write_runner(fileobj, images, comment='', fmt='%22.12f'):
    r"""Write series of ASE Atoms to a RuNNer input.data file.
    
    Parameters
    ----------
    fileobj : fileobj
        Python file object with the target input.data file.
    images : array-like
        List of `Atoms` objects.
    comment : str
        A comment message to be added to each structure.
    
    Raises
    -------
    ValueError : exception
        Raised if the comment line contains newline characters.
    
    """

    comment = comment.rstrip()
    if '\n' in comment:
        raise ValueError('Comment line should not have line breaks.')
    
    for atoms in images:
        fileobj.write('begin\n')
        
        if comment != '':
            fileobj.write('comment %s\n' % (comment))
        
        # Write lattice vectors.
        for vector in atoms.cell:
            lx, ly, lz = vector
            fileobj.write('lattice %s %s %s\n' % (fmt % lx, fmt % ly, fmt % lz))
        
        # Write atoms.
        atom = zip(
            atoms.symbols, 
            atoms.positions, 
            atoms.get_initial_charges(),
            atoms.get_initial_magnetic_moments(),
            atoms.get_forces()
        )
        
        for s, (x, y, z), q, m, (fx, fy, fz) in atom:
            fileobj.write(
                'atom %s %s %s %-2s %s %s %s %s %s\n' 
                % (fmt % x, fmt % y, fmt % z, s, fmt % q, fmt % m, 
                   fmt % fx, fmt % fy, fmt % fz)
            )
        
        fileobj.write('energy %s\n' % (fmt % atoms.get_potential_energy()))

        # Exception handling is necessary as long as totalcharge property is not
        # finalized.
        try:    
            fileobj.write('charge %s\n' % (fmt % atoms.calc.get_property('totalcharge')))
        except PropertyNotImplementedError:
            fileobj.write('charge %s\n' % (fmt % 0.0))        
        fileobj.write('end\n')