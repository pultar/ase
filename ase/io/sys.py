"""
IO support for the qb@ll sys format.
The positions and cell dimensions are in Bohrs.

Contributed by Rafi Ullah <rraffiu@gmail.com>
"""

from ase.atoms import Atoms
from ase.units import Bohr

import re

__all__ = ['read_sys', 'write_sys']

def read_sys(fileobj):
    """
    Function to read a qb@ll sys file.
    fileobj: file object
        File to read from.
    """
    line = fileobj.readline().split()
    cell = []
    cell.append([float(line[2])*Bohr, float(line[3])*Bohr, float(line[4])*Bohr])
    cell.append([float(line[5])*Bohr, float(line[6])*Bohr, float(line[7])*Bohr])
    cell.append([float(line[8])*Bohr, float(line[9])*Bohr, float(line[10])*Bohr])
    while True:
        inp = fileobj.tell() # Not sure if there is a better way to skip these lines.
        line = fileobj.readline()
        if 'species' not in line:
            break
    fileobj.seek(inp)
    positions = []
    symbols = []
    reg = re.compile(r'(\d+|\s+)')
    while True:
        line = fileobj.readline()
        if not line:
            break
        # The units column is ignored.
        a, symLabel, spec, x, y, z = line.split()[0:6]
        positions.append([float(x)*Bohr,float(y)*Bohr,float(z)*Bohr])
        sym = reg.split(str(symLabel))
        symbols.append(sym[0])
    atoms = Atoms(symbols=symbols, cell=cell, positions=positions)
    return atoms

def write_sys(fileobj, atoms):
    """
    Function to write a sys file.
    fileobj: file object
        File to which output is written.
    atoms: Atoms object
        Atoms object specifying the atomic configuration.
    """
    fileobj.write('set cell')
    for i in range(3):
        d = atoms.cell[i]/Bohr
        fileobj.write((' {:6f}  {:6f}  {:6f}').format(*d))
    fileobj.write('  bohr\n')

    ch_sym = atoms.get_chemical_symbols()
    atm_nm = atoms.numbers
    a_pos  = atoms.positions
    an     = list(set(atm_nm))

    for i, s in enumerate(set(ch_sym)):
        fileobj.write(('species {}{} {}.xml \n').format(s,an[i],s))
    for i, (S, Z, (x, y, z)) in enumerate(zip(ch_sym, atm_nm, a_pos)):
        fileobj.write(('atom {0:5} {1:5}  {2:12.6f}{3:12.6f}{4:12.6f} bohr\n').\
        format(S+str(i+1),S+str(Z), x/Bohr, y/Bohr, z/Bohr))
