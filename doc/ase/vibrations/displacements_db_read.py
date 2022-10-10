import ase.db
import ase.io
from ase.vibrations.finite_diff import read_axis_aligned_db

name = 'ethanol'

for xc in ('PBE', 'revPBE'):
    ref_atoms = ase.io.read(f'{name}-{xc}.extxyz')

    # Read data: reduce required precision due as these EXTXYZ only have 8.d.p.
    vibrations = read_axis_aligned_db('displacements.db', ref_atoms=ref_atoms,
                                      metadata={'xc': xc, 'name': name},
                                      threshold=1e-6)

    print(f'XC = {xc}')
    print(vibrations.tabulate())
