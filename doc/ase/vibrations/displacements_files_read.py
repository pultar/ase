import ase.io
from ase.vibrations.finite_diff import read_axis_aligned_forces
from pathlib import Path

atoms = ase.io.read('ethanol_gpaw_opt.xyz')

displacements = [ase.io.read(disp_file)
                 for disp_file in
                 Path('./ethanol_forces').glob('*.xyz')]

vibrations = read_axis_aligned_forces(atoms, displacements)
print("Calculated vibrational frequencies:")
print(vibrations.tabulate())
vibrations.write('ethanol_gpaw_vibs.json')
vibrations.write_jmol('ethanol_gpaw_vibs.xyz')
