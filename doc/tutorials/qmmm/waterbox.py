from ase import Atoms
from ase.calculators.tip4p import TIP4P, rOH, angleHOH
from ase.md import Langevin
import ase.units as units
import numpy as np
from rigid_water import rigid

# Set up water box at 20 deg C density
x = angleHOH * np.pi / 180 / 2
pos = [[0, 0, 0],
       [0, rOH * np.cos(x), rOH * np.sin(x)],
       [0, rOH * np.cos(x), -rOH * np.sin(x)]]
atoms = Atoms('OH2', positions=pos)

vol = ((18.01528 / 6.022140857e23) / (0.9982 / 1e24))**(1 / 3.)
atoms.set_cell((vol, vol, vol))
atoms.center()

atoms = atoms.repeat((7, 7, 7))
atoms.set_pbc(True)

atoms.constraints = rigid(atoms)

tag = 'tip4p_343mol_equil'
atoms.calc = TIP4P()
md = Langevin(atoms, 1 * units.fs, temperature=300 * units.kB, loginterval=100,
              friction=0.01, logfile=tag + '.log', trajectory=tag + '.traj')

md.run(4000)

# Repeat box and equilibrate further.
tag = 'tip4p_2744mol_equil'
atoms.set_constraint()  # Remove constraints before repeating
atoms = atoms.repeat((2, 2, 2))

atoms.constraints = rigid(atoms)


atoms.calc = TIP4P()
md = Langevin(atoms, 2 * units.fs, temperature=300 * units.kB, loginterval=50,
              friction=0.01, logfile=tag + '.log', trajectory=tag + '.traj')

md.run(2000)
