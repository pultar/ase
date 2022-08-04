import ase.build
from ase.calculators.mopac import MOPAC
from ase.vibrations.finite_diff import (get_displacements,
                                        read_axis_aligned_forces)
from ase.optimize import LBFGS, FIRE

atoms = ase.build.molecule('CH3CH2OH')

# Optimise geometry
atoms.calc = MOPAC(task='1SCF DISP GRADIENTS', relscf=1e-8)
dyn = LBFGS(atoms)
dyn.run(fmax=1e-3)

# Secondary optimisation at higher precision
atoms.calc = MOPAC(task='1SCF DISP GRADIENTS PRECISE', relscf=1e-12)
dyn = FIRE(atoms)
dyn.run(fmax=1e-4)

displacements = get_displacements(atoms)
# Attach a calculator so each displacment so that get_forces() will work
for disp in displacements:
    disp.calc = MOPAC(task='1SCF DISP GRADIENTS PRECISE', relscf=1e-14)

# Call get_forces() on each Atoms, and compare structures to build Hessian
print("Calculating vibrations...")
vibrations = read_axis_aligned_forces(atoms, displacements)
print("Calculated vibrational frequencies:")
print(vibrations.tabulate())
vibrations.write('ethanol_mopac_vibs.json')
vibrations.write_jmol('ethanol_mopac_vibs.xyz')
