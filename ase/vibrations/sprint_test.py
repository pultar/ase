import ase.build
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations.finitedifference import get_displacements, read_forces_direct
import ase.io

atoms = ase.build.molecule('C2H6')
atoms.calc = EMT()
opt = BFGS(atoms)
opt.run(fmax=1e-5)

displacements = get_displacements(atoms, delta=0.01, direction='central')

for disp in displacements:
    disp.calc = EMT()
    _ = disp.get_forces()

ase.io.write('my_displacements.traj', displacements, format='traj')
vibs = read_forces_direct(atoms, displacements, method='standard',
                          use_equilibrium_forces=None)

print(vibs.tabulate())


more_displacements = get_displacements(atoms, delta=0.02, direction='central')
for disp in more_displacements:
    disp.calc = EMT()
    _ = disp.get_forces()
    
vibs = read_forces_direct(atoms, displacements + more_displacements,
                          method='standard', use_equilibrium_forces=None)
print(vibs.tabulate())
