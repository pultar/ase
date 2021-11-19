import ase.build
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations.finite_diff import get_displacements, read_forces_direct
import ase.io

atoms = ase.build.molecule('C2H6')
#atoms = ase.build.bulk('Al') * [3, 3, 3]
atoms.calc = EMT()
opt = BFGS(atoms)
opt.run(fmax=1e-5)

displacements = get_displacements(atoms, delta=0.01, direction='central')

for disp in displacements:
    disp.calc = EMT()
    _ = disp.get_forces()

vibs = read_forces_direct(atoms, displacements, method='standard',
                          use_equilibrium_forces=None)
print(vibs.tabulate())

vibs = read_forces_direct(atoms, displacements, method='frederiksen',
                          use_equilibrium_forces=None)
print(vibs.tabulate())

