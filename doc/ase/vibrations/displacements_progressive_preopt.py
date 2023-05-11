from ase import Atoms
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.optimize import QuasiNewton
from ase.build import fcc111, add_adsorbate

h = 1.85
d = 1.10

slab = fcc111('Cu', size=(4, 4, 2), vacuum=10.0)
molecule = Atoms('2N', positions=[(0., 0., 0.), (0., 0., d)])

add_adsorbate(slab, molecule, h, 'ontop')
slab.calc = EMT()
dyn = QuasiNewton(slab, trajectory='N2Cu.traj')
dyn.run(fmax=1e-4)

slab.write('opt_slab.extxyz')

