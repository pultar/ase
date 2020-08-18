from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.rneb import RNEB
from ase.neb import NEB
from ase.build import fcc100, add_adsorbate
import numpy as np
from ase.neb import NEBTools

# create a Cu slab
slab = fcc100('Cu', [3, 3, 3], vacuum=5)
slab.calc = EMT()
qn = BFGS(slab, logfile=None)
qn.run(fmax=0.01)

initial = slab.copy()
add_adsorbate(initial, 'Cu', 1.7, 'hollow')
final = slab.copy()
add_adsorbate(final, 'Cu', 1.7, 'hollow')
ps = final.get_positions()
ps[-1] = ps[-1] + np.array([np.linalg.norm(slab.cell[0, :])/3, 0, 0])
final.set_positions(ps)

initial.calc = EMT()
qn = BFGS(initial, logfile=None)
qn.run(fmax=0.01)

final.calc = EMT()
qn = BFGS(final, logfile=None)
qn.run(fmax=0.01)

rneb = RNEB(logfile=None)
all_sym_ops = rneb.find_symmetries(slab, initial, final)

images = [initial]
for i in range(5):
    image = initial.copy()
    image.set_calculator(EMT())
    images.append(image)
images.append(final)
neb = NEB(images)
neb.interpolate()
# check that path has reflection symmetry for each image pair
refl_sym_ops = rneb.reflect_path(images, sym=all_sym_ops)

print(f"{len(refl_sym_ops)}/{len(all_sym_ops)} are reflection operations")

neb = NEB(images, sym=True, rotations=refl_sym_ops[1])
qn = BFGS(neb, logfile=None)
qn.run(fmax=0.05)

nebtools = NEBTools(images)
print("NEB Barrier is {:5.3f} eV".format(nebtools.get_barrier()[0]))

from ase.visualize import view
view(images)
