import numpy as np

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.neb import NEB, NEBTools
from ase.optimize import BFGS
from ase.rneb import RNEB

# create a Cu slab
slab = fcc100('Cu', [3, 3, 3], vacuum=5)
slab.calc = EMT()
qn = BFGS(slab, logfile=None)
qn.run(fmax=0.01)

# create and relax initial/final image
initial = slab.copy()
add_adsorbate(initial, 'Cu', 1.7, 'hollow')
final = slab.copy()
add_adsorbate(final, 'Cu', 1.7, 'hollow')
ps = final.get_positions()
ps[-1] = ps[-1] + np.array([np.linalg.norm(slab.cell[0, :]) / 3, 0, 0])
final.set_positions(ps)

initial.calc = EMT()
qn = BFGS(initial, logfile=None)
qn.run(fmax=0.01)

final.calc = EMT()
qn = BFGS(final, logfile=None)
qn.run(fmax=0.01)

# Use the RNEB class to find symmetry operations
rneb = RNEB(slab, logfile=None)
all_sym_ops = rneb.find_symmetries(initial, final)

images = [initial]
for i in range(5):
    image = initial.copy()
    image.set_calculator(EMT())
    images.append(image)
images.append(final)
neb = NEB(images)
neb.interpolate()

# check if symmetry operations are also reflection operations
# this is necessary as otherwise the RNEB is not applicable
refl_sym_ops = rneb.reflect_path(images, sym=all_sym_ops)

print(f"{len(refl_sym_ops)}/{len(all_sym_ops)} are reflection operations")

refl_sym_op = refl_sym_ops[1]  # choose any valid operation
neb = NEB(images, reflect_ops=refl_sym_op)
qn = BFGS(neb, logfile=None, trajectory='neb.traj')
qn.run(fmax=0.05)

# Create a figure like that coming from ASE-GUI.
nebtools = NEBTools(images)
fig = nebtools.plot_band()
fig.savefig('reflective_path.png')
