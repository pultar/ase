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

# There are different types of symmetry operations. For them to be
# applicable in the NEB path, they have to be reflection operations.
refl_sym_ops = rneb.reflect_path(images, sym=all_sym_ops)
print(f"{len(refl_sym_ops)}/{len(all_sym_ops)} are reflection operations")

# Now we obtain the reflective images that will be fed to NEB so that
# symmetry will be used.
refl_images = rneb.get_reflective_path(images, all_sym_ops)
neb = NEB(images)
qn = BFGS(neb, logfile=None, trajectory='neb.traj')
qn.run(fmax=0.05)

# Create a figure like that coming from ASE-GUI.
nebtools = NEBTools(images)
fig = nebtools.plot_band()
fig.savefig('reflective_path.png')
