import matplotlib.pyplot as plt
import numpy as np

from ase.build import add_adsorbate, fcc100
from ase.calculators.emt import EMT
from ase.neb import NEB, NEBTools
from ase.optimize import BFGS
from ase.rneb import RNEB

# again create the Cu slab
slab = fcc100('Cu', [3, 3, 3], vacuum=5)
slab.calc = EMT()
qn = BFGS(slab, logfile=None)
qn.run(fmax=0.01)

initial_unrelaxed = slab.copy()
add_adsorbate(initial_unrelaxed, 'Cu', 1.7, 'hollow')
final_unrelaxed = slab.copy()
add_adsorbate(final_unrelaxed, 'Cu', 1.7, 'hollow')
ps = final_unrelaxed.get_positions()
ps[-1] = ps[-1] + np.array([np.linalg.norm(slab.cell[0, :]) / 3, 0, 0])
final_unrelaxed.set_positions(ps)

initial_relaxed = initial_unrelaxed.copy()
initial_relaxed.calc = EMT()
qn = BFGS(initial_relaxed, logfile=None)
qn.run(fmax=0.01)

# We know from above that the path is reflective
# Otherwise we could have used rneb.reflect_path() to check this

# get the final image by symmetry operations
rneb = RNEB(slab, logfile=None)
final_relaxed = rneb.get_final_image(
    init=initial_unrelaxed,
    init_relaxed=initial_relaxed,
    final=final_unrelaxed)

# create the path by adding a single intermediate image
images = [initial_relaxed]
image = initial_relaxed.copy()
image.set_calculator(EMT())
images.append(image)
images.append(final_relaxed)

neb = NEB(images)
neb.interpolate()

# run the reflective middle image NEB (RMI-NEB)
qn = BFGS(neb, logfile=None)
qn.run(fmax=0.01)
middle_image = images[1]

# plot the RMI-NEB barrier
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True)
nebtools = NEBTools(images)
nebtools.plot_band(ax=ax1)
ax1.set_title('RMI-NEB')

# subsequently a NEB on half of the path
images = [initial_relaxed]
for i in range(2):
    image = initial_relaxed.copy()
    image.set_calculator(EMT())
    images.append(image)
# now the middle image is the final image
images.append(middle_image)

neb = NEB(images)
neb.interpolate()
qn = BFGS(neb, logfile=None)
qn.run(fmax=0.01)

# finally plot the CIR-NEB barrier
nebtools = NEBTools(images)
nebtools.plot_band(ax=ax2)
ax2.set_ylabel('')
ax2.set_title('CIR-NEB')
fig.savefig('rmineb.png')
