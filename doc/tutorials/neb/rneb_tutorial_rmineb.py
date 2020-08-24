from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.rneb import RNEB
from ase.neb import NEB
from ase.build import fcc100, add_adsorbate
import numpy as np
from ase.neb import NEBTools

# Now run everything with symmetry operations
# create a bulk Al structure
slab = fcc100('Cu', [3, 3, 3], vacuum=5)
slab.calc = EMT()
qn = BFGS(slab, logfile=None)
qn.run(fmax=0.01)

initial_unrelaxed = slab.copy()
add_adsorbate(initial_unrelaxed, 'Cu', 1.7, 'hollow')
final_unrelaxed = slab.copy()
add_adsorbate(final_unrelaxed, 'Cu', 1.7, 'hollow')
ps = final_unrelaxed.get_positions()
ps[-1] = ps[-1] + np.array([np.linalg.norm(slab.cell[0, :])/3, 0, 0])
final_unrelaxed.set_positions(ps)

initial_relaxed = initial_unrelaxed.copy()
initial_relaxed.calc = EMT()
qn = BFGS(initial_relaxed, logfile=None)
qn.run(fmax=0.01)

# We know from above that the path is reflective
# Otherwise we could have used rneb.reflect_path() to check this

# get the final image by symmetry operations
rneb = RNEB(logfile=None)
final_relaxed = rneb.get_final_image(
    orig=slab,
    init=initial_unrelaxed,
    init_relaxed=initial_relaxed,
    final=final_unrelaxed)

# create the path
images = [initial_relaxed]
for i in range(1):
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

nebtools = NEBTools(images)
print("RMI-NEB Barrier is {:5.3f} eV".format(nebtools.get_barrier()[0]))

# subsequently a NEB on half of the path
images = [initial_relaxed]
for i in range(2):
    image = initial_relaxed.copy()
    image.set_calculator(EMT())
    images.append(image)
images.append(middle_image)  # now the middle image is the final image

neb = NEB(images)
neb.interpolate()
qn = BFGS(neb, logfile=None)
qn.run(fmax=0.01)

nebtools = NEBTools(images)
print("NEB Barrier on half of the path is {:5.3f} eV".format(
    nebtools.get_barrier()[0]))

from ase.visualize import view
view(images)

