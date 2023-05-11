import ase.io
from ase.calculators.emt import EMT
from ase.vibrations.finite_diff import get_displacements, read_axis_aligned_forces

atoms = ase.io.read('opt_slab.extxyz')
atoms.calc = EMT()

# Get an array of atom indices sorted by distance to atom -1
distances = atoms.get_distances(-1, range(len(atoms)), mic=True)
indices = distances.argsort()

# Get a series of displacements ordered by distance
all_displacements = get_displacements(atoms, indices)
calculated_displacements = []
max_frequencies = []

print("n_atoms  Distance / Å   Max frequency / 1/cm ")
print("-------  ------------   --------------------")

for index, distance in enumerate(distances[indices]):
    # Quit once we are far away from N atom
    if distance > 5.5:
        break

    # Compute a batch of 6 displacements for central differences on next atom
    for displacement in all_displacements[index * 6:(index * 6 + 6)]:

        displacement.calc = EMT()
        calculated_displacements.append(displacement)

    vib_data = read_axis_aligned_forces(calculated_displacements, ref_atoms=atoms, indices=indices[:index + 1])
    max_freq = (vib_data.get_frequencies()[-1]).real

    print(f"{index + 1:6d}  {distance:10.5f}  {max_freq:16.5f}")
