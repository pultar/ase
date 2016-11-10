import numpy as np
import scipy  # skip test early if no scipy
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.eos import EquationOfState as EOS, eos_names
scipy  # silence pyflakes

b = bulk('Al', 'fcc', a=4.0, orthorhombic=True)
b.set_calculator(EMT())
cell = b.get_cell()

volumes = []
energies = []
for x in np.linspace(0.98, 1.01, 5):
    b.set_cell(cell * x, scale_atoms=True)
    volumes.append(b.get_volume())
    energies.append(b.get_potential_energy())

results = []
for name in eos_names:
    if name == 'antonschmidt':
        # Someone should fix this!
        continue
    eos = EOS(volumes, energies, name)
    v, e, b = eos.fit()
    print('{0:20} {1:.8f} {2:.8f} {3:.8f} '.format(name, v, e, b))
    assert abs(v - 3.18653628e+01) < 4e-4
    assert abs(e - -9.71501875e-03) < 1e-6
    assert abs(b - 2.44867454e-01) < 3e-4
    results.append((v, e, b))

print(np.ptp(results, 0))
print(np.mean(results, 0))
