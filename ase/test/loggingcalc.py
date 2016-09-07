import numpy as np

from ase.build import bulk
from ase.calculators.lj import LennardJones
from ase.calculators.loggingcalc import LoggingCalculator
from ase.optimize import FIRE, BFGS

a0 = bulk('Cu', cubic=True)

# perturb the atoms
s = a0.get_scaled_positions()
s[:, 0] *= 0.995
a0.set_scaled_positions(s)
a0.rattle(0.05)

log_calc = LoggingCalculator(LennardJones())

for OPT, label in zip([FIRE, BFGS],
                      ["FIRE", "BFGS"]):
    log_calc.set_label(label)
    atoms = a0.copy()
    atoms.set_calculator(log_calc)
    opt = OPT(atoms)
    opt.run(fmax=1e-3)

log_calc.plot()
# import matplotlib.pyplot as plt
# plt.show()
