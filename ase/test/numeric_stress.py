from ase.calculators.test import numeric_stress
from ase.calculators.lj import LennardJones
from ase.build import bulk

a = bulk('Cu')
a *= (3, 3, 3)
a.set_calculator(LennardJones())
assert max(abs(numeric_stress(a)-a.get_stress())) < 1e-9