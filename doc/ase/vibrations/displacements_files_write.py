import ase.build
from gpaw import GPAW
from ase.vibrations.finite_displacements import (
    get_displacements_with_identities)
from ase.optimize import LBFGS
from pathlib import Path

atoms = ase.build.molecule('CH3CH2OH', vacuum=5.)

# Optimise geometry
atoms.calc = GPAW(mode='lcao', basis='dzp')
dyn = LBFGS(atoms)
dyn.run(fmax=1e-3)

atoms.write('ethanol_gpaw_opt.xyz')

disp_dir = Path('.') / 'ethanol_disp'
disp_dir.mkdir(exist_ok=True)
for disp, label in get_displacements_with_identities(atoms):
    disp.write(disp_dir / f'{label}.xyz')
