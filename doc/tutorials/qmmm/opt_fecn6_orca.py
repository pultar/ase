from ase.optimize import FIRE
from ase.io import read
from ase.calculators.orca import ORCA

atoms = read('FeCN6.xyz')

tag = 'FeCN6_opt'

obs = '%scf Convergence verytight '+\
      '\nmaxiter 300 end\n\n %pal nprocs 16 end\n\n'

atoms.calc = ORCA(label=tag,
                  charge=-4,
                  orcasimpleinput='BLYP def2-SVP',
                  orcablocks=obs) 
             
opt = FIRE(atoms, logfile=tag + '.log', trajectory=tag + '.traj')
opt.run(fmax=0.05)
