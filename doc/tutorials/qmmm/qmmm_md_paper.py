import numpy as np
from ase import units
from ase.io import read
from ase.md import Langevin
from ase.calculators.qmmm import (EIQMMM, Embedding, 
                                  LJInteractionsGeneral)
from ase.calculators.combine_mm import CombineMM
from ase.calculators.counterions import AtomicCounterIon as ACI
from ase.calculators.tip4p import TIP4P, epsilon0, sigma0
from ase.calculators.orca import ORCA
from ase.constraints import Hookean 
from rigid_water import rigid

# -------------------------------------------------- Definitions
tag = 'qmmm_md'  # Prefix for output files

# Indices of atoms in QM subsystem
qmidx = list(range(13))  

# Read in simulation box
atoms = read('neutralized.traj')


# -------------------------------------------------- Counterions
# TIP4P LJ Parameters
sigW = np.array([sigma0, 0, 0 ])
epsW = np.array([epsilon0, 0, 0 ])

# K+ LJs:  10.1021/jp8001614
sigK = 2 * 1.590 / 2**(1 / 6.)
epsK = 0.2794651 * units.kcal / units.mol

# Define mmcalc object for EIQMMM calculator. 
# in this sub-atoms object, the calc only sees K+ and water, so
# K+ indices here are [0..3]
mmcalc = CombineMM([0, 1, 2, 3], 
                   1, 3,  # atoms per 'molecule' of each subgroup
                   ACI(1, epsK, sigK),  # Counterion calculator
                   TIP4P(),  # Water calculator
                   [sigK], [epsK],  # LJ params for subgroup 1
                   sigW, epsW)  # LJ params for subgroup 2

# ------------------------------------------------ Lennard-Jones
# MM: K+ and TIP4P as a tuple:
sigma_mm = (np.array([sigK]),  sigW)
epsilon_mm = (np.array([epsK]),  epsW)

# QM: UFF: 10.1021/ja00051a040
epsilonFe = 0.013 * units.kcal / units.mol
sigmaFe = 2 * 2.912 / 2**(1 / 6.) 
epsilonN_qm = 0.069 * units.kcal / units.mol
sigmaN_qm = 3.260
epsilonC_qm = 0.105 * units.kcal / units.mol
sigmaC_qm = 3.431
ljparams_qm = {'Fe': (epsilonFe, sigmaFe),
               'N': (epsilonN_qm, sigmaN_qm),
               'C': (epsilonC_qm, sigmaC_qm)}

# Make numpy arrays with 1 value per QM atom:
epsilon_qm = np.array([ljparams_qm[a.symbol][0] 
                       for a in atoms[qmidx]])
sigma_qm = np.array([ljparams_qm[a.symbol][1] 
                     for a in atoms[qmidx]])

# Set LJ parameters
lj = LJInteractionsGeneral(sigma_qm, epsilon_qm,
                           sigma_mm, epsilon_mm, len(qmidx))


# -------------------------------------------------- Constraints
# Hookean constraints to keep the complex centered
# and the counterions away during re-equillibration
spring = 500 * units.kcal / units.mol
hook_qm = Hookean(0, atoms[0].position, spring, 4)
hooks = [hook_qm]
for h in range(13, 17):  # hook counterions
    hooks.append(Hookean(h, atoms[h].position, spring, 1))

# Rigid water constraints object
water_rigid = rigid(atoms, qmidx, num_cts=4)

# Set constraints
atoms.constraints = tuple([water_rigid] + hooks)

# --------------------------------------------- QM/MM Calculator
obs = '%scf Convergence verytight '+\
      '\nmaxiter 300 end\n\n %pal nprocs 16 end\n\n'

atoms.calc = EIQMMM(selection=qmidx,  # Define QM subsystem
                    qmcalc=ORCA(label=tag,
                                charge=-4, 
                                orcasimpleinput='BLYP def2-SVP',
                                orcablocks=obs),
                    mmcalc=mmcalc,
                    embedding=Embedding(),  # Point charge Vext
                    interaction=lj)  #  # LJ interaction

# ----------------------------------------------------- Dynamics
md = Langevin(atoms, 2.0 * units.fs, temperature=300 * units.kB,
              friction=0.01, logfile=tag + '.log', 
              trajectory=tag + '.traj', loginterval=1)

md.run(20000)

