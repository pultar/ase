from ase.calculators.gaussian import Gaussian
from ase.atoms import Atoms
from ase.optimize.lbfgs import LBFGS
import os


# First test to make sure Gaussian works
calc = Gaussian(method='pbepbe', basis='sto-3g', force='force',
                nproc=1, chk='water.chk', label='water')
calc.clean()

water = Atoms('OHH',
              positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
              calculator=calc)

opt = LBFGS(water)
opt.run(fmax=0.05)

forces = water.get_forces()
energy = water.get_potential_energy()
positions = water.get_positions()

# Then test the IO routines
from ase.io import read
water2 = read('water.log')
forces2 = water2.get_forces()
energy2 = water2.get_potential_energy()
positions2 = water2.get_positions()
#compare distances since positions are different in standard orientation
dist = water.get_all_distances()
dist2 = read('water.log', quantity='structures')[-1].get_all_distances()

assert abs(energy - energy2) < 1e-7
assert abs(forces - forces2).max() < 1e-9
assert abs(positions - positions2).max() < 1e-6
assert abs(dist - dist2).max() < 1e-6



#Test shorter archive section
f = open('tmp.log','w')
f.write("""
 1\1\GINC-TAURUSI6288\SP\UTPSSh\CC-pVTZ\C28H26N2Ni2O8(3)\ROOT\24-May-20
 17\0\\#TPSSh/cc-pVTZ\\\0,3\C,0,-1.113892,2.39848,1.732689\O,0,-0.00181
 6,1.863817,1.357578\O,0,-2.278235,-1.667476,-1.174874\O,0,-0.012538,-1
 .12416,1.537621\O,0,-0.011181,-1.535409,-1.122556\O,0,-2.280221,-1.183
 033,1.655532\H,0,-3.103091,-3.711657,-2.27948\H,0,1.332705,4.021074,-5
 .15464\H,0,1.139766,2.586129,-3.118149\H,0,1.152391,3.754045,2.373701\
 C,0,-0.981146,3.60478,2.613767\C,0,-1.115999,1.48516,-2.076015\C,0,-2.
 122968,4.155009,3.213919\C,0,-2.01044,5.273138,4.040215\C,0,-0.759311,
 5.855577,4.266645\C,0,0.381475,5.313256,3.66613\C,0,0.272525,4.190173,
 2.846374\H,0,-3.089259,3.688318,3.021652\H,0,-2.90154,5.693962,4.50971
 6\H,0,-0.672822,6.732417,4.911188\H,0,1.358853,5.76771,3.839158\C,0,-1
 .006632,-2.381235,3.271914\C,0,-1.123361,-2.075006,-1.487628\C,0,-2.15
 5178,-2.979339,3.810498\C,0,-2.055596,-3.809121,4.927063\C,0,-0.810333
 ,-4.040831,5.519947\C,0,0.337129,-3.442975,4.988801\C,0,0.241193,-2.61
 9884,3.866862\H,0,-3.116794,-2.783129,3.335912\H,0,-2.952064,-4.276745
 ,5.338548\H,0,-0.733742,-4.687731,6.395945\H,0,1.309818,-3.620761,5.45
 1371\H,0,1.126315,-2.149799,3.438535\C,0,-1.126586,-1.496452,2.06709\C
 ,0,-0.997118,-3.290614,-2.356762\C,0,0.251275,-3.692061,-2.855275\C,0,
 0.35266,-4.824152,-3.663685\C,0,-0.790023,-5.568358,-3.974478\C,0,-2.0
 35971,-5.173704,-3.476739\C,0,-2.140841,-4.03744,-2.674852\H,0,1.13270
 5,-3.103343,-2.601619\H,0,1.325877,-5.12916,-4.052889\H,0,-0.709197,-6
 .456298,-4.604382\H,0,-2.928787,-5.754344,-3.716296\Ni,0,0.,0.,0.\C,0,
 -0.991632,2.346971,-3.297149\C,0,-2.13626,2.663387,-4.043152\C,0,-2.03
 1332,3.456407,-5.185736\C,0,-0.784824,3.946953,-5.587552\C,0,0.358839,
 3.63773,-4.844193\C,0,0.257298,2.838449,-3.705816\H,0,-3.098819,2.2733
 62,-3.711795\H,0,-2.924769,3.694786,-5.765855\H,0,-0.704027,4.569814,-
 6.480451\N,0,2.258719,0.004583,-0.010102\Ni,0,-2.408612,0.,0.\N,0,-4.5
 07504,-0.034948,-0.008709\H,0,2.626597,-0.947447,-0.019016\H,0,2.59213
 7,0.489171,-0.844629\H,0,2.623488,0.486949,0.812273\H,0,-4.897915,0.86
 2919,-0.30098\H,0,-4.832526,-0.754709,-0.657957\H,0,-4.866466,-0.25632
 7,0.922311\O,0,-2.269336,1.990384,1.42547\O,0,-2.269647,1.173686,-1.66
 4286\O,0,-0.002568,1.123118,-1.536919\\Version=AM64L-G09RevD.01\State=
 3-A\HF=-4811.7424993\S2=2.363428\S2-1=0.\S2A=2.002312\RMSD=1.327e-09\D
 ipole=-0.162122,-0.9913859,-0.7201406\Quadrupole=15.5706739,-9.4133909
 ,-6.157283,2.2199541,-0.2773899,-4.7341766\PG=C01 [X(C28H26N2Ni2O8)]\\
 @
""")
f.close()
tmp = read('tmp.log', quantity='multiplicity')
assert tmp == 3
os.unlink('tmp.log')
