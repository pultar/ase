from __future__ import print_function
import os
import ase.build
from ase.calculators.siesta.siesta import Siesta

# Create temporary directory for running tests.
test_path = 'tmp_siesta_read_eigenvalues'
if not os.path.exists(test_path):
    os.makedirs(test_path)
os.chdir(test_path)

run_path = 'run_directory'
if not os.path.exists(run_path):
    os.makedirs(run_path)

# Test real calculation of the lithium bulk which produced a gapped .EIG file
atoms = ase.build.bulk('Li', cubic=True)
calc = Siesta(kpts=[2,1,1])
atoms.set_calculator(calc)
atoms.get_potential_energy()

assert len(calc.results['eigenvalues'])==2
assert len(calc.results['kpoints'])==2
assert len(calc.results['kweights'])==2

for k in calc.results['eigenvalues']:
    assert k[1]==0
    assert len(calc.results['eigenvalues'][k])==10

# Remove the test directory.
os.chdir('../..')
os.system('rm -rf {}'.format( test_path))
