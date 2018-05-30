from ase.ce import StructureComparator
import glob
from ase.io import read
import os
from ase.test import NotAvailable
import ase

"""
Inside the folder structure_match_data all structures inside the same subfolder
should be symmetrically equivalent so the comparator should return True
when comparing these two atoms

When comparing atoms-objects from different subfolders they should return False
"""

ase_folder = ase.__file__.rpartition("/")[0]
folder = ase_folder + "/test/cluster_expansion/structure_match_data"
if not os.path.exists(folder):
    raise NotAvailable("Cannot locate the datafiles for the test structures")

test_fname_ref = folder + "/equiv331_"
n_sets = 218
cpp_stepsize = 1
python_stepsize = 20
false_structure_stepsize = 10
verbose = False


def test_equal(comparator, step):
    # All structure in each folder are similar
    for i in range(0, n_sets, step):
        ref_file = test_fname_ref + "{}/equiv331_{}_0.xyz".format(i, i)
        ref_atoms = read(ref_file)
        for struct_file in glob.glob(test_fname_ref + "{}/*.xyz".format(i)):
            atoms = read(struct_file)
            assert comparator.compare(ref_atoms, atoms)


def test_not_equal(comparator, step):
    # Test only a selection of possible combinations
    comparator = StructureComparator()
    for i in range(0, n_sets, step):
        ref_file = test_fname_ref + "{}/equiv331_{}_0.xyz".format(i, i)
        atoms1 = read(ref_file)
        for j in range(i + 1, n_sets, false_structure_stepsize):
            ref_file2 = test_fname_ref + "{}/equiv331_{}_0.xyz".format(j, j)
            atoms2 = read(ref_file2)
            assert not comparator.compare(atoms1, atoms2)


comparator = StructureComparator()
if comparator.use_cpp_version:
    test_equal(comparator, cpp_stepsize)
    test_not_equal(comparator, cpp_stepsize)
    comparator.use_cpp_version = False

if verbose:
    print("C++ tests finished")

# Test with pure python version
# Python about is about 10x slower it is too tedious to test everything
test_equal(comparator, python_stepsize)
test_not_equal(comparator, python_stepsize)
