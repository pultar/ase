"""Test to ensure the orthnormality of the basis functions."""
import os
import itertools
from ase.ce import BulkCrystal


db_name = 'orthonormal_basis.db'
tol = 1E-9


def test_2(basis_function):
    """Test for 2 element case."""
    setting = BulkCrystal(crystalstructure="fcc",
                          a=4.05,
                          basis_elements=[['Au', 'Cu']],
                          size=[3, 3, 3],
                          conc_args={"conc_ratio_min_1": [[1, 0]],
                                     "conc_ratio_max_1": [[0, 1]]},
                          max_cluster_size=2,
                          db_name=db_name,
                          basis_function=basis_function)
    check_orthonormal(setting)


def test_3(basis_function):
    """Test for 3 element case."""
    setting = BulkCrystal(crystalstructure="fcc",
                          a=4.05,
                          basis_elements=[['Au', 'Cu', 'Ag']],
                          size=[3, 3, 3],
                          conc_args={"conc_ratio_min_1": [[1, 0, 0]],
                                     "conc_ratio_max_1": [[0, 1, 0]]},
                          max_cluster_size=2,
                          db_name=db_name,
                          basis_function=basis_function)
    check_orthonormal(setting)


def test_4(basis_function):
    """Test for 4 element case."""
    setting = BulkCrystal(crystalstructure="fcc",
                          a=4.05,
                          basis_elements=[['Au', 'Cu', 'Ag', 'Ni']],
                          size=[3, 3, 3],
                          conc_args={"conc_ratio_min_1": [[1, 0, 0, 0]],
                                     "conc_ratio_max_1": [[0, 1, 0, 0]]},
                          max_cluster_size=2,
                          db_name=db_name,
                          basis_function=basis_function)
    check_orthonormal(setting)


def test_5(basis_function):
    """Test for 5 element case."""
    setting = BulkCrystal(crystalstructure="fcc",
                          a=4.05,
                          basis_elements=[['Au', 'Cu', 'Ag', 'Ni', 'Fe']],
                          size=[3, 3, 3],
                          conc_args={"conc_ratio_min_1": [[1, 0, 0, 0, 0]],
                                     "conc_ratio_max_1": [[0, 1, 0, 0, 0]]},
                          max_cluster_size=2,
                          db_name=db_name,
                          basis_function=basis_function)
    check_orthonormal(setting)

def test_6(basis_function):
    """Test for 6 element case."""
    setting = BulkCrystal(crystalstructure="fcc",
                          a=4.05,
                          basis_elements=[['Au', 'Cu', 'Ag', 'Ni', 'Fe', 'H']],
                          size=[3, 3, 3],
                          conc_args={"conc_ratio_min_1": [[1, 0, 0, 0, 0, 0]],
                                     "conc_ratio_max_1": [[0, 1, 0, 0, 0, 0]]},
                          max_cluster_size=2,
                          db_name=db_name,
                          basis_function=basis_function)
    check_orthonormal(setting)


def check_orthonormal(setting):
    """Check orthonormality."""
    for bf in setting.basis_functions:
        sum = 0
        for key, value in setting.spin_dict.items():
            sum += bf[key] * bf[key]
        sum /= setting.num_unique_elements
        assert abs(sum - 1.0) < tol

    # Check zeros
    alpha = list(range(len(setting.basis_functions)))
    comb = list(itertools.combinations(alpha, 2))
    for c in comb:
        sum = 0
        for key, value in setting.spin_dict.items():
            sum += setting.basis_functions[c[0]][key] \
                * setting.basis_functions[c[1]][key]
        sum /= setting.num_unique_elements
        assert abs(sum) < tol


basis_function = 'sanchez'
test_2(basis_function)
test_3(basis_function)
test_4(basis_function)
test_5(basis_function)
test_6(basis_function)

basis_function = 'vandewalle'
test_2(basis_function)
test_3(basis_function)
test_4(basis_function)
test_5(basis_function)
test_6(basis_function)

os.remove(db_name)
