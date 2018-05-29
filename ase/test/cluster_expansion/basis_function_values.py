"""
Verify that the value assigned to elements stays the same
"""
import os
from ase.ce import BulkCrystal

def test_binary():
    """
    Test the basis functions for a binary Al-Mg system
    """
    db_name = "almg_binary_test.db"
    conc_args = {
        "conc_ratio_min_1":[[1, 0]],
        "conc_ratio_max_1":[[0, 1]]
    }
    bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05, \
    basis_elements=[["Al", "Mg"]], size=[2, 2, 2], conc_args=conc_args, \
    db_name=db_name)

    basis_funcs = bc_setting.basis_functions[0]

    expected_values = {
        "Al":1.0,
        "Mg":-1.0
    }

    tol = 1E-6
    for key in expected_values.keys():
        assert abs(expected_values[key]-basis_funcs[key]) < tol

    # Delete the database
    os.remove(db_name)

test_binary()
