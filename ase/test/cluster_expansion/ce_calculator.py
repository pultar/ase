import os
from random import randint
import numpy as np
from ase.calculators.cluster_expansion import ClusterExpansion
from ase.ce import BulkCrystal, CorrFunction
from ase.build import bulk
from ase.ce.tools import wrap_and_sort_by_position

def generate_ex_eci(bc):
    cf = CorrFunction(bc)
    cf = cf.get_cf(bc.atoms)
    eci = {key: -0.001 for key in cf}
    return eci


def all_cf_match(cf_dict, calc):
    tol = 1E-6
    print(cf_dict)
    print(calc.cf)
    for i in range(len(calc.cf)):
        assert abs(cf_dict[i] - calc.cf[i]) < tol


def get_binary():
    """Return a simple binary test structure."""
    conc_args = {"conc_ratio_min_1": [[1, 0]],
                 "conc_ratio_max_1": [[0, 1]]}
    bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05,
                             basis_elements=[["Au", "Cu"]], size=[3, 3, 3],
                             conc_args=conc_args, db_name=db_name)

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms*(3, 3, 3)
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"
    return bc_setting, wrap_and_sort_by_position(atoms)


def get_ternary():
    """Return a ternary test structure."""
    conc_args = {"conc_ratio_min_1": [[1, 0, 0]],
                 "conc_ratio_max_1": [[0, 1, 0]],
                 "conc_ratio_min_2": [[2, 0, 0]],
                 "conc_ratio_max_2": [[0, 1, 1]]}
    bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05,
                             basis_elements=[["Au", "Cu", "Zn"]],
                             size=[3, 3, 3], conc_args=conc_args,
                             db_name=db_name)

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms*(3, 3, 3)
    for i in range(2):
        atoms[3 * i].symbol = "Au"
        atoms[3 * i + 1].symbol = "Cu"
        atoms[3 * i + 2].symbol = "Zn"
    return bc_setting, wrap_and_sort_by_position(atoms)


def test_update_correlation_functions(bc_setting, atoms, n_trial_configs=20):
    cf = CorrFunction(bc_setting)

    eci = generate_ex_eci(bc_setting)

    calc = ClusterExpansion(bc_setting, cluster_name_eci=eci)
    atoms.set_calculator(calc)

    for _ in range(n_trial_configs):
        indx1 = randint(0, len(atoms) - 1)
        symb1 = atoms[indx1].symbol
        symb2 = symb1
        while symb2 == symb1:
            indx2 = randint(0, len(atoms) - 1)
            symb2 = atoms[indx2].symbol

        atoms[indx1].symbol = symb2
        atoms[indx2].symbol = symb1

        # The calculator should update its correlation functions
        # when the energy is computed
        energy = atoms.get_potential_energy()

        brute_force_cf = cf.get_cf_by_cluster_names(
            atoms, calc.cluster_names, return_type="array")
        assert np.allclose(brute_force_cf, calc.cf)


db_name = 'test.db'
binary_setting, bin_atoms = get_binary()
test_update_correlation_functions(binary_setting, bin_atoms, n_trial_configs=5)
os.remove(db_name)

ternary_setting, tern_atoms = get_ternary()
test_update_correlation_functions(ternary_setting, tern_atoms, n_trial_configs=5)
os.remove(db_name)
