import os
from random import shuffle, randint
import numpy as np
from ase.calculators.cluster_expansion import ClusterExpansion
from ase.ce import BulkCrystal, CorrFunction
from ase.visualize import view

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
    view(bc_setting.atoms)

    atoms = bc_setting.atoms
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"
    view(bc_setting.atoms)
    return bc_setting


def get_ternary():
    """Return a ternary test structure."""
    conc_args = {"conc_ratio_min_1": [[1, 0, 0]],
                 "conc_ratio_max_1": [[0, 1, 0]],
                 "conc_ratio_min_2": [[0, 0, 1]],
                 "conc_ratio_max_2": [[1, 0, 0]]}
    bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05,
                             basis_elements=[["Au", "Cu", "Zn"]],
                             size=[3, 3, 3], conc_args=conc_args,
                             db_name=db_name)

    atoms = bc_setting.atoms
    for i in range(2):
        atoms[3 * i].symbol = "Au"
        atoms[3 * i + 1].symbol = "Cu"
        atoms[3 * i + 2].symbol = "Zn"
    return bc_setting


def test_update_correlation_functions(bc_setting, n_trial_configs=20):
    cf = CorrFunction(bc_setting)
    atoms = bc_setting.atoms

    eci = generate_ex_eci(bc_setting)

    # Insert some Au and Cu elements
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"

    # ---------------------------------
    # init_cf, indices, symbols assigned but never used?
    # shuffle imported but never used.
    # ---------------------------------
    init_cf = cf.get_cf(atoms)

    calc = ClusterExpansion(bc_setting, cluster_name_eci=eci)
    atoms.set_calculator(calc)
    #calc.atoms = atoms

    indices = range(len(atoms))
    symbols = [atom.symbol for atom in atoms]
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
        # print(calc.atoms.set_calculator(calc))
        brute_force_cf = cf.get_cf_by_cluster_names(
            atoms, calc.cluster_names, return_type="array")
        assert np.allclose(brute_force_cf, calc.cf)


db_name = 'test.db'
binary_setting = get_binary()
test_update_correlation_functions(binary_setting, n_trial_configs=5)
os.remove(db_name)

ternary_setting = get_ternary()
test_update_correlation_functions(ternary_setting, n_trial_configs=5)
os.remove(db_name)
