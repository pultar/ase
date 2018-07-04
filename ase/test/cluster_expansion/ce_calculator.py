import os
from random import randint
import numpy as np
from ase.calculators.cluster_expansion import ClusterExpansion
from ase.ce import BulkCrystal, BulkSpacegroup, CorrFunction
from ase.build import bulk
from ase.visualize import view


def generate_ex_eci(setting):
    cf = CorrFunction(setting)
    cf = cf.get_cf(setting.atoms)
    eci = {key: -0.001 for key in cf}
    return eci


def get_binary():
    """Return a simple binary test structure."""
    bc_setting = BulkCrystal(crystalstructure="fcc",
                             a=4.05,
                             basis_elements=[["Au", "Cu"]],
                             size=[3, 3, 3],
                             conc_args={"conc_ratio_min_1": [[1, 0]],
                                        "conc_ratio_max_1": [[0, 1]]},
                             db_name=db_name)

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (3, 3, 3)
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"
    return bc_setting, atoms


def get_ternary():
    """Return a ternary test structure."""
    bc_setting = BulkCrystal(crystalstructure="fcc",
                             a=4.05,
                             basis_elements=[["Au", "Cu", "Zn"]],
                             size=[3, 3, 3],
                             conc_args={"conc_ratio_min_1": [[1, 0, 0]],
                                        "conc_ratio_max_1": [[0, 1, 0]],
                                        "conc_ratio_min_2": [[2, 0, 0]],
                                        "conc_ratio_max_2": [[0, 1, 1]]},
                             db_name=db_name)

    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (3, 3, 3)
    for i in range(2):
        atoms[3 * i].symbol = "Au"
        atoms[3 * i + 1].symbol = "Cu"
        atoms[3 * i + 2].symbol = "Zn"
    return bc_setting, atoms

def get_rocksalt():
    """Test rocksalt where passed atoms with background_atoms."""
    setting = BulkCrystal(basis_elements=[['Li', 'X', 'V'],
                                          ['O']],
                          crystalstructure='rocksalt',
                          a=4.05,
                          size=[3, 3, 3],
                          conc_args={"conc_ratio_min_1": [[2, 0, 1], [3]],
                                     "conc_ratio_max_1": [[2, 1, 0], [3]]},
                          db_name=db_name,
                          max_cluster_size=2,
                          # max_cluster_dist=4.0,
                          ignore_background_atoms=True)

    atoms = bulk("LiO", crystalstructure="rocksalt", a=4.05)
    atoms = atoms * (3, 3, 3)
    Li_indx = [a.index for a in atoms if a.symbol == 'Li']
    O_indx = [a.index for a in atoms if a.symbol == 'O']
    for i in range(3):
        if i < 2:
            atoms[Li_indx[i]].symbol = 'V'
        else:
            atoms[Li_indx[i]].symbol = 'X'
    return setting, atoms

def get_spacegroup():
    """Test rocksalt where passed atoms."""
    setting = BulkSpacegroup(basis_elements=[['O', 'X'], ['O', 'X'],
                                             ['O', 'X'], ['Ta']],
                             basis=[(0., 0., 0.),
                                    (0.3894, 0.1405, 0.),
                                    (0.201,  0.3461, 0.5),
                                    (0.2244, 0.3821, 0.)],
                             spacegroup=55,
                             cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                             size=[1, 2, 2],
                             conc_args={"conc_ratio_min_1": [[5, 0], [2]],
                                        "conc_ratio_max_1": [[4, 1], [2]]},
                             db_name=db_name,
                             max_cluster_size=2,
                             max_cluster_dist=5.0,
                             grouped_basis=[[0, 1, 2], [3]])

    atoms = bulk("LiO", crystalstructure="rocksalt", a=4.05)
    atoms = atoms * (3, 3, 3)
    Li_indx = [a.index for a in atoms if a.symbol == 'Li']
    O_indx = [a.index for a in atoms if a.symbol == 'O']
    for i in range(3):
        atoms[O_indx[i]].symbol = 'F'
        if i < 2:
            atoms[Li_indx[i]].symbol = 'V'
        else:
            atoms[Li_indx[i]].symbol = 'X'
    return setting, atoms


def test_update_correlation_functions(setting, atoms, n_trial_configs=20):
    cf = CorrFunction(setting)

    eci = generate_ex_eci(setting)

    calc = ClusterExpansion(setting, cluster_name_eci=eci)
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

        brute_force_cf = cf.get_cf_by_cluster_names(atoms, calc.cluster_names,
                                                    return_type="array")
        assert np.allclose(brute_force_cf, calc.cf)


db_name = 'test.db'
print('binary')
binary_setting, bin_atoms = get_binary()
test_update_correlation_functions(binary_setting, bin_atoms, n_trial_configs=5)
os.remove(db_name)

print('ternary')
ternary_setting, tern_atoms = get_ternary()
test_update_correlation_functions(ternary_setting, tern_atoms,
                                  n_trial_configs=5)
os.remove(db_name)

print('rocksalt')
rs_setting, rs_atoms = get_rocksalt()
test_update_correlation_functions(rs_setting, rs_atoms,
                                  n_trial_configs=5)
os.remove(db_name)
