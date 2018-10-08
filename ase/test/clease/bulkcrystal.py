"""Test to initiatialize CE using a CEBulk.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import os
from ase.clease import CEBulk, GenerateStructures, Evaluate
from ase.clease.newStruct import MaxAttemptReachedError
from ase.calculators.emt import EMT
from ase.db import connect


def get_members_of_family(setting, cname):
    """Return the members of a given cluster family."""
    members = []
    info = setting.cluster_info_by_name(cname)
    for entry in info:
        members.append(entry["indices"])
    return members


def test_binary_system():
    """Verifies that one can run a CE for the binary Au-Cu system.

    The EMT calculator is used for energy calculations
    """
    db_name = "test_crystal.db"
    conc_args = {"conc_ratio_min_1": [[1, 0]],
                 "conc_ratio_max_1": [[0, 1]]}
    bc_setting = CEBulk(crystalstructure="fcc", a=4.05,
                        basis_elements=[["Au", "Cu"]], size=[3, 3, 3],
                        conc_args=conc_args, db_name=db_name)

    struct_generator = GenerateStructures(bc_setting, struct_per_gen=3)
    struct_generator.generate_initial_pool()

    # Compute the energy of the structures
    calc = EMT()
    database = connect(db_name)
    all_atoms = []
    key_value_pairs = []
    for row in database.select("converged=0"):
        atoms = row.toatoms()
        all_atoms.append(atoms)
        key_value_pairs.append(row.key_value_pairs)

    # Write the atoms to the database
    for atoms, kvp in zip(all_atoms, key_value_pairs):
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        kvp["converged"] = True
        database.write(atoms, key_value_pairs=kvp)

    # Evaluate
    eval_l2 = Evaluate(bc_setting, penalty="l2")
    eval_l2.get_cluster_name_eci(alpha=1E-6, return_type='tuple')
    eval_l2.get_cluster_name_eci(alpha=1E-6, return_type='dict')

    # eval_l1 = Evaluate(bc_setting, penalty="l1")
    # eval_l1.get_cluster_name_eci(alpha=1E-3, return_type='tuple')
    # eval_l1.get_cluster_name_eci(alpha=1E-3, return_type='dict')

    os.remove(db_name)


def test_grouped_basis_supercell():
    # ----------------------------------------------------------##
    # Test probe structure generation with cell size (2, 2, 1). ##
    # ----------------------------------------------------------##
    """Test a case where a grouped_basis is used with supercell."""
    db_name = "test_crystal.db"

    # ------------------------------- #
    # 1 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    setting = CEBulk(basis_elements=[['Na', 'Cl'], ['Na', 'Cl']],
                     crystalstructure="rocksalt",
                     a=4.0,
                     size=[2, 2, 1],
                     conc_args={"conc_ratio_min_1": [[1, 0]],
                                "conc_ratio_max_1": [[0, 1]]},
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.,
                     grouped_basis=[[0, 1]])
    assert setting.num_grouped_basis == 1
    assert len(setting.index_by_grouped_basis) == 1
    assert setting.spin_dict == {'Cl': 1.0, 'Na': -1.0}
    assert setting.num_grouped_elements == 2
    assert len(setting.basis_functions) == 1
    flat = [i for sub in setting.index_by_grouped_basis for i in sub]
    background = [a.index for a in setting.atoms_with_given_dim if
                  a.symbol in setting.background_symbol]
    assert len(flat) == len(setting.atoms_with_given_dim) - len(background)
    try:
        gs = GenerateStructures(setting=setting, struct_per_gen=3)
        gs.generate_initial_pool()
        gs = GenerateStructures(setting=setting, struct_per_gen=2)
        gs.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps=1000,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)

    # ------------------------------- #
    # 2 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    setting = CEBulk(basis_elements=[['Zr', 'Ce'], ['O'], ['O']],
                     crystalstructure="fluorite",
                     a=4.0,
                     size=[2, 2, 3],
                     conc_args={"conc_ratio_min_1": [[1, 0], [2]],
                                "conc_ratio_max_1": [[0, 1], [2]]},
                     db_name=db_name,
                     max_cluster_size=2,
                     max_cluster_dia=4.,
                     grouped_basis=[[0], [1, 2]])
    fam_members = get_members_of_family(setting, "c2_4p000_7")
    assert len(fam_members[0]) == 6  # TODO:  Sometimes 5, which is wrong
    assert len(fam_members[1]) == 6
    assert len(fam_members[2]) == 6
    assert setting.num_grouped_basis == 2
    assert len(setting.index_by_grouped_basis) == 2
    assert setting.spin_dict == {'Ce': 1.0, 'O': -1.0, 'Zr': 0}
    assert setting.num_grouped_elements == 3
    assert len(setting.basis_functions) == 2
    flat = [i for sub in setting.index_by_grouped_basis for i in sub]
    background = [a.index for a in setting.atoms_with_given_dim if
                  a.symbol in setting.background_symbol]
    assert len(flat) == len(setting.atoms_with_given_dim) - len(background)

    try:
        gs = GenerateStructures(setting=setting, struct_per_gen=3)
        gs.generate_initial_pool()
        gs = GenerateStructures(setting=setting, struct_per_gen=2)
        gs.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps=1000,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)

    # # ---------------------------------- #
    # # 2 grouped_basis + background atoms #
    # # ---------------------------------- #
    # # initial_pool + probe_structures    #
    # # ---------------------------------- #
    setting = CEBulk(basis_elements=[['Ca'], ['O', 'F'], ['O', 'F']],
                     crystalstructure="fluorite",
                     a=4.0,
                     size=[2, 2, 2],
                     conc_args={"conc_ratio_min_1": [[1], [2, 0]],
                                "conc_ratio_max_1": [[1], [0, 2]]},
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.,
                     grouped_basis=[[0], [1, 2]],
                     ignore_background_atoms=True)
    # print(setting.supercell_scale_factor)
    assert setting.num_grouped_basis == 1
    assert len(setting.index_by_grouped_basis) == 1
    assert setting.spin_dict == {'F': 1.0, 'O': -1.0}
    assert setting.num_grouped_elements == 2
    assert len(setting.basis_functions) == 1
    flat = [i for sub in setting.index_by_grouped_basis for i in sub]
    background = [a.index for a in setting.atoms_with_given_dim if
                  a.symbol in setting.background_symbol]
    assert len(flat) == len(setting.atoms_with_given_dim) - len(background)

    try:
        gs = GenerateStructures(setting=setting, struct_per_gen=3)
        gs.generate_initial_pool()
        gs = GenerateStructures(setting=setting, struct_per_gen=2)
        gs.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps=1000,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)


test_binary_system()
test_grouped_basis_supercell()
