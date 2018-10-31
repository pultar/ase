"""Test to initiatialize CE using a CEBulk.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import os
from ase.clease import CEBulk, NewStructures, Evaluate
from ase.clease.newStruct import MaxAttemptReachedError
from ase.calculators.emt import EMT
from ase.db import connect
from ase.clease.concentration import Concentration
from ase.build import bulk


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
    basis_elements = [["Au", "Cu"]]
    concentration = Concentration(basis_elements=basis_elements)
    bc_setting = CEBulk(crystalstructure="fcc", a=4.05, size=[3, 3, 3],
                        concentration=concentration, db_name=db_name)

    newstruct = NewStructures(bc_setting, struct_per_gen=3)
    newstruct.generate_initial_pool()

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
    eval_l2 = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
    eval_l2.get_cluster_name_eci(return_type='tuple')
    eval_l2.get_cluster_name_eci(return_type='dict')


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
    basis_elements = [['Na', 'Cl'], ['Na', 'Cl']]
    concentration = Concentration(basis_elements=basis_elements,
                                  grouped_basis=[[0, 1]])
    setting = CEBulk(crystalstructure="rocksalt",
                     a=4.0,
                     size=[2, 2, 1],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.)
    assert setting.num_basis == 1
    assert len(setting.index_by_basis) == 1
    assert setting.spin_dict == {'Cl': 1.0, 'Na': -1.0}
    assert len(setting.basis_functions) == 1
    try:
        ns = NewStructures(setting=setting, struct_per_gen=3)
        ns.generate_initial_pool()
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)

    # ------------------------------- #
    # 2 grouped basis                 #
    # ------------------------------- #
    # initial_pool + probe_structures #
    # ------------------------------- #
    basis_elements = [['Zr', 'Ce'], ['O'], ['O']]
    concentration = Concentration(basis_elements=basis_elements,
                                  grouped_basis=[[0], [1, 2]])
    setting = CEBulk(crystalstructure="fluorite",
                     a=4.0,
                     size=[2, 2, 3],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=2,
                     max_cluster_dia=4.)
    fam_members = get_members_of_family(setting, "c2_4p000_7")
    assert len(fam_members[0]) == 6  # TODO:  Sometimes 5, which is wrong
    assert len(fam_members[1]) == 6
    assert len(fam_members[2]) == 6
    assert setting.num_basis == 2
    assert len(setting.index_by_basis) == 2
    assert setting.spin_dict == {'Ce': 1.0, 'O': -1.0, 'Zr': 0}
    assert len(setting.basis_functions) == 2

    try:
        ns = NewStructures(setting=setting, struct_per_gen=3)
        ns.generate_initial_pool()
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)

    # ---------------------------------- #
    # 2 grouped_basis + background atoms #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    # [["Ca"], ["O", "F"]]
    basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(basis_elements=basis_elements,
                     crystalstructure="fluorite",
                     a=4.0,
                     size=[2, 2, 2],
                     concentration=concentration,
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
        ns = NewStructures(setting=setting, struct_per_gen=3)
        ns.generate_initial_pool()
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)


test_binary_system()
test_grouped_basis_supercell()
