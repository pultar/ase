"""Test to initiatialize CE using a CEBulk.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import os
import json
from ase.clease import CEBulk, CorrFunction, NewStructures, Evaluate, Concentration
from ase.clease.newStruct import MaxAttemptReachedError
from ase.clease.tools import update_db
from ase.calculators.emt import EMT
from ase.db import connect
from ase.test.clease.reference_corr_funcs_bulk import all_cf

# If this is True, the JSON file containing the correlation functions
# Used to check consistency of the reference functions is updated
# This should normally be False
update_reference_file = False
tol = 1E-9

db_name = "test_bulk.db"


def get_members_of_family(setting, cname):
    """Return the members of a given cluster family."""
    members = []
    info = setting.cluster_info_by_name(cname)
    for entry in info:
        members.append(entry["indices"])
    return members

def calculate_cf(setting, atoms):
    cf = CorrFunction(setting)
    return cf.get_cf(atoms, return_type='dict')

def test_corrfunc():
    basis_elements = [['Au', 'Cu']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure='fcc', a=4.05, size=[3, 3, 3],
                        concentration=concentration, db_name=db_name)
    atoms = setting.atoms.copy()
    atoms[0].symbol = 'Cu'
    atoms[3].symbol = 'Cu'
    cf = calculate_cf(setting, atoms)

    if update_reference_file:
        all_cf["binary_fcc"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["binary_fcc"][key]) < tol
    
    os.remove(db_name)

    basis_elements = [['Li', 'V'], ['X', 'O']]
    concentration = Concentration(basis_elements=basis_elements)
    setting = CEBulk(crystalstructure="rocksalt",
                     a=4.0,
                     size=[2, 2, 1],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.)
    atoms = setting.atoms.copy()
    Li_ind = [atom.index for atom in atoms if atom.symbol=='Li']
    X_ind = [atom.index for atom in atoms if atom.symbol=='X']
    atoms[Li_ind[0]].symbol = 'V'
    atoms[X_ind[0]].symbol = 'O'
    cf = calculate_cf(setting, atoms)
    if update_reference_file:
        all_cf["two_basis"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["two_basis"][key]) < tol
    os.remove(db_name)

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
    atoms = setting.atoms.copy()
    atoms[1].symbol = 'Cl'
    atoms[7].symbol = 'Cl'
    cf = calculate_cf(setting, atoms)
    if update_reference_file:
        all_cf["one_grouped_basis"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["one_grouped_basis"][key]) < tol
    os.remove(db_name)

    basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
    concentration = Concentration(basis_elements=basis_elements,
                                  grouped_basis=[[0], [1, 2]])
    setting = CEBulk(crystalstructure="fluorite",
                     a=4.0,
                     size=[2, 2, 2],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.,
                     ignore_background_atoms=True)
    atoms = setting.atoms.copy()
    O_ind = [atom.index for atom in atoms if atom.symbol=='O']
    atoms[O_ind[0]].symbol = 'F'
    atoms[O_ind[1]].symbol = 'F'
    cf = calculate_cf(setting, atoms)
    if update_reference_file:
        all_cf["two_grouped_basis_bckgrnd"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["two_grouped_basis_bckgrnd"][key]) < tol
    os.remove(db_name)


def test_binary_system():
    """Verifies that one can run a CE for the binary Au-Cu system.

    The EMT calculator is used for energy calculations
    """
    basis_elements = [['Au', 'Cu']]
    concentration = Concentration(basis_elements=basis_elements)
    bc_setting = CEBulk(crystalstructure='fcc', a=4.05, size=[3, 3, 3],
                        concentration=concentration, db_name=db_name)

    newstruct = NewStructures(bc_setting, struct_per_gen=3)
    newstruct.generate_initial_pool()

    # Compute the energy of the structures
    calc = EMT()
    database = connect(db_name)

    # Write the atoms to the database
    # for atoms, kvp in zip(all_atoms, key_value_pairs):
    for row in database.select([("converged", "=", False)]):
        atoms = row.toatoms()
        atoms.set_calculator(calc)
        atoms.get_potential_energy()
        update_db(uid_initial=row.id, final_struct=atoms, db_name=db_name)
    # Evaluate
    eval_l2 = Evaluate(bc_setting, fitting_scheme="l2", alpha=1E-6)
    eval_l2.get_cluster_name_eci(return_type='tuple')
    eval_l2.get_cluster_name_eci(return_type='dict')

    # Test subclusters for pairs
    for cluster in bc_setting.cluster_info_given_size(2):
        name = list(cluster.keys())[0]
        sub_cl = set(bc_setting.subclusters(name))
        assert sub_cl == set(["c0", "c1"])

    # Test a few known clusters. Triplet nearest neighbour
    name = "c3_01nn_0"
    sub_cl = set(bc_setting.subclusters(name))
    assert sub_cl == set(["c0", "c1", "c2_01nn_0"])

    name = "c3_02nn_0"
    sub_cl = set(bc_setting.subclusters(name))
    assert sub_cl == set(["c0", "c1", "c2_01nn_0", "c2_02nn_0"])

    name = "c4_01nn_0"
    sub_cl = set(bc_setting.subclusters(name))
    assert sub_cl == set(["c0", "c1", "c2_01nn_0", "c3_01nn_0"])

    os.remove(db_name)


def test_initial_pool():
    from ase.db import connect
    basis_elements = [['Li', 'V'], ['X', 'O']]
    concentration = Concentration(basis_elements=basis_elements)

    setting = CEBulk(crystalstructure="rocksalt",
                     a=4.0,
                     size=[2, 2, 1],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.)
    ns = NewStructures(setting=setting, struct_per_gen=2)
    ns.generate_initial_pool()

    # At this point there should be the following
    # structures in the DB
    expected_names = ["V1_O1_0", "Li1_X1_0", 
                      "V1_X1_0", "Li1_O1_0"]
    db = connect(db_name)
    for name in expected_names:
        num = sum(1 for row in db.select(name=name))
        assert num == 1
    os.remove(db_name)

def test_1grouped_basis_probe():
    """Test a case where a grouped_basis is used with supercell."""
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
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_random_structures()
        ns.generate_initial_pool()
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)


def test_2grouped_basis_probe():
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
    fam_members = get_members_of_family(setting, "c2_06nn_0")
    print(fam_members)
    assert len(fam_members[0]) == 6  # TODO:  Sometimes 5, which is wrong
    assert len(fam_members[1]) == 6
    assert len(fam_members[2]) == 6
    assert setting.num_basis == 2
    assert len(setting.index_by_basis) == 2
    assert setting.spin_dict == {'Ce': 1.0, 'O': -1.0, 'Zr': 0}
    assert len(setting.basis_functions) == 2

    try:
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)

def test_2grouped_basis_bckgrnd_probe():
    # ---------------------------------- #
    # 2 grouped_basis + background atoms #
    # ---------------------------------- #
    # initial_pool + probe_structures    #
    # ---------------------------------- #
    basis_elements = [['Ca'], ['O', 'F'], ['O', 'F']]
    concentration = Concentration(basis_elements=basis_elements,
                                  grouped_basis=[[0], [1, 2]])
    setting = CEBulk(crystalstructure="fluorite",
                     a=4.0,
                     size=[2, 2, 2],
                     concentration=concentration,
                     db_name=db_name,
                     max_cluster_size=3,
                     max_cluster_dia=4.,
                     ignore_background_atoms=True)
    assert setting.num_basis == 2
    assert len(setting.index_by_basis) == 2
    assert setting.spin_dict == {'F': 1.0, 'O': -1.0}
    assert len(setting.basis_functions) == 1

    try:
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_initial_pool()
        ns = NewStructures(setting=setting, struct_per_gen=2)
        ns.generate_probe_structure(init_temp=1.0, final_temp=0.001,
                                    num_temp=5, num_steps_per_temp=100,
                                    approx_mean_var=True)

    except MaxAttemptReachedError as exc:
        print(str(exc))

    os.remove(db_name)

print('binary')
test_binary_system()

print('1 grouped basis with probe structure')
test_1grouped_basis_probe()

print('2 grouped basis with probe structure')
test_2grouped_basis_probe()

print('2 grouped basis + background + probe structure')
test_2grouped_basis_bckgrnd_probe()

print('initial pool')
test_initial_pool()
test_corrfunc()

if update_reference_file:
    print("Updating the reference correlation function file")
    print("This should normally not be done.")
    with open("reference_corr_funcs_bulk.py", 'w') as outfile:
        json.dump(all_cf, outfile, indent=2, separators=(',', ': '))