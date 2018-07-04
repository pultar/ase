"""Test to initiatialize CE using a BulkSpacegroup.

1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""
import os
import json
from ase.ce import BulkSpacegroup, GenerateStructures, CorrFunction
from ase.test.cluster_expansion.reference_corr_funcs import all_cf

# If this is True, the JSON file containing the correlation functions
# Used to check consistency of the reference functions is updated
# This should normally be False
update_reference_file = False


def test_spgroup_217():
    """Test the initialization of spacegroup 217."""
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a, b, c, alpha, beta, gamma]
    basis = [(0, 0, 0), (0.324, 0.324, 0.324),
             (0.3582, 0.3582, 0.0393), (0.0954, 0.0954, 0.2725)]
    conc_args = {"conc_ratio_min_1": [[1, 0]],
                 "conc_ratio_max_1": [[0, 1]]}
    db_name = "test_spacegroup.db"
    basis_elements = [["Al", "Mg"], ["Al", "Mg"], ["Al", "Mg"], ["Al", "Mg"]]

    # Test with grouped basis
    bsg = BulkSpacegroup(basis_elements=basis_elements,
                         basis=basis,
                         spacegroup=217,
                         cellpar=cellpar,
                         conc_args=conc_args,
                         max_cluster_size=4,
                         db_name=db_name,
                         size=[1, 1, 1],
                         grouped_basis=[[0, 1, 2, 3]],
                         max_cluster_dist=5.0)
    assert bsg.num_trans_symm == 29
    atoms = bsg.atoms.copy()
    atoms[0].symbol = "Mg"
    atoms[10].symbol = "Mg"
    atoms[20].symbol = "Mg"
    atoms[30].symbol = "Mg"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)

    if update_reference_file:
        all_cf["sp_217_grouped"] = cf
    assert all_cf["sp_217_grouped"] == cf
    os.remove(db_name)


def test_grouped_basis_with_large_dist():
    # Test with grouped basis with a supercell
    db_name = "test_spacegroup.db"
    tol = 1E-9
    bsg = BulkSpacegroup(basis_elements=[['O', 'X'], ['O', 'X'],
                                         ['O', 'X'], ['Ta']],
                         basis=[(0., 0., 0.),
                                (0.3894, 0.1405, 0.),
                                (0.201,  0.3461, 0.5),
                                (0.2244, 0.3821, 0.)],
                         spacegroup=55,
                         cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                         size=[1, 1, 2],
                         conc_args={"conc_ratio_min_1": [[5, 0], [2]],
                                    "conc_ratio_max_1": [[4, 1], [2]]},
                         db_name=db_name,
                         max_cluster_size=2,
                         max_cluster_dist=5.0,
                         grouped_basis=[[0, 1, 2], [3]])
    assert bsg.unique_elements == ['O', 'Ta', 'X']
    assert bsg.spin_dict == {'O': 1.0, 'Ta': -1.0, 'X': 0.0}
    assert len(bsg.basis_functions) == 2

    atoms = bsg.atoms.copy()
    indx_to_X = [0, 4, 8, 12, 16]
    for indx in indx_to_X:
        atoms[indx].symbol = "X"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)
    if update_reference_file:
        all_cf["Ta_O_X_grouped"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["Ta_O_X_grouped"][key]) < tol

    gs = GenerateStructures(setting=bsg, struct_per_gen=3)
    gs.generate_initial_pool()
    # gs = GenerateStructures(setting=bsg, struct_per_gen=3)
    # gs.generate_probe_structure(init_temp=10., final_temp=1., num_temp=2,
    #                             num_steps=10, approx_mean_var=True)
    os.remove(db_name)


    bsg = BulkSpacegroup(basis_elements=[['O', 'X'], ['Ta'], ['O', 'X'],
                                         ['O', 'X']],
                         basis=[(0., 0., 0.),
                                (0.2244, 0.3821, 0.),
                                (0.3894, 0.1405, 0.),
                                (0.201,  0.3461, 0.5)],
                         spacegroup=55,
                         cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                         size=[2, 2, 3],
                         conc_args={"conc_ratio_min_1": [[2], [5, 0]],
                                    "conc_ratio_max_1": [[2], [4, 1]]},
                         db_name=db_name,
                         max_cluster_size=2,
                         max_cluster_dist=5.0,
                         grouped_basis=[[1], [0, 2, 3]],
                         ignore_background_atoms=True)
    assert bsg.unique_elements == ['O', 'X']
    assert bsg.spin_dict == {'O': 1.0, 'X': -1.0}
    assert bsg.basis_elements == [['O', 'X'], ['O', 'X'], ['O', 'X']]
    assert len(bsg.basis_functions) == 1

    atoms = bsg.atoms.copy()
    indx_to_X = [0, 4, 8, 12, 16]
    for indx in indx_to_X:
        atoms[indx].symbol = "X"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)
    if update_reference_file:
        all_cf["Ta_O_X_ungrouped"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["Ta_O_X_ungrouped"][key]) < tol

    os.remove(db_name)

    bsg = BulkSpacegroup(basis_elements=[['Li', 'X', 'V'], ['Li', 'X', 'V'],
                                         ['O', 'F']],
                         basis=[(0.00, 0.00, 0.00),
                                (1./3, 2./3, 0.00),
                                (1./3, 0.00, 0.25)],
                         spacegroup=167,
                         cellpar=[5.123, 5.123, 13.005, 90., 90., 120.],
                         size=[1, 1, 1],
                         conc_args={"conc_ratio_min_1": [[0, 2, 1], [2, 1]],
                                    "conc_ratio_max_1": [[2, 0, 1], [2, 1]]},
                         db_name=db_name,
                         grouped_basis=[[0, 1], [2]],
                         max_cluster_size=2,
                         max_cluster_dist=5.0)
    assert bsg.unique_elements == ['F', 'Li', 'O', 'V', 'X']
    assert bsg.spin_dict == {'F': 2.0, 'Li': -2.0, 'O': 1.0, 'V': -1.0, 'X': 0}
    assert len(bsg.basis_functions) == 4

    atoms = bsg.atoms.copy()
    indx_to_X = [6, 33, 8, 35]
    for indx in indx_to_X:
        atoms[indx].symbol = "X"
    corr = CorrFunction(bsg)
    cf = corr.get_cf(atoms)
    if update_reference_file:
        all_cf["Li_X_V_O_F"] = cf
    for key in cf.keys():
        assert abs(cf[key] - all_cf["Li_X_V_O_F"][key]) < tol

    os.remove(db_name)


def sum_cf(cf):
    sum = 0.0
    for key, value in cf.items():
        sum += value
    return sum


test_spgroup_217()
test_grouped_basis_with_large_dist()

if update_reference_file:
    print ("Updating the reference correlation function file")
    print ("This should normally not be done.")
    with open("reference_corr_funcs.py", 'w') as outfile:
        json.dump(all_cf, outfile, indent=2, separators=(',', ': '))
