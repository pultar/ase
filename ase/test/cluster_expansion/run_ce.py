"""
Test:
1. Initialize the CE
2. Add a few structures
3. Compute the energy
4. Run the evaluation routine
"""

import os
from ase.ce import BulkCrystal
from ase.ce import GenerateStructures
from ase.ce import Evaluate
from ase.calculators.emt import EMT  # Use this calculator as it is fast
from ase.db import connect


def test_binary_system():
    """
    Verifies that one can run a CE for the binary Au-Cu system.
    The EMT calculator is used for energy calculations
    """
    db_name = "aucu_binary_test.db"
    conc_args = {
        "conc_ratio_min_1": [[1, 0]],
        "conc_ratio_max_1": [[0, 1]]
    }
    bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05,
                             basis_elements=[["Au", "Cu"]], size=[3, 3, 3], conc_args=conc_args,
                             db_name=db_name)

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
    evaluator = Evaluate(bc_setting, penalty="l2", lamb=1E-6)
    evaluator.get_cluster_name_eci_dict
    os.remove(db_name)


test_binary_system()
