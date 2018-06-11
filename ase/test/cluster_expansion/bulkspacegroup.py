"""Test to initiate all possible cases of a BulkSpacegroup.

It should not be nessecary to call GenerateStructures, Evaluate etc.
with BulkSpacegroup as this is tested in run_ce.py for BulkCrystal.
"""
import os
from ase.ce import BulkSpacegroup


def test_spgroup_217():
    """Initiate a BulkSpacegroup to describe gamma phase."""
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
    db_name = "test.db"
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
    os.remove(db_name)

    conc_args = {"conc_ratio_min_1": [[1, 0], [1, 0], [1, 0], [1, 0]],
                 "conc_ratio_max_1": [[0, 1], [0, 1], [0, 1], [0, 1]]}
    basis_elements = [["Al", "Mg"], ["Si", "Mg"], ["Cu", "Mg"], ["Zn", "Mg"]]
    # Test without grouped basis
    bsg = BulkSpacegroup(basis_elements=basis_elements,
                         basis=basis,
                         spacegroup=217,
                         cellpar=cellpar,
                         conc_args=conc_args,
                         max_cluster_size=4,
                         db_name=db_name,
                         size=[1, 1, 1],
                         max_cluster_dist=5.0)
    os.remove(db_name)

def test_grouped_basis_with_large_dist():
    # Test with grouped basis
    db_name = "test.db"

    bsg = BulkSpacegroup(basis_elements=[['O', 'X'], ['O', 'X'],
                                         ['O', 'X'], ['Ta']],
                         basis=[(0., 0., 0.),
                                (0.3894, 0.1405, 0.),
                                (0.201,  0.3461, 0.5),
                                (0.2244, 0.3821, 0.)],
                         spacegroup=55,
                         cellpar=[6.25, 7.4, 3.83, 90, 90, 90],
                         size=[2, 2, 3],
                         conc_args={"conc_ratio_min_1": [[5, 0], [2]],
                                    "conc_ratio_max_1": [[4, 1], [2]]},
                         db_name=db_name,
                         max_cluster_size=3,
                         max_cluster_dist=5.0,
                         grouped_basis=[[0, 1, 2], [3]])
    os.remove(db_name)

    bsg = BulkSpacegroup(basis_elements=[['Li', 'X', 'V'], ['Li', 'X', 'V'],
                                         ['O', 'F']],
                         basis=[(0.000000, 0.000000, 0.000000),
                                (0.333333, 0.666667, 0.000000),
                                (0.333333, 0.000000, 0.250000)],
                         spacegroup=167,
                         cellpar=[5.123, 5.123, 13.005, 90., 90., 120.],
                         size=[1, 1, 1],
                         conc_args={"conc_ratio_min_1": [[0, 2, 1], [2, 1]],
                                    "conc_ratio_max_1": [[2, 0, 1], [2, 1]]},
                         db_name=db_name,
                         grouped_basis=[[0, 1], [2]],
                         max_cluster_size=2,
                         max_cluster_dist=5.0)
    os.remove(db_name)


# test_spgroup_217()
test_grouped_basis_with_large_dist()
