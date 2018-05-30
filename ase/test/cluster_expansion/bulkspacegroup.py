from ase.ce import BulkSpacegroup
import os

"""
This test should try to initiate all possible cases of a BulkSpacegroup
it should not be nessecary to call GenerateStructures, Evaluate etc.
with BulkSpacegroup as this is tested in run_ce.py for BulkCrystal
"""

def test_spgroup_217():
    """
    Initiate a BulkSpacegroup to describe gamma phae
    """
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Al","Al","Al","Al"]
    #symbols = ["Al","Al","Al","Al"]
    basis = [(0,0,0),(0.324,0.324,0.324),(0.3582,0.3582,0.0393),(0.0954,0.0954,0.2725)]
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]]
    }
    db_name = "test_db_binary_almg.db"
    basis_elements = [["Al","Mg"],["Al","Mg"],["Al","Mg"],["Al","Mg"]]

    # Test with grouped basis
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=217, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=4, db_name=db_name, size=[1,1,1], grouped_basis=[[0,1,2,3]], max_cluster_dist=5.0 )
    os.remove(db_name)

    conc_args = {
        "conc_ratio_min_1":[[1,0], [1,0],[1,0],[1,0]],
        "conc_ratio_max_1":[[0,1], [0,1],[0,1],[0,1]]
    }
    basis_elements = [["Al","Mg"],["Si","Mg"],["Cu","Mg"],["Zn","Mg"]]
    # Test without grouped basis
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=217, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=4, db_name=db_name, size=[1,1,1], max_cluster_dist=5.0 )
    os.remove(db_name)

test_spgroup_217()
