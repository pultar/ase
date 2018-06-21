
import os
from ase.ce import BulkCrystal, CorrFunction
from ase.ce.corrFunc import equivalent_deco

db_name = "test.db"
conc_args = {"conc_ratio_min_1": [[1, 0]],
             "conc_ratio_max_1": [[0, 1]]}
bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05,
                         basis_elements=[["Au", "Cu", "Si"]], size=[4, 4, 4],
                         conc_args=conc_args, db_name=db_name)


def test_trans_matrix():
    """Check that the MIC distance between atoms are correct"""
    atoms = bc_setting.atoms
    tm = bc_setting.trans_matrix
    ref_dist = atoms.get_distance(0, 1, mic=True)
    for indx in range(len(atoms)):
        dist = atoms.get_distance(indx, tm[indx, 1], mic=True)
        assert abs(dist - ref_dist) < 1E-5


def get_mic_dists(atoms, cluster):
    dists = []
    for indx in cluster:
        dist = atoms.get_distances(indx, cluster, mic=True)
        dists.append(dist)
    return dists


def test_order_indep_ref_indx():
    """Check that the order of the elements are independent of the
    reference index. This does only apply for clusters with only inequivalent
    sites"""

    for size in range(3, len(bc_setting.cluster_indx[0])):
        for i in range(0, len(bc_setting.cluster_indx[0][size])):
            if bc_setting.cluster_eq_sites[0][size][i]:
                # The cluster contains symmetrically equivalent sites
                # and then this test does not apply
                continue
            cluster = bc_setting.cluster_indx[0][size][i]
            cluster_order = bc_setting.cluster_order[0][size][i]

            init_cluster = [0] + cluster[0]
            init_cluster = [init_cluster[indx] for indx in cluster_order[0]]
            atoms = bc_setting.atoms

            # Make sure that when the other indices in init_cluster are reference
            # indices, the order is the same
            for ref_indx in cluster[0]:
                found_cluster = False
                for subcluster, order in zip(cluster, cluster_order):
                    new_cluster = [ref_indx]
                    for indx in subcluster:
                        trans_indx = bc_setting.trans_matrix[ref_indx, indx]
                        new_cluster.append(trans_indx)

                    # Check if all elements are the same
                    if sorted(new_cluster) == sorted(init_cluster):
                        new_cluster = [new_cluster[indx] for indx in order]
                        found_cluster = True
                        assert init_cluster == new_cluster
                assert found_cluster


def test_interaction_contribution_symmetric_clusters():
    """Test that when one atom is surrounded by equal atoms,
    the contribution from all clusters within one category is
    the same"""
    from ase.build import bulk
    from ase.ce.tools import wrap_and_sort_by_position

    # Create an atoms object that fits with BulkCrystal
    atoms = bulk("Au", crystalstructure="fcc", a=4.05)
    atoms = atoms * (6, 6, 6)
    atoms = wrap_and_sort_by_position(atoms)

    # Put an Si atom at the origin
    atoms[0].symbol = "Si"

    # Define decoration numbers and make sure they are different
    deco = [[], [], [0, 1], [0, 1, 1], [0, 1, 1, 1]]
    cf = CorrFunction(bc_setting)
    bf = bc_setting.basis_functions
    for size in range(2, len(bc_setting.cluster_indx[0])):
        for cat in range(len(bc_setting.cluster_indx[0][size])):
            cluster = bc_setting.cluster_indx[0][size][cat]
            orders = bc_setting.cluster_order[0][size][cat]
            equiv_sites = bc_setting.cluster_eq_sites[0][size][cat]

            equiv_deco = equivalent_deco(deco[size], equiv_sites)
            if len(equiv_deco) == size:
                # Calculate reference contribution for this cluster category
                indices = [0] + cluster[0]
                indices = [indices[indx] for indx in orders[0]]
                ref_sp = 0.0
                counter = 0
                for dec in equiv_deco:
                    ref_sp_temp = 1.0
                    for dec_num, indx in zip(dec, indices):
                        ref_sp_temp *= bf[dec_num][atoms[indx].symbol]
                    counter += 1
                    ref_sp += ref_sp_temp
                ref_sp /= counter

                # Calculate the spin product for this category
                sp, count = cf._spin_product_one_ref_indx(
                    0, atoms, cluster, orders, equiv_sites, deco[size])
                sp /= count
                assert abs(sp - ref_sp) < 1E-4


test_trans_matrix()
test_order_indep_ref_indx()
test_interaction_contribution_symmetric_clusters()
os.remove(db_name)
