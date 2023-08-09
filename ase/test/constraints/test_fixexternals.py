def test_fixexternals():
    import numpy as np
    from ase.calculators.emt import EMT
    from ase.optimize import BFGS
    from ase.constraints import FixExternals
    from ase.build import fcc111, add_adsorbate, molecule

    def sort_ivec(ivec):
        tmp_ivec = np.zeros([3, 3])
        cartesian_basis = np.diag(np.ones(3))
        for i in range(3):
            d1 = np.dot(ivec[:, 0], cartesian_basis[:, i])
            d2 = np.dot(ivec[:, 1], cartesian_basis[:, i])
            d3 = np.dot(ivec[:, 2], cartesian_basis[:, i])
            if abs(d1) > abs(d2) and abs(d1) > abs(d3):
                if d1 <= 0:
                    tmp_ivec[:, i] = -ivec[:, 0]
                else:
                    tmp_ivec[:, i] = ivec[:, 0]
                ivec[:, 0] = 0
            if abs(d2) > abs(d1) and abs(d2) > abs(d3):
                if d2 <= 0:
                    tmp_ivec[:, i] = -ivec[:, 1]
                else:
                    tmp_ivec[:, i] = ivec[:, 1]
                ivec[:, 1] = 0
            if abs(d3) > abs(d2) and abs(d3) > abs(d1):
                if d3 <= 0:
                    tmp_ivec[:, i] = -ivec[:, 2]
                else:
                    tmp_ivec[:, i] = ivec[:, 2]
                ivec[:, 2] = 0
        return tmp_ivec

    size = [3, 3, 4]
    syst = fcc111(symbol='Cu', size=size, a=3.58)
    adsorbate = molecule('CH3OH')
    add_adsorbate(syst, adsorbate, 2.5, 'ontop')
    syst.center(vacuum=8.5, axis=2)
    indices = [36, 37, 38, 39, 40, 41]
    init_ads = syst[indices].copy()
    init_ads_ivecval = init_ads.get_moments_of_inertia(vectors=True)
    init_ads_ivec = sort_ivec(np.transpose(init_ads_ivecval[1]))
    init_ads_com = init_ads.get_center_of_mass()
    c = FixExternals(syst, indices)
    syst.set_constraint(c)
    syst.calc = EMT()
    dyn = BFGS(syst)
    dyn.run(fmax=0.05)
    final = syst.copy()
    del final.constraints
    final_ads = final[indices]
    final_ads_ivecval = final_ads.get_moments_of_inertia(vectors=True)
    final_ads_ivec = sort_ivec(np.transpose(final_ads_ivecval[1]))
    final_ads_com = final_ads.get_center_of_mass()
    assert np.max(abs(final_ads_ivec - init_ads_ivec)) < 1e-8
    assert np.max(abs(final_ads_com - init_ads_com)) < 1e-8
