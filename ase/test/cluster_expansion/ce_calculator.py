from ase.calculators.cluster_expansion import ClusterExpansion
from ase.ce import BulkCrystal
from ase.ce import CorrFunction
from random import shuffle


def generate_ex_eci(bc):
    cf = CorrFunction(bc)
    cf = cf.get_cf(bc.atoms)
    eci = {key: -0.001 for key in cf}
    return eci


def all_cf_match(cf_dict, calc):
    tol = 1E-4
    for i, name in enumerate(calc.cluster_names):
        print(cf_dict[name], calc.cf[i])
        assert abs(cf_dict[name] - calc.cf[i]) < tol


def test_update_correlation_functions(n_trial_configs=20):
    db_name = "aucu_binary_test.db"
    conc_args = {
        "conc_ratio_min_1": [[1, 0]],
        "conc_ratio_max_1": [[0, 1]]
    }
    bc_setting = BulkCrystal(crystalstructure="fcc", a=4.05,
                             basis_elements=[["Au", "Cu"]], size=[3, 3, 3], conc_args=conc_args,
                             db_name=db_name)
    cf = CorrFunction(bc_setting)
    atoms = bc_setting.atoms

    eci = generate_ex_eci(bc_setting)

    # Insert some Au and Cu elements
    for i in range(int(len(atoms) / 2)):
        atoms[i].symbol = "Au"
        atoms[-i - 1].symbol = "Cu"
    init_cf = cf.get_cf(atoms)

    calc = ClusterExpansion(bc_setting, cluster_name_eci=eci, init_cf=init_cf)
    atoms.set_calculator(calc)

    indices = range(len(atoms))
    symbols = [atom.symbol for atom in atoms]
    for _ in range(n_trial_configs):
        shuffle(symbols)
        for i, symb in enumerate(symbols):
            atoms[i].symbol = symb

        # The calculator should update its correlation functions
        # when the energy is computed
        energy = atoms.get_potential_energy()
        brute_force_cf = cf.get_cf(atoms)
        all_cf_match(brute_force_cf, calc)


test_update_correlation_functions(n_trial_configs=20)
