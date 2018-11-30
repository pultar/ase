import numpy as np
from ase.clease import CECrystal, Concentration, NewStructures, Evaluate
from ase.io import read
from ase.visualize import view
from ase.db import connect
from ase.clease.tools import wrap_and_sort_by_position
import json


if __name__ == '__main__':
    basis_elements = [['O', 'X'], ['O', 'X'], ['O', 'X'], ['Ta']]
    grouped_basis = [[0, 1, 2], [3]]
    # A_eq = [[0, 0, 1]]
    # b_eq = [1]
    A_lb = [[5, 0, 0]]
    b_lb = [4]
    conc = Concentration(basis_elements=basis_elements,
                         grouped_basis=grouped_basis,
                         A_lb=A_lb, b_lb=b_lb)

    basis = [(0., 0., 0.),
             (0.3894, 0.1405, 0.),
             (0.201, 0.3461, 0.5),
             (0.2244, 0.3821, 0.)]
    spacegroup = 55
    cellpar = [6.25, 7.4, 3.83, 90, 90, 90]
    supercell_factor = 12
    db_name = 'emin_2.db'
    max_cluster_size = 4
    max_cluster_dia = [5.0, 5.0, 4.5]

    setting = CECrystal(basis=basis, spacegroup=spacegroup, cellpar=cellpar,
                        supercell_factor=supercell_factor, concentration=conc,
                        db_name=db_name,
                        max_cluster_size=max_cluster_size,
                        max_cluster_dia=max_cluster_dia,
                        basis_function='sanchez', ignore_background_atoms=True)

    # double = 7.0
    # triple = 3.0
    # quad = 3.0

    # eval = Evaluate(setting=setting, parallel=False,
    #                 fitting_scheme='l1', max_cluster_size=4,
    #                 max_cluster_dia=[double, triple, quad])
    # alpha = eval.plot_CV(alpha_min=1E-7, alpha_max=1.0, num_alpha=200)
    # eval.set_fitting_scheme(fitting_scheme='l1', alpha=alpha)
    # eval.save_cluster_name_eci()
    with open('cluster_eci_2.json') as json_data:
        cluster_name_eci = json.load(json_data)

    # db = connect(db_name)
    # atoms = []
    # for row in db.select(gen=10):
    #     atoms.append(row.toatoms())
    # ns = NewStructures(setting=setting, generation_number=15, struct_per_gen=5)
    # # ns.generate_probe_structure(approx_mean_var=True, num_steps_per_temp=1000)
    # ns.generate_Emin_structure(atoms=atoms,
    #                            random_composition=True,
    #                            cluster_name_eci=cluster_name_eci)
