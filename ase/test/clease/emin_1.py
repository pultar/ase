from ase.clease import CEBulk, CorrFunction, Concentration, NewStructures, Evaluate
from ase.clease.tools import wrap_and_sort_by_position, reconfigure
from ase.calculators.vasp import Vasp2
from ase.db import connect
from ase.visualize import view
from sys import argv

db_name = 'emin_1_temp.db'
basis_elements = basis_elements = [['Li', 'Mn', 'X'],
                                   ['O', 'V']]
A_eq = [[0, 3, 0, 0, 0]]
b_eq = [1]
A_lb = [[0, 0, 0, 3, 0]]
b_lb = [2]
conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq,
                     A_lb=A_lb, b_lb=b_lb)

setting = CEBulk(crystalstructure='rocksalt',
                 a=4.1,
                 #  supercell_factor=27,
                 size=[1,1,1],
                 concentration=conc,
                 db_name=db_name,
                 max_cluster_size=4,
                 max_cluster_dia=[7.0, 7.0, 4.0],
                 cubic=False,
                 basis_function='sanchez')

# reconfigure(setting)
# exit()

# ns = NewStructures(setting=setting, generation_number=1, struct_per_gen=5)
# ns.generate_probe_structure(approx_mean_var=True)

with open('emin_1_cluster_names.txt') as f:
    clusters = f.readlines()
clusters = [x.strip() for x in clusters]
# print(clusters)



eval = Evaluate(setting=setting, cluster_names=clusters, fitting_scheme='l2',
                alpha=1E-7)
cluster_name_eci = eval.get_cluster_name_eci()

atoms = connect('emin_1.db').get(53).toatoms()

ns = NewStructures(setting=setting, generation_number=1, struct_per_gen=2)
ns.generate_Emin_structure(atoms=atoms, init_temp=10000, final_temp=1,
                           num_temp=3, num_steps_per_temp=100,
                           cluster_name_eci=cluster_name_eci,
                           random_composition=True)
