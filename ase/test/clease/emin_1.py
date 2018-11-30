from ase.clease import CEBulk, CorrFunction, Concentration, NewStructures, Evaluate
from ase.clease.tools import wrap_and_sort_by_position
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
# setting.reconfigure_settings()
# setting.view_clusters()
# setting._set_active_template_by_uid(4)
# print(setting.cluster_info_by_name('c4_02nn_1'))
# exit()
print(setting.cluster_info_by_name('c3_03nn_0')[0]['indices'])
# print(len(setting.cluster_info_by_name('c4_02nn_1')[0]['indices']))
# print(setting.cluster_info_by_name('c4_02nn_1')[0]['max_cluster_dia'])
# print(setting.cluster_info_by_name('c3_06nn_2')[0]['descriptor'])
# print(setting.cluster_info_by_name('c3_06nn_3')[0]['descriptor'])
# print(len(setting.cluster_names))
# cf = CorrFunction(setting)
# cf.reconfig_db_entries()

# exit()

# for i in range(setting.template_atoms.num_templates):
#     setting._set_active_template_by_uid(i)
#     print(i, setting.template_atoms.get_size()[i])

#     cluster_names_setting = sorted(setting.cluster_names)

#     db = connect(db_name)

#     kvp = connect(db_name).get(15).key_value_pairs
#     cluster_names_db = []
#     for key in kvp.keys():
#         if key.startswith(('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
#                            'c6', 'c7', 'c8', 'c9')):
#             cluster_names_db.append(key)
#     cluster_names_db = sorted(cluster_names_db)

#     # print(len(cluster_names_db))
#     print(len(cluster_names_setting))
#     assert len(cluster_names_setting) == len(cluster_names_db)
#     for i in range(len(cluster_names_setting)):
#         assert cluster_names_setting[i] == cluster_names_db[i]

# db = connect(db_name)
# ids = [row.id for row in db.select(struct_type='initial')]

# cluster_names_matrix = []
# for id in ids:
#     kvp = connect(db_name).get(id).key_value_pairs
#     cluster_names_db = []
#     for key in kvp.keys():
#         if key.startswith(('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
#                            'c6', 'c7', 'c8', 'c9')):
#             cluster_names_db.append(key)
#     cluster_names_db = sorted(cluster_names_db)
#     cluster_names_matrix.append(cluster_names_db)





# ids = [row.id for row in db.select(struct_type='initial')]
# for id in ids:
#     kvp = connect(db_name).get(id).key_value_pairs
#     cluster_names_db = []
#     for key in kvp.keys():
#         if key.startswith(('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
#                            'c6', 'c7', 'c8', 'c9')):
#             cluster_names_db.append(key)
#     cluster_names_db = sorted(cluster_names_db)

#     assert len(cluster_names_setting) == len(cluster_names_db)
#     for i in range(len(cluster_names_setting)):
#         assert cluster_names_setting[i] == cluster_names_db[i]
#         print('c3_06nn_3_200' in cluster_names_setting, 'c3_06nn_3_200' in cluster_names_db)

# double = 7.0
# triple = 3.0

# eval = Evaluate(setting=setting, parallel=True, num_core=10,
#                 fitting_scheme='l1', max_cluster_size=4,
#                 max_cluster_dia=[double, triple, 0.0])
# alpha = eval.plot_CV(alpha_min=1E-7, alpha_max=1.0, num_alpha=200,
#                      scale='log')
# eval.set_fitting_scheme(fitting_scheme='l1', alpha=alpha)
# # eval.plot_fit(interactive=True)
# eval.save_cluster_name_eci()

# ns = NewStructures(setting=setting, generation_number=1, struct_per_gen=5)
# ns.generate_probe_structure(approx_mean_var=True)

with open('emin_1_cluster_names.txt') as f:
    clusters = f.readlines()
clusters = [x.strip() for x in clusters]
# print(clusters)

eval = Evaluate(setting=setting, cluster_names=clusters, fitting_scheme='l2',
                alpha=1E-7)
cluster_name_eci = eval.get_cluster_name_eci()
# print(cluster_name_eci)

atoms = connect(db_name).get(53).toatoms()
from ase.calculators.clease import Clease
calc = Clease(setting, cluster_name_eci=cluster_name_eci)
corr_func = CorrFunction(setting)
#atoms.set_calculator(calc)

from ase.visualize import view
#view(atoms)
print(setting.basis_functions)
atoms = setting.atoms.copy()
atoms.set_calculator(calc)
# view(atoms)
# exit()
atoms[1].symbol = 'Mn'
#atoms[9].symbol = 'X'


# atoms[12].symbol = "Li"
# atoms[2].symbol = "Mn"
# view(setting.atoms)
init_cf = calc.get_cf_dict()
energy = atoms.get_potential_energy()
cf_brute = corr_func.get_cf_by_cluster_names(atoms, cluster_name_eci)
cf_calc = calc.get_cf_dict()
for k in cf_brute.keys():
    indx = calc.cluster_names.index(k)
    calc_change = cf_calc[k] - init_cf[k]
    brute_change = cf_brute[k] - init_cf[k]
    print(k, init_cf[k], brute_change, calc_change, calc_change/brute_change)

# ns = NewStructures(setting=setting, generation_number=1, struct_per_gen=5)
# ns.generate_Emin_structure(atoms=atoms, init_temp=10000, final_temp=1,
#                            num_temp=3, num_steps_per_temp=100,
#                            cluster_name_eci=cluster_name_eci,
#                            random_composition=True)
# 3
# 24
# 12
# 36
#
# [0 1 1]
#
# (x0*x1*x1 + x1*x0*x0)
#
# [0 1 2]
# (x0*x1*x2 + x1*x0*x2 + x2*x1*x0)
