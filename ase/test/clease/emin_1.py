from ase.clease import CEBulk, CorrFunction, Concentration, NewStructures, Evaluate
from ase.clease.tools import wrap_and_sort_by_position
from ase.calculators.vasp import Vasp2
from ase.db import connect
from ase.visualize import view
from sys import argv

db_name = 'emin_1_temp.db'
basis_elements = basis_elements = [['Li', 'Mn', 'X'],
                                   ['O', 'X']]
A_eq = [[0, 3, 0, 0, 0]]
b_eq = [1]
A_lb = [[0, 0, 0, 3, 0]]
b_lb = [2]
conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq,
                     A_lb=A_lb, b_lb=b_lb)

setting = CEBulk(crystalstructure='rocksalt',
                 a=4.1,
                 #supercell_factor=27,
                 size=[1,1,1],
                 concentration=conc,
                 db_name=db_name,
                 max_cluster_size=4,
                 max_cluster_dia=[7.0, 7.0, 4.0],
                 cubic=False,
                 basis_function='sanchez')
setting.reconfigure_settings()
# setting.view_clusters()
# setting._set_active_template_by_uid(0)
# print(setting.cluster_info_by_name('c3_06nn_3'))
# print(len(setting.cluster_names))
cf = CorrFunction(setting)
cf.reconfig_db_entries()

exit()

for i in range(setting.template_atoms.num_templates):
    setting._set_active_template_by_uid(i)
    print(i, setting.template_atoms.get_size()[i])
    
    cluster_names_setting = sorted(setting.cluster_names)

    db = connect(db_name)

    kvp = connect(db_name).get(15).key_value_pairs
    cluster_names_db = []
    for key in kvp.keys():
        if key.startswith(('c0', 'c1', 'c2', 'c3', 'c4', 'c5',
                           'c6', 'c7', 'c8', 'c9')):
            cluster_names_db.append(key)
    cluster_names_db = sorted(cluster_names_db)

    # print(len(cluster_names_db))
    print(len(cluster_names_setting))
    assert len(cluster_names_setting) == len(cluster_names_db)
    # for i in range(len(cluster_names_setting)):
    #     assert cluster_names_setting[i] == cluster_names_db[i]

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