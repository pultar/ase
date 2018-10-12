import numpy as np
from ase.clease.concentration import Concentration


def test1():
    basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
    conc = Concentration(basis_elements=basis_elements)
    conc = conc.get_random_concentration()
    sum1 = np.sum(conc[:3])
    assert abs(sum1 - 1) < 1E-9
    sum2 = np.sum(conc[3:])
    assert abs(sum2 - 1) < 1E-9


def fixed_composition():
    basis_elements = [['Li', 'Ru'], ['O', 'X']]
    A_eq = [[0, 3, 0, 0], [0, 0, 0, 2]]
    b_eq = [1, 1]
    conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq)
    rand = conc.get_random_concentration()
    assert np.allclose(rand, np.array([2./3, 1./3, 0.5, 0.5]))


def test2():
    basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
    A_eq = [[0, 3, 0, 0, 0]]
    b_eq = [1]
    A_lb = [[0, 0, 0, 3, 0]]
    b_lb = [2]

    conc = Concentration(basis_elements=basis_elements, A_eq=A_eq, b_eq=b_eq,
                         A_lb=A_lb, b_lb=b_lb)
    # print(conc.get_conc_min_component(4))

    conc = conc.get_random_concentration()
    print(conc)
    sum1 = np.sum(conc[:3])
    assert abs(sum1 - 1) < 1E-9
    sum2 = np.sum(conc[3:])
    assert abs(sum2 - 1) < 1E-9


test1()
fixed_composition()
for _ in range(10):
    test2()
