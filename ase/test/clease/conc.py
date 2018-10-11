from ase.clease.concentration import Concentration


def test1():
    basis_elements = [['Li', 'Ru', 'X'], ['O', 'X']]
    conc = Concentration(basis_elements=basis_elements)
    print(conc.get_random_concentration())


test1()
