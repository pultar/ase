"""Module for setting up pseudospins and basis functions."""
import numpy as np


class BasisFunction(object):
    """Base-class for all Basis Functions."""

    def __init__(self, unique_elements):
        self.unique_elements = unique_elements
        self.num_unique_elements = len(unique_elements)
        self.spin_dict = self.get_spin_dict()
        self.basis_function = self.get_basis_function()

    def get_spin_dict(self):
        """Get spin dictionary."""
        pass

    def get_basis_function(self):
        """Get basis function."""
        pass


class Sanchez(BasisFunction):
    """Pseudospin and basis function from Sanchez et al.

    Sanchez, J. M., Ducastelle, F., & Gratias, D. (1984).
    Generalized cluster description of multicomponent systems.
    Physica A: Statistical Mechanics and Its Applications, 128(1–2), 334–350.
    """

    def __init__(self):
        BasisFunction.__init__()

    def get_spin_dict(self):
        """Define pseudospins for all consistuting elements."""
        # Find odd/even
        spin_values = []
        if self.num_unique_elements % 2 == 1:
            highest = (self.num_unique_elements - 1) / 2
        else:
            highest = self.num_unique_elements / 2
        # Assign spin value for each element
        while highest > 0:
            spin_values.append(highest)
            spin_values.append(-highest)
            highest -= 1
        if self.num_unique_elements % 2 == 1:
            spin_values.append(0)

        spin_dict = {}
        for x in range(self.num_unique_elements):
            spin_dict[self.unique_elements[x]] = spin_values[x]
        return spin_dict

    def _get_basis_functions(self):
        """Create basis functions to guarantee the orthonormality."""
        # d0_0 = np.array([1.0, np.sqrt(3. / 2), np.sqrt(2. / 5),
        #                  1. / np.sqrt(2), np.sqrt(3. / 14)])
        # d0_1 = np.array([0.0, 0.0,  -17. / (3 * np.sqrt(10)),
        #                  -17. / (6 * np.sqrt(2)), -7. / 6])
        # d1_1 = np.array([0.0, 0.0, np.sqrt(5. / 2) / 3,
        #                  5. / (6 * np.sqrt(2)), 1. / 6])
        # d0_2 = np.array([0.0, 0.0, 0.0, 0.0, 131. / (15 * np.sqrt(4))])
        # d1_2 = np.array([0.0, 0.0, 0.0, 0.0, -7 * np.sqrt(7. / 2) / 12])
        # d2_2 = np.array([0.0, 0.0, 0.0, 0.0, np.sqrt(7. / 2) / 20])

        coeff_d = np.array([
            [1.0, np.sqrt(3. / 2), np.sqrt(2. / 5), 1. / np.sqrt(2),
             np.sqrt(3. / 14)],
            [0.0, 0.0,  -17. / (3 * np.sqrt(10)), -17. / (6 * np.sqrt(2)),
             -7. / 6],
            [0.0, 0.0, np.sqrt(5. / 2) / 3, 5. / (6 * np.sqrt(2)), 1. / 6],
            [0.0, 0.0, 0.0, 0.0, 131. / (15 * np.sqrt(4))],
            [0.0, 0.0, 0.0, 0.0, -7 * np.sqrt(7. / 2) / 12],
            [0.0, 0.0, 0.0, 0.0, np.sqrt(7. / 2) / 20]])

        # c0_1 = np.array([0.0, np.sqrt(2), -5. / 3,
        #                  -1 * np.sqrt(10. / 7), -np.sqrt(2)])
        # c1_1 = np.array([0.0,  -3 / np.sqrt(2), 2. / 3,
        #                  np.sqrt(5. / 14), 3. / (7 * np.sqrt(2))])
        # c0_2 = np.array([0.0, 0.0, 0.0,
        #                  3 * np.sqrt(2. / 7), 9 * np.sqrt(3. / 2) / 5])
        # c1_2 = np.array([0.0, 0.0, 0.0,
        #                  -155. / (12 * np.sqrt(14)), -101. / (28 * np.sqrt(6))])
        # c2_2 = np.array([0.0, 0.0, 0.0,
        #                  15 * np.sqrt(7. / 2) / 12, 7. / (20 * np.sqrt(6))])

        coeff_c = np.array([
            [0.0, np.sqrt(2), -5. / 3, -1 * np.sqrt(10. / 7), -np.sqrt(2)],
            [0.0,  -3 / np.sqrt(2), 2. / 3, np.sqrt(5. / 14),
             3. / (7 * np.sqrt(2))],
            [0.0, 0.0, 0.0, 3 * np.sqrt(2. / 7), 9 * np.sqrt(3. / 2) / 5],
            [0.0, 0.0, 0.0, -155. / (12 * np.sqrt(14)),
             -101. / (28 * np.sqrt(6))],
            [0.0, 0.0, 0.0, 5 * np.sqrt(7. / 2) / 12, 7. / (20 * np.sqrt(6))]])

        num_elements = self.num_unique_elements
        if num_elements > 6:
            raise ValueError("only compounds consisting of 2 to 6 types of"
                             " elements are supported")

        bf_list = []

        bf = {}
        for key, value in self.spin_dict.items():
            bf[key] = d0_0 * value
        bf_list.append(bf)

        if self.num_unique_elements > 2:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = c0_1 + (c1_1 * value**2)
            bf_list.append(bf)

        if self.num_unique_elements > 3:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = d0_1 * value + (d1_1 * (value**3))
            bf_list.append(bf)

        if self.num_unique_elements > 4:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = c0_2 + (c1_2 * (value**2)) + (c2_2 * (value**4))
            bf_list.append(bf)

        if self.num_unique_elements > 5:
            bf = {}
            for key, value in self.spin_dict.items():
                bf[key] = d0_2 + (d1_2 * (value**3)) + (d2_2 * (value**5))
            bf_list.append(bf)

        return bf_list


class VandeWalle(BasisFunction):
    """Pseudospin and basis function from van de Walle.

    van de Walle, A. (2009).
    Multicomponent multisublattice alloys, nonconfigurational entropy and other
    additions to the Alloy Theoretic Automated Toolkit. Calphad, 33(2),
    266–278.
    """

    def __init__(self):
        BasisFunction.__init__()

    def get_spin_dict(self):
        """Define pseudospins for all consistuting elements."""
        spin_values = range(self.num_unique_elements)

        spin_dict = {}
        for x in range(self.num_unique_elements):
            spin_dict[self.unique_elements[x]] = spin_values[x]
        return spin_dict

    def get_basis_function(self):
        if alpha == 0:
            return 1





# def kronecker(i, j):
#     if i == j:
#         return 1
#     return 0
#
# spin_dict = {"Cu":0, "Au":1, "Zn":2}
#
# bf1 = kronecker(sigma, 0)
# bf2 = kronecker(sigma, 1)
# bf3 = kronecker(sigma, 2)
#
# def get_basis_function():
#     basis_function = []
#     basis_function.append(bf1)
#
#     for i in range(num_elemtns):
#         for key, value in spin_dict:
#
#
#     [{"Cu":1, "Au":0, "Zn":0}, {"Cu":0, "Au":1, "Zn":0}, {"Cu":0, "Au":0, "Zn":1}]
