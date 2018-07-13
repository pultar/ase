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
        if self.num_unique_elements == 2:
            d0_0 = 1.
        elif self.num_unique_elements == 3:
            d0_0 = np.sqrt(3. / 2)
            c0_1 = np.sqrt(2)
            c1_1 = -3 / np.sqrt(2)
        elif self.num_unique_elements == 4:
            d0_0 = np.sqrt(2. / 5)
            c0_1 = -5. / 3
            c1_1 = 2. / 3
            d0_1 = -17. / (3 * np.sqrt(10))
            d1_1 = np.sqrt(5. / 2) / 3
        elif self.num_unique_elements == 5:
            d0_0 = 1. / np.sqrt(2)
            c0_1 = -1 * np.sqrt(10. / 7)
            c1_1 = np.sqrt(5. / 14)
            d0_1 = -17. / (6 * np.sqrt(2))
            d1_1 = 5. / (6 * np.sqrt(2))
            c0_2 = 3 * np.sqrt(2. / 7)
            c1_2 = -155. / (12 * np.sqrt(14))
            c2_2 = 15 * np.sqrt(7. / 2) / 12
        elif self.num_unique_elements == 6:
            d0_0 = np.sqrt(3. / 14)
            c0_1 = -np.sqrt(2)
            c1_1 = 3. / (7 * np.sqrt(2))
            d0_1 = -7. / 6
            d1_1 = 1. / 6
            c0_2 = 9 * np.sqrt(3. / 2) / 5
            c1_2 = -101. / (28 * np.sqrt(6))
            c2_2 = 7. / (20 * np.sqrt(6))
            d0_2 = 131. / (15 * np.sqrt(4))
            d1_2 = -7 * np.sqrt(7. / 2) / 12
            d2_2 = np.sqrt(7. / 2) / 20
        else:
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
