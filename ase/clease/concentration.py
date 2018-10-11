"""Class containing a manager for setting up concentrations of species."""
import numpy as np
from scipy.optimize import linprog
from scipy.linalg import null_space


class Concentration(object):
    def __init__(self, basis_elements=None, A_ub=None, b_up=None, A_eq=None,
                 b_eq=None):
        num_implicit_eq = len(basis_elements)
        self.num_concs = len([x for sub in basis_elements for x in sub])

        num_usr_ub = 0
        if A_ub is not None:
            num_usr_ub = A_ub.shape[0]
        num_ub = self.num_concs + num_usr_ub
        self.A_ub = np.zeros((num_ub, self.num_concs), dtype=int)
        self.b_ub = np.zeros(num_ub, dtype=int)

        num_usr_eq = 0
        if A_eq is not None:
            num_usr_eq = A_eq.shape[0]

        num_eq = num_implicit_eq + num_usr_eq
        self.A_eq = np.zeros((num_eq, self.num_concs), dtype=int)
        self.b_eq = np.zeros(num_eq, dtype=int)

        # Ensure concentration in each basis sum to 1
        start_col = 0
        for i, basis in enumerate(basis_elements):
            if start_col + len(basis) >= self.A_eq.shape[1]:
                self.A_eq[i, start_col:] = 1
            else:
                self.A_eq[i, start_col:start_col+len(basis)] = 1
            start_col += len(basis)
            self.b_eq[i] = 1

        # Ensure that we have only positive concentrations
        for i in range(self.num_concs):
            self.A_ub[i, i] = -1

    def get_random_concentration(self):
        """Generate a valid random concentration."""
        satisfied = False
        while not satisfied:
            x = np.random.rand()

        c = 2*np.random.rand(self.num_concs)-1
        print(c)
        opt_res = linprog(c, A_ub=self.A_ub, b_ub=self.b_ub, A_eq=self.A_eq,
                          b_eq=self.b_eq)
        return opt_res["x"]
