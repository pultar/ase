"""Class containing a manager for setting up concentrations of species."""
import numpy as np
from scipy.optimize import minimize
from collections import OrderedDict


class Concentration(object):
    def __init__(self, basis_elements=None, A_lb=None, b_lb=None, A_eq=None,
                 b_eq=None):
        self.basis_elements = basis_elements
        num_implicit_eq = len(basis_elements)
        self.num_concs = len([x for sub in basis_elements for x in sub])
        self.fixed_element_constraint_added = False

        num_usr_lb = 0
        if A_lb is not None:
            A_lb = np.array(A_lb)
            num_usr_lb = A_lb.shape[0]
        self.A_lb = np.zeros((self.num_concs, self.num_concs), dtype=int)
        self.b_lb = np.zeros(self.num_concs, dtype=int)

        num_usr_eq = 0
        if A_eq is not None:
            A_eq = np.array(A_eq)
            num_usr_eq = A_eq.shape[0]

        self.A_eq = np.zeros((num_implicit_eq, self.num_concs), dtype=int)
        self.b_eq = np.zeros(num_implicit_eq, dtype=int)

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
            self.A_lb[i, i] = 1

        if num_usr_eq > 0:
            self.add_usr_defined_eq_constraints(A_eq, b_eq)

        if num_usr_lb > 0:
            self.add_usr_defined_ineq_constraints(A_lb, b_lb)

    def _remove_redundant_entries(self, A, b):
        for row_num in range(A.shape[0]-1, -1, -1):
            for prev_row in range(row_num-1):
                cond1 = np.allclose(A[row_num, :], A[prev_row, :])
                cond2 = abs(b[row_num] - b[prev_row]) < 1E-9
                if cond1 and cond2:
                    A = np.delete(A, row_num, axis=0)
                    b = np.delete(b, row_num)
                    break
        return A, b

    def add_usr_defined_eq_constraints(self, A_eq, b_eq):
        """Add user defined constraints."""
        self.A_eq = np.vstack((self.A_eq, A_eq))
        self.b_eq = np.append(self.b_eq, b_eq)
        self.A_eq, self.b_eq = self._remove_redundant_entries(self.A_eq,
                                                              self.b_eq)

    def add_usr_defined_ineq_constraints(self, A_lb, b_lb):
        """Add the user defined constraints."""
        self.A_lb = np.vstack((self.A_lb, A_lb))
        self.b_lb = np.append(self.b_lb, b_lb)
        self.A_lb, self.b_lb = self._remove_redundant_entries(self.A_lb,
                                                              self.b_lb)

    def set_conc_ranges(self, ranges):
        """Set concentration based on lower and upper bound

        Arguments:
        ==========
            ranges - nested list of tuples with the same shape as
                     basis_elements.
                     If basis_elements is [["Li", "Ru", "X"], ["O", "X"]],
                     ranges coulde be
                     [[(0, 1), (0.2, 0.5), (0, 1)], [(0, 0.8), (0, 0.2)]]
        """
        flatten_rng = [item for sublist in ranges for item in sublist]
        A_lb = np.zeros((2*self.num_concs, self.num_concs))
        b_lb = np.zeros(2*self.num_concs)
        for i, item in enumerate(flatten_rng):
            A_lb[2*i, i] = 1
            A_lb[2*i+1, i] = -1
            b_lb[2*i] = item[0]
            b_lb[2*i+1] = -item[1]
        self.add_usr_defined_ineq_constraints(A_lb, b_lb)

    def _get_integers(self, string, variable_range=None):
        """Extract all the integers from a string."""
        if variable_range is None:
            variable_symbols = []
        else:
            variable_symbols = list(variable_range.keys())
        signs = {"+": 1, "-": -1}
        integers = []
        current_indx = 0
        sum_of_variable_coeff = {k: 0 for k in variable_symbols}
        active_sign = 1
        while current_indx < len(string):
            connected_to_symbol = False
            if string[current_indx] in signs.keys():
                active_sign = signs[string[current_indx]]
            elif string[current_indx] == "<":
                active_sign = 1

            if string[current_indx].isdigit():
                substr = string[current_indx]
                # find how many consecutive digits are there
                indx = current_indx+1
                while indx < len(string):
                    # end of integer found
                    if not string[indx].isdigit():
                        # case where integer followed by a variable symbol
                        # (e.g., 4x)
                        if string[indx] in variable_symbols:
                            connected_to_symbol = True
                        break
                    substr += string[indx]
                    indx += 1

                # got an integer number
                if not connected_to_symbol:
                    integers.append(int(substr))
                    current_indx = indx
                else:
                    symbol = string[indx]
                    sum_of_variable_coeff[symbol] += active_sign*int(substr)
                    current_indx = indx + 1

            # case where variable symbol is defined without integer
            elif string[current_indx] in variable_symbols:
                symbol = string[current_indx]
                sum_of_variable_coeff[symbol] += active_sign
                current_indx += 1
            else:
                current_indx += 1

        # Ensure valid chemical formula
        for k, v in sum_of_variable_coeff.items():
            if v != 0:
                raise ValueError("Invalid formula! {} {}"
                                 "".format(string, sum_of_variable_coeff))
        return integers

    def set_conc_formula_unit(self, formulas=None, variable_range=None):
        """Set concentration based on formula unit strings.

        Arguments:
        =========
        formulas: list of strings
            format of formula strings:
                1. formula string should be provided per basis.
                2. formula string can only have integer numbers.
                3. only one variable is allowed per basis.
                4. each variable should have at least one instance of 'clean'
                   representation (e.g., <x>, <y>)
            e.g., ["Li<x>Ru<1>X<2-x>", "O<3-y>X<y>"], ['Al<4-4x>Mg<3x>Si<x>'']
        variable range: dict
            range of each variable used in formulas.
            key is a string, and the value should be int or float
            e.g., {"x": (0, 2), "y": (0, 0.7)}, {'x': (0., 1.)}
        """
        element_conc = self._parse_formula_unit_string(formulas)
        num_atoms_in_basis = []
        for formula in formulas:
            num_atoms_in_basis.append(
                np.sum(self._get_integers(formula, variable_range)))
        print('\n\n')
        print('element_conc = {}'.format(element_conc))
        print('variable_range = {}'.format(variable_range))
        print('num_atoms_in_basis = {}'.format(num_atoms_in_basis))

        A_eq = []
        b_eq = []
        col = 0

        # Al<4-4x>Mg<3x>Si<x>
        # Al<2-x>Mg<x>Si<x>
        # Al<3-x-2y>Mg<y>Si<y>X<x> {'x': (0, 1), 'y': (0, 1)}

        # Set equality matrix and vector due to one or more elements has one
        # fixed concentration value.
        # Note: element_conc is an OrderedDict so k, v pairs come in the
        # same order they were added
        for basis, basis_elem in enumerate(element_conc):
            for k, v in basis_elem.items():
                if v.isdigit():
                    row = np.zeros(self.num_concs)
                    row[col] = num_atoms_in_basis[basis]
                    A_eq.append(row)
                    b_eq.append(int(v))
                col += 1

        # Extract relations between elements (e.g., <x> and <2x>)
        num_elements_with_symbol = {k: 0 for k in variable_range.keys()}
        for sym in variable_range.keys():
            for basis_elem in element_conc:
                for element_symbol, conc in basis_elem.items():
                    if conc.find(sym) != -1:
                        num_elements_with_symbol[sym] += 1

        print('num_elements_with_symbol = {}'.format(num_elements_with_symbol))
        reference_elements = {}

        for sym in variable_range.keys():
            num_additional_constraints = num_elements_with_symbol[sym] - 2
            if num_additional_constraints == 0:
                continue

            # Find the reference element (element containing <x>)
            ref_element = None
            ref_col = None
            col = 0
            for basis_elem in element_conc:
                for k, v in basis_elem.items():
                    if v == sym:
                        ref_element = k
                        reference_elements[sym] = k
                        ref_col = col
                        break
                    col += 1
            print('ref_element = {}'.format(ref_element))
            print('ref_col = {}'.format(ref_col))

            if ref_element is None:
                raise RuntimeError("Did not find reference element for symbol "
                                   "".format(sym))

            # Extract the coefficients of other elements with their
            # concenctrations defined as multiples of the symbol
            # (e.g., <2x>, <3x>)
            ignore_conditions = ["+", "-"]
            col = 0
            for basis_elem in element_conc:
                for k, v in basis_elem.items():
                    if k == ref_element or sym not in v:
                        col += 1
                        continue

                    if any(char in v for char in ignore_conditions):
                        col += 1
                        continue
                    coeff = self._get_coeff(v)
                    row = np.zeros(self.num_concs)
                    row[ref_col] = -coeff
                    row[col] = 1
                    A_eq.append(row)
                    b_eq.append(0)
                    col += 1

        if A_eq:
            A_eq = np.array(A_eq)
            self.add_usr_defined_eq_constraints(A_eq, b_eq)

        print(self.A_eq)
        print(self.b_eq)

        # [0, 3, 0, 0, 0] = [1]
        # []
        # 0 <= x_Li <= 2
        # 0 <= X <= 0.7
        # Calculate number of elements in formula unit
        # x + 1 + 2 - x = tot_1
        # 3 - y + y = tot_2
        # tot_1*x_Ru = 1

    def _get_coeff(self, string):
        """Get the coefficient in front of the symbol.

        Arugments:
        ==========
        string: str
            string of the following form 3x, 10y, 3z etc.
        """
        if not string[0].isdigit():
            return 1

        number = ""
        for character in string:
            if not character.isdigit():
                break
            number += character
        return int(number)


    def _parse_formula_unit_string(self, formulas):
        element_variable = []
        for basis_num, formula in enumerate(formulas):
            split1 = formula.split("<")
            elements = []
            math = []
            for i, substr in enumerate(split1):
                if i == 0:
                    elements.append(substr)
                elif i == len(split1)-1:
                    split2 = substr.split(">")
                    math.append(split2[0])
                else:
                    split2 = substr.split(">")
                    elements.append(split2[1])
                    math.append(split2[0])
            if elements != self.basis_elements[basis_num]:
                raise ValueError("elements in 'formulas' and 'basis_elements' "
                                 "should match.")
            element_variable.append(OrderedDict(zip(elements, math)))
        return element_variable

    def _get_constraints(self):
        constraints = []
        for i in range(self.A_eq.shape[0]):
            new_constraint = {"type": "eq",
                              "fun": equality_constraint,
                              "args": (self.A_eq[i, :], self.b_eq[i])}
            constraints.append(new_constraint)

        for i in range(self.A_lb.shape[0]):
            new_constraint = {"type": "ineq",
                              "fun": inequality_constraint,
                              "args": (self.A_lb[i, :], self.b_lb[i])}
            constraints.append(new_constraint)
        return constraints

    def _add_fixed_element_in_each_basis(self):
        """Add constraints corresponding to the fixing elements in basis."""
        if self.fixed_element_constraint_added:
            return
        from random import choice
        indices = []
        start = 0
        ranges = self.get_individual_comp_range()
        min_range = 0.01
        maxiter = 1000
        for basis in self.basis_elements:
            iteration = 0
            rng = 0.5*min_range
            while rng < min_range and iteration < maxiter:
                iteration += 1
                indx = choice(range(start, start+len(basis)))
                rng = ranges[indx][1] - ranges[indx][0]
            if iteration >= maxiter:
                self.fixed_element_constraint_added = False
                return
            indices.append(indx)
            start += len(basis)

        A = np.zeros((len(indices), self.num_concs))
        b = np.zeros(len(indices))
        for i, indx in enumerate(indices):
            A[i, indx] = 1
            rng = ranges[indx][1] - ranges[indx][0]
            b[i] = np.random.rand()*rng + ranges[indx][0]

        # Add constraints
        self.A_eq = np.vstack((self.A_eq, A))
        self.b_eq = np.append(self.b_eq, b)
        self.fixed_element_constraint_added = True

    def _remove_fixed_element_in_each_basis_constraint(self):
        """Remove the last rows."""
        if not self.fixed_element_constraint_added:
            return
        num_basis = len(self.basis_elements)
        self.A_eq = self.A_eq[:-num_basis, :]
        self.b_eq = self.b_eq[:-num_basis]
        self.fixed_element_constraint_added = False

    def get_individual_comp_range(self):
        """Return the concentration range of each component."""
        ranges = []
        for i in range(self.num_concs):
            xmin = self.get_conc_min_component(i)
            xmax = self.get_conc_max_component(i)
            ranges.append((xmin[i], xmax[i]))
        return ranges

    def get_random_concentration(self):
        """Generate a valid random concentration."""
        assert self.A_eq.shape[0] == len(self.b_eq)

        self._add_fixed_element_in_each_basis()
        # Setup the constraints
        constraints = self._get_constraints()
        x0 = 2.0*np.random.rand(self.num_concs)

        # Find the closest vector to x0 that satisfies all constraints
        opt_res = minimize(objective_random, x0, args=(x0,),
                           method="SLSQP", jac=obj_jac_random,
                           constraints=constraints)
        self._remove_fixed_element_in_each_basis_constraint()
        return opt_res["x"]

    def get_conc_min_component(self, comp):
        """Generate all end points of the composition domain."""
        constraints = self._get_constraints()

        x0 = np.random.rand(self.num_concs)

        # Find the closest vector to x0 that satisfies all constraints
        opt_res = minimize(objective_component_min, x0, args=(comp,),
                           method="SLSQP", jac=obj_jac_component_min,
                           constraints=constraints)
        return opt_res["x"]

    def get_conc_max_component(self, comp):
        """Generate all end points of the composition domain."""
        constraints = self._get_constraints()

        x0 = np.random.rand(self.num_concs)

        # Find the closest vector to x0 that satisfies all constraints
        opt_res = minimize(objective_component_max, x0, args=(comp,),
                           method="SLSQP", jac=obj_jac_component_max,
                           constraints=constraints)
        return opt_res["x"]


# Helper function used by the minimization algorithm
def objective_component_min(x, indx):
    return x[indx]


def obj_jac_component_min(x, indx):
    jac = np.zeros(len(x))
    jac[indx] = np.sign(x[indx])
    return jac


def objective_component_max(x, indx):
    return -x[indx]


def obj_jac_component_max(x, indx):
    jac = np.zeros(len(x))
    jac[indx] = -np.sign(x[indx])
    return jac


def objective_random(x, x0):
    """Return the difference between x and x0."""
    diff = x - x0
    return diff.dot(diff)


def obj_jac_random(x, x0):
    """The jacobian of the objective function."""
    return 2.0*(x-x0)


def equality_constraint(x, vec, rhs):
    """Equality constraint. Return 0 if constraints are satisfied."""
    return vec.dot(x) - rhs


def eq_jac(x, vec, rhs):
    """Jacobian of the equalitu constraint equation."""
    return vec


def inequality_constraint(x, vec, rhs):
    """Inequality constraints. Return a non-negative number if constraints
       are satisfied."""
    return vec.dot(x) - rhs


def ineq_jac(x, vec, rhs):
    """Jacobian of the inequality constraint equations."""
    return vec
