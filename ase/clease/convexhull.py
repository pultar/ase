import numpy as np
import ase.db
import os
from operator import itemgetter
from copy import deepcopy


class ConvexHull(object):
    """
    Evaluates data from database.
    """
    def __init__(self, db_name, select_cond=None, conc_fix_name=None,
                 conc_fix_value=None, atoms_per_fu=None, conc_scale=1.0,
                 conc_var_name=None):
        self.db_name = db_name
        self.atoms_per_fu = atoms_per_fu
        self.select_cond = [('converged', '=', True)]
        if select_cond is not None:
            for cond in select_cond:
                self.select_cond.append(cond)
        self.conc_scale = conc_scale
        self.conc_fix_name = conc_fix_name
        self.conc_fix_value = conc_fix_value
        self.conc_var_name = conc_var_name

        if os.path.isfile(self.db_name):
            self.db = ase.db.connect(self.db_name)

        if self.conc_fix_name is None and self.select_cond is None:
            rows = self.db.select()
        elif self.conc_fix_name is None:
            rows = self.db.select(self.select_cond)
        else:
            self.select_cond.append(('{}'.format(self.conc_fix_name),
                                    '=', self.conc_fix_value))
            rows = self.db.select(self.select_cond)

        conc = []
        for row in rows:
            conc.append(row.get(self.conc_var_name)*self.conc_scale)
        self.max_conc = max(conc)
        self.min_conc = min(conc)
        self.max_conc_energies = self._get_conc_energies(self.max_conc)
        self.min_conc_energies = self._get_conc_energies(self.min_conc)
        self.Emin_max_conc = min(self.max_conc_energies)
        self.Emin_min_conc = min(self.min_conc_energies)

    def _get_conc_energies(self, conc_scaled):
        energies = []
        conc = float(conc_scaled)/self.conc_scale
        select_cond = deepcopy(self.select_cond)
        if select_cond is None:
            select_cond = []
        select_cond.append(('{}'.format(self.conc_var_name), '=', conc))
        rows = self.db.select(select_cond)

        for row in rows:
            energies.append(row.energy * self.atoms_per_fu / row.natoms)

        return np.array(energies)

    def _get_concentration_rel_energy_pairs(self, conditions=None):
        tuples = []
        if conditions is None:
            rows = self.db.select(self.select_cond)
        else:
            select_cond = deepcopy(self.select_cond)
            if select_cond is None:
                select_cond = []
            for x in range(len(conditions)):
                select_cond.append(conditions[x])
            rows = self.db.select(select_cond)

        names = []
        for row in rows:
            c = row.get(self.conc_var_name) * self.conc_scale
            conc_range = self.max_conc - self.min_conc
            dE = (row.energy * self.atoms_per_fu / row.natoms) - \
                 (((c - self.min_conc) / conc_range) * self.Emin_max_conc) - \
                 (((self.max_conc - c) / conc_range) * self.Emin_min_conc)
            tuples.append((c, dE))
            names.append(row.name)
        return tuples, names

    def plot(self, ocv=False, ref=None, ref_energy=None):
        import matplotlib.pyplot as plt
        from ase.clease.interactive_plot import InteractivePlot

        conc_rel_energy, names = self._get_concentration_rel_energy_pairs()
        convex_hull_data = list(zip(*conc_rel_energy))
        conc_values = list(set(convex_hull_data[0]))
        # ----------------------------------------------------
        # find the minimum energy for each concentration value
        # ----------------------------------------------------
        conc_Emin_pairs = []
        for conc in conc_values:
            if conc == self.min_conc or conc == self.max_conc:
                conc_Emin_pairs.append((conc, 0.))
            else:
                # find minimum energy
                min_energy = 1000
                for pair in conc_rel_energy:
                    if pair[0] == conc and pair[1] < min_energy:
                        min_energy = pair[1]
                conc_Emin_pairs.append((conc, min_energy))
        # sort it based on concentration values
        conc_Emin_pairs = sorted(conc_Emin_pairs, key=itemgetter(0))
        # remove the points with Erel higher than zero
        conc_Emin_pairs = [x for x in conc_Emin_pairs if x[1] <= 0.0]

        # -------------------------------------
        # Get minimum energy point to be traced
        # -------------------------------------
        min_points = [min(conc_Emin_pairs, key=itemgetter(1))]

        # scan left
        left = [p for p in conc_Emin_pairs if p[0] < min_points[0][0]]
        while min_points[0] != conc_Emin_pairs[0]:
            slopes = []
            for point in left:
                slopes.append(float((min_points[0][1] - point[1]))
                              / (min_points[0][0] - point[0]))
            min_index = slopes.index(max(slopes))
            min_points = [left[min_index]] + min_points
            left = left[:min_index]

        # scan right
        right = [p for p in conc_Emin_pairs if p[0] > min_points[-1][0]]
        while min_points[-1] != conc_Emin_pairs[-1]:
            slopes = []
            for point in right:
                slopes.append(float((min_points[-1][1] - point[1]))
                              / (min_points[-1][0] - point[0]))
            min_index = slopes.index(min(slopes))
            min_points = min_points + [right[min_index]]
            right = right[min_index+1:]

        x_min = self.min_conc - 0.1
        x_max = self.max_conc + 0.1
        y_min = min(conc_rel_energy, key=itemgetter(1))[1] - 0.1
        y_max = max(conc_rel_energy, key=itemgetter(1))[1] + 0.1

        if ocv:
            if type(ref) is not str:
                raise TypeError("ref must be a string type")
            if type(ref_energy) is not float:
                raise TypeError("ref_energy must be a float type")
            # Get the potential profile
            conc_range = self.max_conc - self.min_conc
            e_ref = ref_energy
            equib_pot = self.Emin_max_conc - self.Emin_min_conc
            equib_pot -= conc_range * e_ref
            equib_pot /= -conc_range
            conc_pot_pairs = []
            for x in range(1, len(min_points)):
                pot = min_points[x][1] - min_points[x-1][1]
                pot /= min_points[x-1][0] - min_points[x][0]
                pot += equib_pot
                conc_pot_pairs.append((min_points[x-1][0], pot))
                conc_pot_pairs.append((min_points[x][0], pot))

        # ----
        # Plot
        # ----
        if not ocv:
            f, ax = plt.subplots()
            # sc = ax.scatter(*convex_hull_data, color='k')
            sc = ax.plot(*convex_hull_data, color='k', ls="", marker="o")[0]
            ax.plot((x_min, x_max), (0, 0), 'r--')
            ax.plot(*zip(*min_points), color='k')
            ax.set_title('Convex-Hull')
            ax.axis([x_min, x_max, y_min, y_max])
            ax.set_ylabel(r'$E_{rel}$ (eV/f.u.)')
            ax.set_xlabel(r'concentration')
            InteractivePlot(f, ax, [sc], [names])

        else:
            f, ax = plt.subplots(2, sharex=True)
            ax[0].set_title('Convex-Hull & Open-circuit Voltage')
            sc = ax[0].plot(*convex_hull_data, color='k', ls="", marker="o")[0]
            ax[0].plot((x_min, x_max), (0, 0), 'r--')
            ax[0].plot(*zip(*min_points), color='k')
            ax[0].axis([x_min, x_max, y_min, y_max])
            ax[0].set_ylabel(r'$E_{rel}$ (eV/f.u.)')
            ax[1].plot((x_min, x_max), (equib_pot, equib_pot), 'k--')
            ax[1].plot(*zip(*conc_pot_pairs), color='k')
            ax[1].set_ylabel(r'OCV w.r.t. {} (V)'.format(ref))
            ax[1].set_xlabel(r'concentration')
            f.subplots_adjust(hspace=0)
            InteractivePlot(f, ax[0], [sc], [names])
