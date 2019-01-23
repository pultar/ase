from __future__ import division
import numpy as np
import ase.db
import os
from operator import itemgetter
from copy import deepcopy
from scipy.spatial import ConvexHull as SciConvexHull
from ase.db import connect
from itertools import product


class ConvexHull(object):
    """
    Evaluates data from database and construct the convex hull

    Arguments
    ==========
    db_name: str
        Name of the database
    select_cond: list
        Select conditions for retrieving data from the database
        If None, the select condition will be [[('converged', '=', True)]
    atoms_per_fu: int
        Number of atoms per formula unit
    conc_scale: float
        Concentration scale. In the plots the concentration
        will be divided by this number
    conc_ranges: dict
        Dictionary with lower and upper bound for the concentrations to
        be included.
        Example:
        If one want to limit the evaluation to configurations having
        gold concentrations in the range 0 to 0.5 this argument would
        be {"Au": (0, 0.5)}.
    """
    def __init__(self, db_name, select_cond=None, atoms_per_fu=1, conc_scale=1.0,
                 conc_ranges={}):
        self.db_name = db_name
        self.atoms_per_fu = atoms_per_fu
        self.conc_scale = conc_scale
        self.conc_ranges = conc_ranges

        if select_cond is None:
            self.select_cond = [('converged', '=', True)]
        else:
            self.select_cond = select_cond

        self._unique_elem = sorted(list(self.unique_elements()))
        self.end_points = self._get_end_points()
        self.weights = self._weighting_coefficients(self.end_points)
        self.num_varying_concs = 1

        self.energies, self.concs, self.db_ids = self._get_conc_energies()

    def unique_elements(self):
        """Return the number of unique elements."""
        elems = set()
        db = connect(self.db_name)
        for row in db.select(self.select_cond):
            if not self._include_row(row):
                continue
            count = row.count_atoms()
            elems = elems.union(set(count.keys()))
        return elems

    def _include_row(self, row):
        """Return True if data from the row should be included."""

        atoms_count = row.count_atoms()
        for k in atoms_count.keys():
            atoms_count[k] /= row.natoms
        
        for k, v in self.conc_ranges.items():
            if atoms_count[k] < v[0] or atoms_count[k] > v[1]:
                return False
        return True

    def _get_end_points(self):
        """Return the end points based on information in the database

        The algorithm seeks one configuration that maximize
        the composition of each element in the database.
        """
        end_points = {k: {} for k in self._unique_elem}
        for k, v in end_points.items():
            for k2 in self._unique_elem:
                v["{}_conc".format(k2)] = 0.0
            v["energy"] = 0.0

        db = connect(self.db_name)
        for row in db.select(self.select_cond):
            if not self._include_row(row):
                continue
            
            count = row.count_atoms()
            for k in self._unique_elem:
                if k not in count.keys():
                    count[k] = 0.0
                else:
                    count[k] /= row.natoms
            for k, v in end_points.items():
                if k not in count.keys():
                    continue
                
                # Check if the current structure
                # is an endpoint
                if count[k] > v["{}_conc".format(k)]:
                    v["energy"] = row.energy/row.natoms
                    for k_count in count.keys():
                        v["{}_conc".format(k_count)] = count[k_count]
        return end_points

    def _weighting_coefficients(self, end_points):
        """Return a dictionary with coefficient on reference 
           energy should be weighted.

        The weights are constructed as follows:
        1. The formation energies of each end point should 
            be zero
        2. To obtain the formation energy of an arbitrary
            structure, one should subtract the inner product
            between the concentration and the weights
           
        Arguments
        =========
        end_points: dict
            Dictionary with end point information (internally calculated)   
        """
        matrix = np.zeros((len(end_points), len(self._unique_elem)))
        rhs = np.zeros(len(end_points))
        row = 0
        for k, v in end_points.items():
            for j, symb in enumerate(self._unique_elem):
                matrix[row, j] = v["{}_conc".format(symb)]
            rhs[row] = v["energy"]
            row += 1
        
        try:
            inv_mat = np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            inv_mat = np.linalg.pinv(matrix)
        
        coeff = inv_mat.dot(rhs)
        weights = {s: coeff[i] for i, s in enumerate(self._unique_elem)}
        return weights

    def _get_conc_energies(self):
        """Read concentrations, energy and ID from the database."""
        energies = []
        ids = []
        conc = {k: [] for k in self._unique_elem}
        db = connect(self.db_name)

        for row in db.select(self.select_cond):
            if not self._include_row(row):
                continue
            count = row.count_atoms()

            for k in conc.keys():
                if k not in count.keys():
                    conc[k].append(0.0)
                else:
                    conc[k].append(count[k]/row.natoms)

            final_struct_id = row.get("final_struct_id", -1)
            if final_struct_id >= 0:
                # New format where energy is stored in a separate DB entry
                form_energy = db.get(id=final_struct_id).energy/row.natoms
            else:
                # Old format where the energy is stored in the init structure
                form_energy = row.energy/row.natoms


            # Subtract the appropriate weights
            form_energy -= sum(conc[k][-1]*self.weights[k] for k in conc.keys())
            energies.append(form_energy)
            ids.append(row.id)

        return np.array(energies), conc, ids

    def get_convex_hull(self, conc_var=None):
        """Return the convex hull.
        
        Arguments
        ==========
        conc_var: str
            Concentration variable used when calculating the 
            convex hull.
        """

        if conc_var is None:
            num_comp = len(self._unique_elem) - 1 
            x = np.zeros((len(self.energies), num_comp))
            elems = list(self._unique_elem)
            for i in range(num_comp):
                x[:, i] = self.concs[elems[i]]
            nX = num_comp
        elif conc_var in self._unique_elem:
            x = np.array(self.concs[conc_var])
            nX = 1
        else:
            raise ValueError("conc_var has to be {} or None"
                             "".format(self._unique_elem))

        points = np.vstack((x.T, self.energies.T)).T
        conv_hull = SciConvexHull(points)
        return conv_hull

    def _is_lower_conv_hull(self, simplex):
        """Return True if the simplex contain points that are
            non-positive.

        Arguments
        =========
        simplex: list
            List with indices of the points that lies
            on the convex hull    
        """
        return all(self.energies[i] <= 0.0 for i in simplex)

    def plot(self):
        """Plot formation energies."""
        from matplotlib import pyplot as plt

        num_plots = len(self._unique_elem) - 1
        
        fig = plt.figure()
        elems = list(self._unique_elem)
        for i in range(num_plots):
            ax = fig.add_subplot(1, num_plots, i+1)

            x = np.array(self.concs[elems[i]])
            x /= self.conc_scale
            
            ax.plot(x, self.energies*self.atoms_per_fu, "o", mfc="none")
            c_hull = self.get_convex_hull(conc_var=elems[i])

            if self.atoms_per_fu > 1:
                unit = "eV/f.u."
            else:
                unit = "eV/atom"
            if i == 0:
                ax.set_ylabel("Formation energy ({})".format(unit))
            else:
                ax.set_yticklabels([])

            for simpl in c_hull.simplices:
                if self._is_lower_conv_hull(simpl):
                    x_cnv = [x[simpl[0]], x[simpl[1]]]
                    y_cnv = [self.energies[simpl[0]], self.energies[simpl[1]]]
                    ax.plot(x_cnv, y_cnv, color="black", marker="x")
            ax.set_xlabel("{} conc".format(elems[i]))
        return fig

    def show_structures_on_convex_hull(self):
        """Show all entries on the convex hull."""
        from ase.gui.gui import GUI
        from ase.gui.images import Images

        c_hull = self.get_convex_hull()
        indices = set()
        for simplex in c_hull.simplices:
            if self._is_lower_conv_hull(simplex):
                indices = indices.union(simplex)
        
        cnv_hull_atoms = []
        db = connect(self.db_name)
        for i in indices:
            db_id = self.db_ids[i]
            atoms = db.get(id=db_id).toatoms()
            cnv_hull_atoms.append(atoms)

        images = Images()
        images.initialize(cnv_hull_atoms)
        gui = GUI(images, expr='')
        gui.run()

    def cosine_similarity_convex_hull(self, conc, tot_en, cnv_hull=None):
        """Calculate the Cosine similarity with structures on the convex hull.
        
        Arguments
        ==========
        conc: dict
            Dictionary with concentrations. If the system consists of
            0.3 Au, 0.5 Cu and 0.2 Zn, this would be 
            {'Au': 0.3, 'Cu': 0.5, 'Zn': 0.2}
        tot_en: float
            Total energy per atom
        cnv_hull: scipy.spatial.ConvexHull
            Convex hull object holding the data.
        """

        if cnv_hull is None:
            cnv_hull = self.get_convex_hull()
        
        indices = set()
        for simplex in cnv_hull.simplices:
            if self._is_lower_conv_hull(simplex):
                indices = indices.union(simplex)
        
        data = cnv_hull.points[list(indices), :]
        
        form_energy = tot_en - sum(self.weights[k]*conc[k] for k in conc.keys())

        min_en = np.min(data[:, -1])
        max_en = np.max(data[:, -1])
        diff = max_en - min_en

        # Normalize the energy data
        data[:, -1] /= diff

        mean = np.mean(data, axis=0)
        data -= mean

        data_vec = np.zeros(data.shape[1])
        data_vec[:-1] = [conc.get(k, 0.0) for k in self._unique_elem[:-1]]
        data_vec[-1] = form_energy/diff
        data_vec -= mean

        inner_prod = data.dot(data_vec)
        inner_prod /= np.sqrt(data_vec.dot(data_vec))
        inner_prod /= np.sqrt(np.diag(data.dot(data.T)))
        return np.max(inner_prod)


        

    # def plot(self, ocv=False, ref=None, ref_energy=None):
    #     import matplotlib.pyplot as plt
    #     from ase.clease.interactive_plot import InteractivePlot

    #     conc_rel_energy, names = self._get_concentration_rel_energy_pairs()
    #     convex_hull_data = list(zip(*conc_rel_energy))
    #     conc_values = list(set(convex_hull_data[0]))
    #     # ----------------------------------------------------
    #     # find the minimum energy for each concentration value
    #     # ----------------------------------------------------
    #     conc_Emin_pairs = []
    #     for conc in conc_values:
    #         if conc == self.min_conc or conc == self.max_conc:
    #             conc_Emin_pairs.append((conc, 0.))
    #         else:
    #             # find minimum energy
    #             min_energy = 1000
    #             for pair in conc_rel_energy:
    #                 if pair[0] == conc and pair[1] < min_energy:
    #                     min_energy = pair[1]
    #             conc_Emin_pairs.append((conc, min_energy))
    #     # sort it based on concentration values
    #     conc_Emin_pairs = sorted(conc_Emin_pairs, key=itemgetter(0))
    #     # remove the points with Erel higher than zero
    #     conc_Emin_pairs = [x for x in conc_Emin_pairs if x[1] <= 0.0]

    #     # -------------------------------------
    #     # Get minimum energy point to be traced
    #     # -------------------------------------
    #     min_points = [min(conc_Emin_pairs, key=itemgetter(1))]

    #     # scan left
    #     left = [p for p in conc_Emin_pairs if p[0] < min_points[0][0]]
    #     while min_points[0] != conc_Emin_pairs[0]:
    #         slopes = []
    #         for point in left:
    #             slopes.append(float((min_points[0][1] - point[1])) /
    #                           (min_points[0][0] - point[0]))
    #         min_index = slopes.index(max(slopes))
    #         min_points = [left[min_index]] + min_points
    #         left = left[:min_index]

    #     # scan right
    #     right = [p for p in conc_Emin_pairs if p[0] > min_points[-1][0]]
    #     while min_points[-1] != conc_Emin_pairs[-1]:
    #         slopes = []
    #         for point in right:
    #             slopes.append(float((min_points[-1][1] - point[1])) /
    #                           (min_points[-1][0] - point[0]))
    #         min_index = slopes.index(min(slopes))
    #         min_points = min_points + [right[min_index]]
    #         right = right[min_index+1:]

    #     x_min = self.min_conc - 0.1
    #     x_max = self.max_conc + 0.1
    #     y_min = min(conc_rel_energy, key=itemgetter(1))[1] - 0.1
    #     y_max = max(conc_rel_energy, key=itemgetter(1))[1] + 0.1

    #     if ocv:
    #         if type(ref) is not str:
    #             raise TypeError("ref must be a string type")
    #         if type(ref_energy) is not float:
    #             raise TypeError("ref_energy must be a float type")
    #         # Get the potential profile
    #         conc_range = self.max_conc - self.min_conc
    #         e_ref = ref_energy
    #         equib_pot = self.Emin_max_conc - self.Emin_min_conc
    #         equib_pot -= conc_range * e_ref
    #         equib_pot /= -conc_range
    #         conc_pot_pairs = []
    #         for x in range(1, len(min_points)):
    #             pot = min_points[x][1] - min_points[x-1][1]
    #             pot /= min_points[x-1][0] - min_points[x][0]
    #             pot += equib_pot
    #             conc_pot_pairs.append((min_points[x-1][0], pot))
    #             conc_pot_pairs.append((min_points[x][0], pot))

    #     # ----
    #     # Plot
    #     # ----
    #     if not ocv:
    #         f, ax = plt.subplots()
    #         # sc = ax.scatter(*convex_hull_data, color='k')
    #         sc = ax.plot(*convex_hull_data, color='k', ls="", marker="o")[0]
    #         ax.plot((x_min, x_max), (0, 0), 'r--')
    #         ax.plot(*zip(*min_points), color='k')
    #         ax.set_title('Convex-Hull')
    #         ax.axis([x_min, x_max, y_min, y_max])
    #         ax.set_ylabel(r'$E_{rel}$ (eV/f.u.)')
    #         ax.set_xlabel(r'concentration')
    #         InteractivePlot(f, ax, [sc], [names])

    #     else:
    #         f, ax = plt.subplots(2, sharex=True)
    #         ax[0].set_title('Convex-Hull & Open-circuit Voltage')
    #         sc = ax[0].plot(*convex_hull_data, color='k', ls="", marker="o")[0]
    #         ax[0].plot((x_min, x_max), (0, 0), 'r--')
    #         ax[0].plot(*zip(*min_points), color='k')
    #         ax[0].axis([x_min, x_max, y_min, y_max])
    #         ax[0].set_ylabel(r'$E_{rel}$ (eV/f.u.)')
    #         ax[1].plot((x_min, x_max), (equib_pot, equib_pot), 'k--')
    #         ax[1].plot(*zip(*conc_pot_pairs), color='k')
    #         ax[1].set_ylabel(r'OCV w.r.t. {} (V)'.format(ref))
    #         ax[1].set_xlabel(r'concentration')
    #         f.subplots_adjust(hspace=0)
    #         InteractivePlot(f, ax[0], [sc], [names])

        # def _get_concentration_rel_energy_pairs(self, conditions=None):
    #     tuples = []
    #     if conditions is None:
    #         rows = self.db.select(self.select_cond)
    #     else:
    #         select_cond = deepcopy(self.select_cond)
    #         if select_cond is None:
    #             select_cond = []
    #         for x in range(len(conditions)):
    #             select_cond.append(conditions[x])
    #         rows = self.db.select(select_cond)

    #     names = []
    #     for row in rows:
    #         c = row.get(self.conc_var_name) * self.conc_scale
    #         conc_range = self.max_conc - self.min_conc
    #         dE = (row.energy * self.atoms_per_fu / row.natoms) - \
    #              (((c - self.min_conc) / conc_range) * self.Emin_max_conc) - \
    #              (((self.max_conc - c) / conc_range) * self.Emin_min_conc)
    #         tuples.append((c, dE))
    #         names.append(row.name)
    #     return tuples, names
