"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2017 Charles Thomas Johnson, JaeHwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function
from ase.calculators.openmx.import_functions import read_nth_to_last_value
from numpy import ndarray
import numpy as np
try:
    from matplotlib import pyplot as plt
except RuntimeError:
    print('Failed to load matplot display, do not try to visualise plots!')


class Band:
    def __init__(self, calc):
        self.calc = calc
        self.magk = []
        self.xpoints = []
        self.xlabels = []
        init_euler = 'initial_magnetic_moments_euler_angles'
        for kpath_dict in self.calc['band_kpath']:
            self.xlabels.append(kpath_dict['path_symbols'][0])
        self.xlabels.append(self.calc['band_kpath'][-1]['path_symbols'][1])
        if np.any(self.calc['initial_magnetic_moments'] is not None) or \
                np.all(self.calc[init_euler] is None):
            if(calc.debug):
                print("Reading Up bands and Down bands")
            self.spin_up_bands = []
            self.spin_down_bands = []
        else:
            if(calc.debug):
                print("Reading bands")
            self.bands = []

    @staticmethod
    def plot_brillouin_2d(atoms, ax=None, size=0.8, kpath=None, **kwargs):
        from scipy.spatial import Voronoi
        import matplotlib.pyplot as plt

        def get_reciprocal(a):
            cr = np.array([np.cross(a[1], a[2]), np.cross(a[2], a[0]),
                           np.cross(a[0], a[1])])
            crd = np.array([np.dot(a[0], np.cross(a[1], a[2])),
                            np.dot(a[1], np.cross(a[2], a[0])),
                            np.dot(a[2], np.cross(a[0], a[1]))])
            return 2*np.pi*cr/crd
        b = get_reciprocal(atoms.cell)
        b1 = b[:2, :2][0]
        b2 = b[:2, :2][1]
        points = []
        for M1 in range(-1, 2):
            for M2 in range(-1, 2):
                points.append(M1*b1+M2*b2)
        vor = Voronoi(points)

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot()
            plt.xlim((-1, 1))
            plt.ylim((-1, 1))

        if kwargs.get('show_points', True):
            plt.plot(vor.points[:, 0], vor.points[:, 1],  '.')
            plt.arrow(0, 0, b1[0], b1[1], head_width=0.02,
                      head_length=0, fc='k', ec='b')
            plt.arrow(0, 0, b2[0], b2[1], head_width=0.02,
                      head_length=0, fc='k', ec='b')

        if kpath is not None:
            plt.plot(kpath[:, 0], kpath[:, 1], 't-')

        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                line = np.array([l for l in vor.vertices[simplex]
                                if np.linalg.norm(l) < size])
                line = np.append(line, [line[0]], axis=0)
                plt.plot(line[:, 0], line[:, 1], 'k-')
        plt.show()

    @staticmethod
    def plot_brillouin_3d(atoms, ax=None, size=0.9, kpath=None, **kw):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        Axes3D
        if ax is None:
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')
            plt.ylim((-1, 1))
            ax.pbaspect = [1.0, 1.0, 1]
        from scipy.spatial import Voronoi

        def get_reciprocal(a):
            cr = np.array([np.cross(a[1], a[2]), np.cross(a[2], a[0]),
                           np.cross(a[0], a[1])])
            crd = np.array([np.dot(a[0], np.cross(a[1], a[2])),
                            np.dot(a[1], np.cross(a[2], a[0])),
                            np.dot(a[2], np.cross(a[0], a[1]))])
            return 2*np.pi*cr/crd
        b = get_reciprocal(atoms.cell)
        b1, b2, b3 = tuple(b)
        points = []
        for M1 in range(-1, 2):
            for M2 in range(-1, 2):
                for M3 in range(-1, 2):
                    points.append(M1*b1+M2*b2+M3*b3)
        vor = Voronoi(points)
        if vor.points.shape[1] != 3:
            raise ValueError("Voronoi diagram is not 3-D")

        if kw.get('show_points', True):
            ax.plot(vor.points[:, 0], vor.points[:, 1], vor.points[:, 2], '.')
            ax.plot(kpath[:, 0], kpath[:, 1], kpath[:, 2], 'r-')

        for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
            simplex = np.asarray(simplex)
            if np.all(simplex >= 0):
                line = np.array([l for l in vor.vertices[simplex]
                                if np.linalg.norm(l) < size])
                line = np.append(line, [line[0]], axis=0)
                ax.plot(line[:, 0], line[:, 1], line[:, 2], 'k-')

    def read_band(self):
        kpath = self.calc['band_kpath']
        number_of_paths_per_band = len(kpath)
        num_path = number_of_paths_per_band
        rn = read_nth_to_last_value
        kpts_for_each_path = [kpath[i]['kpts'] for i in range(num_path)]
        first_line = ''
        with open(self.calc.label + '.Band', 'r') as f:
            first_line = f.readline()
            i = -1
            char = ' '
            while char == ' ':
                i += 1
                char = first_line[i]
            j = 1
            while char != ' ':
                j += 1
                char = first_line[j]
        number_of_bands = int(first_line[i:j])
        kpts_per_band = sum(kpts_for_each_path)
        bands = ndarray((number_of_bands, kpts_per_band), float)
        magk = ndarray(kpts_per_band, float)

        if np.any(self.calc['initial_magnetic_moments'] is None) and \
           np.all(self.calc['initial_magnetic_moments_euler_angles'] is None):
            with open(self.calc.label + '.BANDDAT1', 'r') as f:
                line = 'start'
                while line != '':
                    line = f.readline()
                    if line != '\n' and line != '':
                        if float(read_nth_to_last_value(line, 2)) == 0:
                            for i in range(number_of_bands):
                                for j in range(kpts_per_band):
                                    while line == '\n':
                                        line = f.readline()
                                    if i == 0:
                                        magk[j] = rn(line, 2)
                                    bands[i][j] = read_nth_to_last_value(line)
                                    line = f.readline()

            spin_up_bands = bands
            spin_down_bands = ndarray((number_of_bands, kpts_per_band), float)
            with open(self.calc.label + '.BANDDAT2', 'r') as f:
                line = 'start'
                while line != '':
                    line = f.readline()
                    if line != '\n' and line != '':
                        if float(read_nth_to_last_value(line, 2)) == 0:
                            for i in range(number_of_bands):
                                for j in range(kpts_per_band):
                                    while line == '\n':
                                        line = f.readline()
                                    spin_down_bands[i][j] = rn(line)
                                    line = f.readline()
            self.spin_up_bands = spin_up_bands
            self.spin_down_bands = spin_down_bands
        else:
            self.bands = bands
        self.magk = magk
        index = 0
        for kpts in kpts_for_each_path:
            self.xpoints.append(magk[index])
            index += kpts
        self.xpoints.append(magk[-1])

    def plot_bands(self, erange=(-5, 5), fermi_level=True, file_format=False,
                   fileName=None):
        if fileName is None:
            fileName = self.calc.prefix
        if np.any(self.calc['initial_magnetic_moments'] is None) and \
           np.all(self.calc['initial_magnetic_moments_euler_angles'] is None):
            figure, axes = plt.subplots(1, 2, sharey=True,
                                        sharex=False,
                                        squeeze=False)
            for band in self.spin_up_bands:
                axes[0][0].plot(self.magk, band, 'r')
            axes[0][0].set_ylim(ymin=erange[0], ymax=erange[1])
            if fermi_level:
                axes[0][0].axhspan(erange[0], 0., color='y', alpha=0.5)
            for band in self.spin_down_bands:
                axes[0][1].plot(self.magk, band, 'c')
            axes[0][1].set_ylim(ymin=erange[0], ymax=erange[1])
            if fermi_level:
                axes[0][1].axhspan(erange[0], 0., color='y', alpha=0.5)
            axes[0][1].set_xticks(self.xpoints)
            axes[0][1].set_xticklabels(self.xlabels)
            axes[0][1].set_xlim(xmin=min(self.magk), xmax=max(self.magk))
            for xpoint in self.xpoints:
                axes[0][1].axvspan(xpoint, xpoint, color='k')
        else:
            figure, axes = plt.subplots(1, 1, squeeze=False)
            for band in self.bands:
                axes[0][0].plot(self.magk, band, 'b')
            axes[0][0].set_ylim(ymin=erange[0], ymax=erange[1])
            if fermi_level:
                axes[0][0].axhspan(erange[0], 0., color='y', alpha=0.5)
        axes[0][0].set_ylabel('Energy Above Fermi Level (eV)')
        axes[0][0].set_xticks(self.xpoints)
        axes[0][0].set_xticklabels(self.xlabels)
        axes[0][0].set_xlim(xmin=min(self.magk), xmax=max(self.magk))
        for xpoint in self.xpoints:
            axes[0][0].axvspan(xpoint, xpoint, color='k')
        figure.suptitle(self.calc.prefix + ': Band Dispersion')
        if file_format:
            plt.show()
            plt.savefig(filename=self.calc.prefix + '.Band.' + file_format)
        if not file_format:
            plt.show()
        return figure, axes

    def get_band(self, erange=(-5, 5), fermi_level=True, file_format=False,
                 fileName=None):
        self.read_band()
        return self.plot_bands(erange=erange, fermi_level=fermi_level,
                               file_format=file_format, fileName=fileName)
