# flake8: noqa
"""Definition of the XrayDebye class.

This module defines the XrayDebye class for calculation
of X-ray scattering properties from atomic cluster
using Debye formula.
"""


from itertools import combinations
import warnings

import numpy as np

from xrayfunctions import initialize_atomic_form_factor_splines, pdist_in_chunks, cdist_in_chunks


def one_species_contribution(positions, form_factor_spline, s) -> np.ndarray:
    
    contribution = np.zeros_like(s)
    n_atoms = positions.shape[0]

    for distances in pdist_in_chunks(positions):
        for i in range(s.shape[0]):
            contribution[i] += form_factor_spline(s[i]) ** 2 * (
                n_atoms + 2 * np.sum(np.sinc(distances * s[i] * 2))
            )

    return contribution


def one_species_hist_contribution(positions, form_factor_spline, s, bin_width=1e-3) -> np.ndarray:
    n_atoms: int = positions.shape[0]
    contribution: np.ndarray = np.zeros_like(s)

    if positions.shape[0] == 1:
        return form_factor_spline(s) ** 2 * n_atoms


    for distances in pdist_in_chunks(positions):
        nbins = int(np.ceil(np.ptp(distances) / bin_width)) + 1

        dist_hist, bin_edges = np.histogram(distances, bins=nbins)
        nontrivial_bins = np.nonzero(dist_hist)

        dist_hist, bin_edges = (
            dist_hist[nontrivial_bins],
            bin_edges[nontrivial_bins] + (bin_edges[1] - bin_edges[0]) / 2,
        )

        for i in range(s.shape[0]):
            contribution[i] += form_factor_spline(s[i]) ** 2 * (
                n_atoms + 2 * np.sum(dist_hist * np.sinc(bin_edges * s[i] * 2))
            )

    return contribution


def two_species_contribution(
    positions1, positions2, form_factor_spline1, form_factor_spline2, s
) -> np.ndarray:
    contribution = np.zeros_like(s)

    for distances in cdist_in_chunks(positions1, positions2):
        for i in range(s.shape[0]):
            contribution[i] += (
                2.0
                * form_factor_spline1(s[i])
                * form_factor_spline2(s[i])
                * np.sum(np.sinc(distances * s[i] * 2))
            )

    return contribution


def two_species_hist_contribution(
    positions1, positions2, form_factor_spline1, form_factor_spline2, s, bin_width=1e-2
) -> np.ndarray:
    contribution = np.zeros_like(s)

    for distances in cdist_in_chunks(positions1, positions2):
        nbins = int(np.ceil(np.ptp(distances) / bin_width)) + 1

        dist_hist, bin_edges = np.histogram(distances, bins=nbins)
        nontrivial_bins = np.nonzero(dist_hist)
        dist_hist, bin_edges = (
            dist_hist[nontrivial_bins],
            bin_edges[nontrivial_bins] + (bin_edges[1] - bin_edges[0]) / 2,
        )
        for i in range(s.shape[0]):
            contribution[i] += (
                2.0
                * form_factor_spline1(s[i])
                * form_factor_spline2(s[i])
                * np.sum(dist_hist * np.sinc(bin_edges * s[i] * 2))
            )

    return contribution


# End support


class XRDData:
    mode = "XRD"
    xvalues_name = "2theta"

    def __init__(self, twotheta, intensities):
        self.twotheta = twotheta
        self.intensities = intensities

    @classmethod
    def calculate(cls, xrd, twotheta):
        if twotheta is None:
            twotheta = np.linspace(15, 55, 100)
        else:
            twotheta = np.asarray(twotheta)

        svalues = 2 * np.sin(twotheta * np.pi / 180 / 2.0) / xrd.wavelength
        result = xrd.get(svalues)
        return cls(twotheta, np.array(result))

    def xvalues(self):
        return self.twotheta.copy()


class SAXSData:
    mode = "SAXS"
    xvalues_name = "q(1/Å)"

    def __init__(self, qvalues, intensities):
        self.qvalues = qvalues
        self.intensities = intensities

    @classmethod
    def calculate(cls, xrd, qvalues):
        if qvalues is None:
            qvalues = np.logspace(-3, -0.3, 100)
        else:
            qvalues = np.asarray(qvalues)

        svalues = qvalues / (2 * np.pi)
        result = xrd.get(svalues)
        return cls(qvalues, result)

    def xvalues(self):
        return self.qvalues.copy()


output_data_class = {"XRD": XRDData, "SAXS": SAXSData}


class XrayDebye:
    """
    Class for calculation of XRD or SAXS patterns.
    """

    def __init__(
        self,
        atoms: "ase.Atoms",
        wavelength: float,
        damping: float=0.04,
        method: str="Iwasa",
        alpha: float=1.01,
        histogram_approximation: bool=True
    ):
        """[summary]

        Parameters
        ----------
        atoms : ase.Atoms
            atoms object for which calculation will be performed
        wavelength : float
            X-ray wavelength in Angstrom. Used for XRD and to setup dumpings
        damping : float, optional
            thermal damping factor parameter (B-factor), by default 0.04
        method : str, optional
            method of calculation (damping and atomic factors affected), by default "Iwasa"

            If set to 'Iwasa' than angular damping and q-dependence of
            atomic factors are used.

            For any other string there will be only thermal damping
            and constant atomic factors (`f_a(q) = Z_a`)
        alpha : float, optional
            parameter for angular damping of scattering intensity.
            Close to 1.0 for unplorized beam, by default 1.01
        histogram_approximation : bool, optional
            [description], by default True
        """
        # XXX deprecate warn

        self.wavelength = wavelength
        self.method = method
        self.alpha = alpha

        self.damping = damping
        self.atoms = atoms

        self.x2q = {
            "XRD": lambda x: 2 * np.sin(x * np.pi / 180 / 2.0) / self.wavelength,
            "SAXS": lambda x: x,
        }
        self.atomic_form_factor_dict = initialize_atomic_form_factor_splines(
            set(self.atoms.symbols), np.linspace(0, 6, 500)
        )

        self.hist_approx = histogram_approximation

        self._xrddata = None
        # TODO: setup atomic form factors if method != 'Iwasa'

    def set_damping(self, damping):
        """set B-factor for thermal damping"""
        self.damping = damping

    def get(self, s):
        r"""Get the powder x-ray (XRD) scattering intensity
        using the Debye-Formula at single point.

        Parameters:

        s: float array, in inverse Angstrom
            scattering vector value (`s = q / 2\pi`).

        Returns:
            Intensity at given scattering vector `s`.
        """

        pre = np.exp(-self.damping * s ** 2 / 2)


        if self.method == "Iwasa":
            sinth = self.wavelength * s / 2.0
            positive = 1.0 - sinth ** 2
            positive[positive < 0] = 0
            costh = np.sqrt(positive)
            cos2th = np.cos(2.0 * np.arccos(costh))
            pre *= costh / (1.0 + self.alpha * cos2th ** 2)

        I = np.zeros_like(s)

        # Calculate contribution from pairs of same atomic species
        symbols = set(self.atoms.symbols)

        for symbol in symbols:
            print(f"Calculating on-diagonal {symbol} block")
            if not self.hist_approx:
                I[:] += one_species_contribution(
                    self.atoms[self.atoms.symbols == symbol].positions,
                    self.atomic_form_factor_dict[symbol],
                    s,
                )
            else:
                I[:] += one_species_hist_contribution(
                self.atoms[self.atoms.symbols == symbol].positions, self.atomic_form_factor_dict[symbol],
                s)

        # Calculation contribution from pairs of different atomic species
        symbols_pairs = combinations(symbols, 2)

        for symbols_pair in symbols_pairs:
            symbol1, symbol2 = symbols_pair
            print(f"Calculating off-diagonal {symbol1}+{symbol2} blocks")
            if not self.hist_approx:
                I[:] += two_species_contribution(
                    self.atoms[self.atoms.symbols == symbol1].positions,
                    self.atoms[self.atoms.symbols == symbol2].positions,
                    self.atomic_form_factor_dict[symbol1],
                    self.atomic_form_factor_dict[symbol2],
                    s,
                )

            else:
                I[:] += two_species_hist_contribution(
                self.atoms[self.atoms.symbols == symbol1].positions, self.atoms[self.atoms.symbols == symbol2].positions, self.atomic_form_factor_dict[symbol1],
                self.atomic_form_factor_dict[symbol2],s)


        lin_zhigilei_factor = [len(self.atoms.symbols == x) * self.atomic_form_factor_dict[x](s) ** 2 for x in symbols]

        return pre * I # / np.sum(lin_zhigilei_factor, axis=0)

    def calc_pattern(self, x=None, mode="XRD"):
        r"""
        Calculate X-ray diffraction pattern or
        small angle X-ray scattering pattern.

        Parameters:

        x: float array
            points where intensity will be calculated.
            XRD - 2theta values, in degrees;
            SAXS - q values in 1/Å
            (`q = 2 \pi \cdot s = 4 \pi \sin( \theta) / \lambda`).
            If ``x`` is ``None`` then default values will be used.

        mode: {'XRD', 'SAXS'}
            the mode of calculation: X-ray diffraction (XRD) or
            small-angle scattering (SAXS).

        Returns:
            list of intensities calculated for values given in ``x``.
        """
        assert mode in ["XRD", "SAXS"]

        cls = output_data_class[mode]

        self._xrddata = cls.calculate(self, x)
        return self._xrddata.intensities

        # if mode == 'XRD':
        # elif mode == 'SAXS':
        #    return SAXSData.calculate(self)
        # intensity_list = np.array(result)
        # xrd = XRDebyeData(mode, twotheta_list, q_list, intensity_list)
        # self._xrddata = xrd
        # return xrd

    @property
    def mode(self):
        return self._xrddata.mode

    def write_pattern(self, filename):
        """Save calculated data to file specified by ``filename`` string."""
        with open(filename, "w") as fd:
            self._write_pattern(fd)

    def _write_pattern(self, fd):
        data = self._xrddata

        fd.write("# Wavelength = %f\n" % self.wavelength)
        fd.write(f"# {data.xvalues_name}\tIntensity\n")

        for xval, yval in zip(data.xvalues(), data.intensities):
            fd.write("  %f\t%f\n" % (xval, yval))

    def plot_pattern(self, filename=None, show=False, ax=None):
        """Plot XRD or SAXS depending on filled data

        Uses Matplotlib to plot pattern. Use *show=True* to
        show the figure and *filename='abc.png'* or
        *filename='abc.eps'* to save the figure to a file.

        Returns:
            ``matplotlib.axes.Axes`` object."""

        import matplotlib.pyplot as plt

        if ax is None:
            plt.clf()  # clear figure
            ax = plt.gca()

        if self.mode == "XRD":
            x, y = np.array(self.twotheta_list), np.array(self.intensity_list)
            ax.plot(x, y / np.max(y), ".-")
            ax.set_xlabel("2$\\theta$")
            ax.set_ylabel("Intensity")
        elif self.mode == "SAXS":
            x = self._xrddata.xvalues()
            y = np.array(self._xrddata.intensities)
            ax.loglog(x, y / np.max(y), ".-")
            ax.set_xlabel("q, 1/Å")
            ax.set_ylabel("Intensity")
        else:
            raise Exception("No data available, call calc_pattern() first")

        if show:
            plt.show()
        if filename is not None:
            fig = ax.get_figure()
            fig.savefig(filename)

        return ax
