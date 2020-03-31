"""Votca Spectrum calculator."""


import matplotlib.pyplot as plt
import numpy as np

from typing import Callable, NamedTuple
from ..atoms import Atoms


class LabeLEnergies(NamedTuple):
    """Namedtuple containing the labels and energies."""

    label: str
    function_energies: Callable[[], np.ndarray]


def show_and_save(handle, show: True, outfile: str) -> None:
    """Show and save a plot."""
    if show:
        handle.show()
    handle.savefig(outfile)


def spectrum(
        atoms: Atoms, kind: str = 'ks', show: bool = True, outfile: str = 'spectrum.png') -> None:
    """Create a 'band' plot for a give kind of energies (ks,qp,singlet and triplet).

    show:
        visualize the graph
    outputfile:
        path to store the graph.
    """
    dict_kinds = {"ks": LabeLEnergies(
        'Kohn-Sham Energies', atoms.get_ks_energies),
        "kp": LabeLEnergies(
            'Quasiparticle Energies', atoms.get_qp_energies),
        "singlet": LabeLEnergies(
            'Singlet Energies', atoms.get_singlet_energies),
        "triplet": LabeLEnergies(
            'Triple Energies', atoms.get_triplet_energies)
    }

    # Search for the label and energy corresponding to kind
    label, function_energies = dict_kinds[kind]
    energies = function_energies()

    ax = plt.gca()
    ax.hlines(energies, -0.5, 0.5)
    ax.set_xticks([])
    ax.set_xlabel(label)
    ax.set_ylabel('Energy [eV]')
    show_and_save(plt, show, outfile)


def GWspectrum(
        atoms: Atoms, show: bool = True, outfile: str = 'GWspectrum.png') -> None:
    """Create a 'band' plot for GW spectrum (KS vs. QP_pert vs. QP).

    Parameters
    ----------
    show
        visualize the graph
    outputfile
        path to store the graph.

    """
    plt.close("all")
    hrt2ev = 27.2214
    ks = hrt2ev * atoms.get_ks_energies()
    qp_pert = hrt2ev * atoms.get_qp_pert_energies()
    qp = hrt2ev * atoms.get_qp_energies()
    plt.hlines(ks[:len(qp)], 0, 1)
    plt.hlines(qp_pert, 1.5, 2.5)
    plt.hlines(qp, 3, 4)
    plt.xticks([0.5, 2, 3.5], ('KS', 'PQP', 'DQP'))
    plt.ylabel('Energy [eV]')
    show_and_save(plt, show, outfile)


def self_energy(atoms: Atoms, show: bool = True, outfile: str = 'selfenergy.png') -> None:
    """Sketch the self-energy correction  respect to the Kohn-Sham energies.

    Parameters
    ----------
    show
        visualize the graph
    outputfile
        path to store the graph.


    """
    plt.close("all")
    ks = atoms.get_ks_energies()
    qp = atoms.get_qp_energies()
    ks = ks[:len(qp)]
    plt.scatter(ks, qp - ks, marker='o', alpha=0.7, c='red')
    plt.ylabel('QP - KS (eV)')
    plt.xlabel('KS (eV)')
    show_and_save(plt, show, outfile)


def gaussian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Compute the value of a gaussian with center x and fwhm at position x."""
    # FWHM = 2*sqrt(2 ln2) sigma = 2.3548 sigma
    sigma = fwhm / 2.3548
    return(np.exp(-0.5 * ((x - center) / sigma)**2) / sigma / np.sqrt(2.0 * np.pi))


def lorentzian(x: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Compute the value of a lorentzian with center x and fwhm at position x."""
    return(fwhm * fwhm / ((x - center)**2 + fwhm * fwhm) / np.pi)


def epsilon_2(atoms, omega_in=0.1, omega_fin=30, num=300, fwhm=1e-2, axis=0, outfile='BSEepsilon.png', show=True):
    hrt2ev = 27.2214
    singlet = hrt2ev * atoms.get_singlet_energies()
    tdipoles = atoms.get_transition_dipoles()[:, axis]
    w = np.linspace(omega_in, omega_fin, num=num)
    eps2 = []
    for x in w:
        c = 0
        for a, b in zip(singlet, tdipoles):
            c += lorentzian(x, a, fwhm) * b**2
        eps2.append(c)
    plt.close("all")
    plt.plot(w, eps2)
    show_and_save(plt, show, outfile)
