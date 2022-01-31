from typing import Tuple
import numpy as np

from ase.units import Bohr, Ha
from ase.data import covalent_radii
from ase.neighborlist import NeighborList
from .polarizability import StaticPolarizabilityCalculator


class LippincottStuttman:
    # atomic polarizability values from:
    #   Lippincott and Stutman J. Phys. Chem. 68 (1964) 2926-2940
    #   DOI: 10.1021/j100792a033
    # see also:
    #   Marinov and Zotov Phys. Rev. B 55 (1997) 2938-2944
    #   DOI: 10.1103/PhysRevB.55.2938
    # unit: Angstrom^3
    atomic_polarizability = {
        "H": 0.592,
        "Li": 6.993,
        "Be": 3.802,
        "B": 1.358,
        "C": 0.978,
        "Na": 12.884,
        "Mg": 8.370,
        "A1": 3.918,
        "Si": 2.988,
        "K": 21.593,
        "Ca": 15.4756,
        "Ga": 5.635,
        "Ge": 3.848,
        "Rb": 25.239,
        "Sr": 18.242,
        "In": 7.867,
        "Sn": 5.256,
        "Cs": 35.717,
        "Ba": 24.584,
        "Tl": 17.118,
        "Pb": 13.609,
        "N": 0.743,
        "O": 0.592,
        "F": 0.490,
        "P": 2.367,
        "S": 1.820,
        "Cl": 1.388,
        "As": 3.302,
        "Se": 2.524,
        "Br": 1.941,
        "Kr": 5.256,
        "Sb": 4.864,
        "Te": 3.802,
        "I": 2.972,
        "Bi": 12.175,
        "Po": 9.334,
        "At": 7.928,
    }

    # reduced electronegativity Table I
    reduced_eletronegativity = {
        "H": 1.0,
        "Li": 0.439,
        "Be": 0.538,
        "B": 0.758,
        "C": 0.846,
        "N": 0.927,
        "O": 1.0,
        "F": 1.056,
        "Na": 0.358,
        "Mg": 0.414,
        "Al": 0.533,
        "Si": 0.583,
        "P": 0.63,
        "S": 0.688,
        "Cl": 0.753,
        "K": 0.302,
        "Ca": 0.337,
        "Ga": 0.472,
        "Ge": 0.536,
        "As": 0.564,
        "Se": 0.617,
        "Br": 0.633,
        "Rb": 0.286,
        "Sr": 0.319,
        "In": 0.422,
        "Sn": 0.483,
        "Sb": 0.496,
        "Te": 0.538,
        "I": 0.584,
        "Cs": 0.255,
        "Ba": 0.289,
        "Tl": 0.326,
        "Pb": 0.352,
        "Bi": 0.365,
        "Po": 0.399,
        "At": 0.421,
    }

    def __call__(self, el1: str, el2: str, length: float) -> Tuple[float, float]:
        """Bond polarizability

        Parameters
        ----------
        el1: element string
        el2: element string
        length: float

        Returns
        -------
        alphal: float
          Parallel component
        alphap: float
          Perpendicular component
        """
        alpha1 = self.atomic_polarizability[el1]
        alpha2 = self.atomic_polarizability[el2]
        ren1 = self.reduced_eletronegativity[el1]
        ren2 = self.reduced_eletronegativity[el2]

        sigma = 1.0
        if el1 != el2:
            sigma = np.exp(-((ren1 - ren2) ** 2) / 4)

        # parallel component
        alphal = sigma * length ** 4 / (4 ** 4 * alpha1 * alpha2) ** (1.0 / 6)
        # XXX consider fractional covalency ?

        # prependicular component
        alphap = (ren1 ** 2 * alpha1 + ren2 ** 2 * alpha2) / (ren1 ** 2 + ren2 ** 2)
        # XXX consider fractional covalency ?

        return alphal, alphap


class Linearized:
    def __init__(self):
        self._data = {
            # L. Wirtz, M. Lazzeri, F. Mauri, A. Rubio,
            # Phys. Rev. B 2005, 71, 241402.
            #      R0     al    al'   ap    ap'
            "CC": (1.53, 1.69, 7.43, 0.71, 0.37),
            "BN": (1.56, 1.58, 4.22, 0.42, 0.90),
        }

    def __call__(self, el1: str, el2: str, length: float) -> Tuple[float, float]:
        """Bond polarizability

        Parameters
        ----------
        el1: element string
        el2: element string
        length: float

        Returns
        -------
        alphal: float
          Parallel component
        alphap: float
          Perpendicular component
        """
        if el1 > el2:
            bond = el2 + el1
        else:
            bond = el1 + el2
        assert bond in self._data
        length0, al, ald, ap, apd = self._data[bond]

        return al + ald * (length - length0), ap + apd * (length - length0)


class BondPolarizability(StaticPolarizabilityCalculator):
    def __init__(self, model=LippincottStuttman()):
        self.model = model

    def __call__(self, atoms, radiicut=1.5):
        """Sum up the bond polarizability from all bonds

        Parameters
        ----------
        atoms: Atoms object
        radiicut: float
          Bonds are counted up to
          radiicut * (sum of covalent radii of the pairs)
          Default: 1.5

        Returns
        -------
        polarizability tensor with unit (e^2 Angstrom^2 / eV).
        Multiply with Bohr * Ha to get (Angstrom^3)
        """
        radii = np.array([covalent_radii[z] for z in atoms.numbers])
        nl = NeighborList(radii * 1.5, skin=0, self_interaction=False)
        nl.update(atoms)
        pos_ac = atoms.get_positions()

        alpha = 0
        for ia, atom in enumerate(atoms):
            indices, offsets = nl.get_neighbors(ia)
            pos_ac = atoms.get_positions() - atoms.get_positions()[ia]

            for ib, offset in zip(indices, offsets):
                weight = 1
                if offset.any():  # this comes from a periodic image
                    weight = 0.5  # count half the bond only

                dist_c = pos_ac[ib] + np.dot(offset, atoms.get_cell())
                dist = np.linalg.norm(dist_c)
                al, ap = self.model(atom.symbol, atoms[ib].symbol, dist)

                eye3 = np.eye(3) / 3
                alpha += weight * (al + 2 * ap) * eye3
                alpha += (
                    weight * (al - ap) * (np.outer(dist_c, dist_c) / dist ** 2 - eye3)
                )
        return alpha / Bohr / Ha
