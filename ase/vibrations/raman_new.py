from pathlib import Path
import numpy as np
import typing as tp

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.polarizability import StaticPolarizabilityCalculator
from ase.calculators.excitation_list import ExcitationList, ExcitationListCalculator
from ase.parallel import world, paropen, parprint
from ase.vibrations.vibrations import VibrationsRunner
from ase.vibrations.displacements import Displacement
from ase.vibrations.resonant_raman import _copy_atoms_calc

EC = tp.TypeVar('EC', bound='ExcitationListCalculator')
PC = tp.TypeVar('PC', bound='StaticPolarizabilityCalculator')

class ResonantRamanRunner:
    """Base class for resonant Raman calculators using finite differences.

    Parameters
    ----------
    vibrations: VibrationsRunner
        The VibrationsRunner object
    ExcitationsCalculator: object
        Calculator for excited states
    exkwargs: dict
        Arguments given to the ExcitationsCalculator object
    exext: string
        Extension for filenames of Excitation lists (results of
        the ExcitationsCalculator).
    overlap : function or False
        Function to calculate overlaps between excitation at
        equilibrium and at a displaced position. Calculators are
        given as first and second argument, respectively.

    Example
    -------

    >>> from ase.calculators.h2morse import (H2Morse,
    ...                                      H2MorseExcitedStatesCalculator)
    >>> from ase.vibrations.resonant_raman import ResonantRamanCalculator
    >>>
    >>> atoms = H2Morse()
    >>> rmc = ResonantRamanCalculator(atoms, H2MorseExcitedStatesCalculator)
    >>> rmc.run()

    This produces all necessary data for further analysis.
    """
    def __init__(self,
                 vibrations: VibrationsRunner,
                 ExcitationsCalculator: tp.Type[EC],
                 exkwargs=None,
                 exname=str,
                 exext='.ex.gz',
                 overlap: tp.Optional[tp.Callable[[Calculator, Calculator], np.ndarray]]=None,
                 ):
        self._atoms = vibrations.atoms
        self._vibrations = vibrations
        self._ExListCalc = ExcitationsCalculator
        self._exname = exname
        self._exkwargs = exkwargs or {}
        self._overlap = overlap
        self._exext = exext

        if self._overlap:
            self._eq_calculator = _copy_atoms_calc(self._atoms)
        else:
            self._eq_calculator = None

    def calculate(self, atoms: Atoms, disp: Displacement):
        """Call ground and excited state calculation"""
        assert atoms == self._atoms  # XXX action required
        returnvalue = self._vibrations.calculate(atoms, disp)

        if self._overlap:
            """Overlap is determined as

            ov_ij = int dr displaced*_i(r) eqilibrium_j(r)
            """
            ov_nn = self._overlap(self._atoms.calc,
                                  self._eq_calculator)
            self.save_ov_nn(ov_nn, disp)

        exlist = self.compute_exlist(atoms)
        self.save_exlist(exlist, disp)
        return returnvalue

    def run(self):
        self._vibrations.run()

    def _exlist_filename(self, disp: Displacement):
        return Path(self._exname) / f'pol.{disp.name}{self._exext}'

    def compute_exlist(self, atoms: Atoms):
        excalc = self._ExListCalc(**self._exkwargs)  # type: ignore
        return excalc.calculate(atoms)

    def save_exlist(self, exlist: ExcitationList, disp: Displacement):
        # XXX each exobj should allow for self._exname as Path
        exlist.write(str(self._exlist_filename(disp)))

    def load_exlist(self, disp: Displacement) -> ExcitationList:
        # XXX each exobj should allow for self._exname as Path
        excalc = self._ExListCalc(**self._exkwargs)  # type: ignore
        return excalc.read(str(self._exlist_filename(disp)), **self._exkwargs)

    def _ov_nn_filename(self, disp: Displacement):
        return Path(self._exname) / (disp.name + '.ov')

    def save_ov_nn(self, ov_nn: np.ndarray, disp: Displacement):
        if world.rank == 0:
            np.save(self._ov_nn_filename(disp), ov_nn)

    def load_ov_nn(self, disp: Displacement) -> np.ndarray:
        return np.load(self._ov_nn_filename(disp))


class StaticPolarizabilityRamanRunner:
    """FIXME TODO"""
    def __init__(self,
                 vibrations: VibrationsRunner,
                 PolarizabilityCalculator: tp.Type[PC],
                 polkwargs=None,
                 polname=str,
                 polext='.pol.gz',
                 ):
        self._vibrations = vibrations
        self._polobj = PolarizabilityCalculator
        self._polname = polname
        self._polkwargs = polkwargs or {}
        self._polext = polext

    def calculate(self, atoms: Atoms, disp: Displacement):
        returnvalue = self._vibrations.calculate(atoms, disp)
        pol_tensor = self.compute_static_polarizability(atoms)
        self.save_static_polarizability(pol_tensor, disp)
        return returnvalue

    def run(self):
        self._vibrations.run()

    def _static_polarizability_filename(self, disp: Displacement):
        return Path(self._polname) / f'pol.{disp.name}{self._polext}'

    def compute_static_polarizability(self, atoms: Atoms) -> np.ndarray:
        polobj = self._PolCalc(**self._polkwargs)  # type: ignore
        pol_tensor = polobj(atoms)
        return pol_tensor

    def save_static_polarizability(self, pol_tensor: np.ndarray, disp: Displacement):
        if world.rank == 0:
            np.savetxt(self._static_polarizability_filename(disp), pol_tensor)

    def load_static_polarizability(self, disp: Displacement):
        return np.loadtxt(self._static_polarizability_filename(disp))
