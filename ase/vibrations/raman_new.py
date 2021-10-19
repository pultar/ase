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


class ResonantRamanRunner:
    """TODO DOC"""
    def __init__(self,
                 vibrations: VibrationsRunner,
                 excalc: ExcitationListCalculator,
                 exname=str,
                 exext='.ex.gz',
                 overlap: tp.Optional[tp.Callable[[Calculator, Calculator], np.ndarray]]=None,
                 ):
        self._atoms = vibrations.atoms
        self._vibrations = vibrations
        self._excalc = excalc
        self._exname = exname
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
            # Overlap is determined as
            #
            # ov_ij = int dr displaced*_i(r) eqilibrium_j(r)
            ov_nn = self._overlap(self._atoms.calc,
                                  self._eq_calculator)
            self.save_ov_nn(ov_nn, disp)

        exlist = self._compute_exlist(atoms)
        self.save_exlist(exlist, disp)
        return returnvalue

    def run(self):
        self._vibrations.run()

    def _exlist_filename(self, disp: Displacement):
        return Path(self._exname) / f'ex.{disp.name}{self._exext}'

    def _compute_exlist(self, atoms: Atoms):
        return self._excalc.calculate(atoms)

    def save_exlist(self, exlist: ExcitationList, disp: Displacement):
        # XXX each exobj should allow for self._exname as Path
        exlist.write(str(self._exlist_filename(disp)))

    def load_exlist(self, disp: Displacement) -> ExcitationList:
        # XXX each exobj should allow for self._exname as Path
        return self._excalc.read(str(self._exlist_filename(disp)))

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
                 polcalc: StaticPolarizabilityCalculator,
                 polname=str,
                 polext='.pol.gz',
                 ):
        self._vibrations = vibrations
        self._polcalc = polcalc
        self._polname = polname
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
        pol_tensor = self._polcalc(atoms)
        return pol_tensor

    def save_static_polarizability(self, pol_tensor: np.ndarray, disp: Displacement):
        if world.rank == 0:
            np.savetxt(self._static_polarizability_filename(disp), pol_tensor)

    def load_static_polarizability(self, disp: Displacement):
        return np.loadtxt(self._static_polarizability_filename(disp))

# TODO: need something Plazcek related to get RamanOutput from StaticPolarizabilityRamanRunner
#
