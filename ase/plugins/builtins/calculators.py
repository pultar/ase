def calculators():
    from ase.calculators.demon.demon import Demon  # NOQA: F401
    from ase.calculators.kim.kim import KIM  # NOQA: F401
    from ase.calculators.openmx.openmx import OpenMX  # NOQA: F401
    from ase.calculators.siesta.siesta import Siesta  # NOQA: F401
    from ase.calculators.turbomole.turbomole import Turbomole  # NOQA: F401
    from ase.calculators.vasp.vasp import Vasp  # NOQA: F401
    from ase.calculators.abinit import Abinit  # NOQA: F401
    from ase.calculators.acemolecule import ACE  # NOQA: F401
    from ase.calculators.acn import ACN  # NOQA: F401
    from ase.calculators.aims import Aims  # NOQA: F401
    from ase.calculators.amber import Amber  # NOQA: F401
    from ase.calculators.castep import Castep  # NOQA: F401
    from ase.calculators.combine_mm import CombineMM  # NOQA: F401
    from ase.calculators.counterions import AtomicCounterIon  # NOQA: F401
    from ase.calculators.cp2k import CP2K  # NOQA: F401
    from ase.calculators.demonnano import DemonNano  # NOQA: F401
    from ase.calculators.dftb import Dftb  # NOQA: F401
    from ase.calculators.dftd3 import DFTD3  # NOQA: F401
    from ase.calculators.dmol import DMol3  # NOQA: F401
    from ase.calculators.eam import EAM  # NOQA: F401
    from ase.calculators.elk import ELK  # NOQA: F401
    from ase.calculators.emt import EMT  # NOQA: F401
    from ase.calculators.espresso import Espresso  # NOQA: F401
    from ase.calculators.ff import ForceField  # NOQA: F401
    from ase.calculators.gamess_us import GAMESSUS  # NOQA: F401
    from ase.calculators.gaussian import GaussianDynamics  # NOQA: F401
    from ase.calculators.gromacs import Gromacs  # NOQA: F401
    from ase.calculators.lammpslib import LAMMPSlib  # NOQA: F401
    from ase.calculators.lammpsrun import LAMMPS  # NOQA: F401
    from ase.calculators.lj import LennardJones  # NOQA: F401
    from ase.calculators.mopac import MOPAC  # NOQA: F401
    from ase.calculators.morse import MorsePotential  # NOQA: F401
    from ase.calculators.nwchem import NWChem  # NOQA: F401
    from ase.calculators.octopus import Octopus  # NOQA: F401
    from ase.calculators.onetep import Onetep  # NOQA: F401
    from ase.calculators.orca import ORCA  # NOQA: F401
    from ase.calculators.plumed import Plumed  # NOQA: F401
    from ase.calculators.psi4 import Psi4  # NOQA: F401
    from ase.calculators.qchem import QChem  # NOQA: F401
    from ase.calculators.qmmm import SimpleQMMM  # NOQA: F401
    from ase.calculators.qmmm import EIQMMM  # NOQA: F401
    from ase.calculators.qmmm import RescaledCalculator  # NOQA: F401
    from ase.calculators.qmmm import ForceConstantCalculator  # NOQA: F401
    from ase.calculators.qmmm import ForceQMMM  # NOQA: F401
    from ase.calculators.tip3p import TIP3P  # NOQA: F401
    from ase.calculators.tip4p import TIP4P  # NOQA: F401

    from .nonexistent_calculator import mock_if_not_exists
    asap = mock_if_not_exists('asap3', 'EMT')
    gpaw = mock_if_not_exists('gpaw', 'GPAW')
    hobbit = mock_if_not_exists('hobbit', 'Calculator')
    del mock_if_not_exists

    out = locals()
    out = {k: v for k, v in out.items() if not k.startswith('_')}
    return out


#  just expose all the calculators imported above
_calculators = calculators()
locals().update(_calculators)
__all__ = list(_calculators.keys())
