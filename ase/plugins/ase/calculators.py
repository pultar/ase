def calculators():
    from ase.calculators.demon.demon import Demon
    #from ase.calculators.kim.kim import KIM
    from ase.calculators.openmx.openmx import OpenMX
    from ase.calculators.siesta.siesta import Siesta
    from ase.calculators.turbomole.turbomole import Turbomole
    from ase.calculators.vasp.vasp import Vasp
    from ase.calculators.abinit import Abinit
    from ase.calculators.acemolecule import ACE
    from ase.calculators.acn import ACN
    from ase.calculators.aims import Aims
    from ase.calculators.amber import Amber
    from ase.calculators.castep import Castep
    from ase.calculators.combine_mm import CombineMM
    from ase.calculators.counterions import AtomicCounterIon
    from ase.calculators.cp2k import CP2K
    from ase.calculators.demonnano import DemonNano
    from ase.calculators.dftb import Dftb
    from ase.calculators.dftd3 import DFTD3
    from ase.calculators.dmol import DMol3
    from ase.calculators.eam import EAM
    from ase.calculators.elk import ELK
    from ase.calculators.emt import EMT
    from ase.calculators.espresso import Espresso
    from ase.calculators.ff import ForceField
    from ase.calculators.gamess_us import GAMESSUS
    #from ase.calculators.gaussian import GaussianDynamics
    from ase.calculators.gromacs import Gromacs
    from ase.calculators.lammpslib import LAMMPSlib
    from ase.calculators.lammpsrun import LAMMPS
    from ase.calculators.lj import LennardJones
    from ase.calculators.mopac import MOPAC
    from ase.calculators.morse import MorsePotential
    from ase.calculators.nwchem import NWChem
    from ase.calculators.octopus import Octopus
    from ase.calculators.onetep import Onetep
    from ase.calculators.orca import ORCA
    from ase.calculators.plumed import Plumed
    from ase.calculators.psi4 import Psi4
    from ase.calculators.qchem import QChem
    from ase.calculators.qmmm import SimpleQMMM
    from ase.calculators.qmmm import EIQMMM
    from ase.calculators.qmmm import RescaledCalculator
    from ase.calculators.qmmm import ForceConstantCalculator
    from ase.calculators.qmmm import ForceQMMM
    from ase.calculators.tip3p import TIP3P
    from ase.calculators.tip4p import TIP4P

    from .nonexistent_calculator import mock_if_not_exists
    asap=mock_if_not_exists('asap3', 'EMT')
    gpaw=mock_if_not_exists('gpaw', 'GPAW')
    hobbit=mock_if_not_exists('hobbit', 'Calculator')
    del mock_if_not_exists

    out = locals()
    out = {k:v for k,v in out.items() if not k.startswith('_')}
    return out

#  just expose all the calculators imported above
calculators = calculators()
locals().update(calculators)
__all__ = list(calculators.keys())
