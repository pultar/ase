"""Test that KIM works with a relaxation"""

from ase.cluster import Icosahedron
from ase.calculators.kim import KIM
from ase.optimize import BFGS


def test_relax():
    # Create structure
    atoms = Icosahedron('Ar', latticeconstant=3., noshells=2)

    # create calculator
    modelname = 'ex_model_Ar_P_Morse_07C'
    calc = KIM(modelname)

    # attach calculator to the atoms
    atoms.set_calculator(calc)

    opt = BFGS(atoms)
    opt.run(fmax=0.05)


if __name__ == '__main__':
    test_relax()
