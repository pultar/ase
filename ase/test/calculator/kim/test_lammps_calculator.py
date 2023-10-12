"""
Test that the KIM calculator correctly interfaces with ASEs lammprun and
lammpslib interfaces.
"""

import numpy as np
from pytest import mark
from ase.lattice.cubic import FaceCenteredCubic
from ase.atoms import Atoms


# ignore warning that semi-periodic cells might be wrong
@mark.calculator
def test_energy_forces_stress_lammpsrun(KIM):
    """
    To test that the calculator can produce correct energy and forces.  This
    is done by comparing the energy for an FCC argon lattice with an example
    model to the known value; the forces/stress returned by the model are
    compared to numerical estimates via finite difference.

    This test is taken from test_energy_forces_stress.py and adds only the
    simulator argument. The partially periodic box is changed to fully periodic
    since lammpsrun does not seem to play nicely with partial periodicity.
    """

    # Create an FCC atoms crystal
    atoms = FaceCenteredCubic(
        directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        size=(1, 1, 1),
        symbol="Ar",
        pbc=(1, 1, 1),
        latticeconstant=3.0,
    )

    # Perturb the x coordinate of the first atom by less than the cutoff
    # distance
    atoms.positions[0, 0] += 0.01

    # this will not work with lammpslib as ex_model_ar_P_Morse_07C is a
    # portable model and portable models are not yet supported in lammpslib
    calc = KIM("ex_model_Ar_P_Morse_07C", simulator="lammpsrun")
    atoms.calc = calc

    # Get energy and analytical forces/stress from KIM model
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

    # Previously computed energy for this configuration for this model
    energy_ref = 46.330936654344425  # eV

    # Compute forces and virial stress numerically
    forces_numer = calc.calculate_numerical_forces(atoms, d=0.0001)
    stress_numer = calc.calculate_numerical_stress(atoms, d=0.0001, voigt=True)

    tol = 1e-6
    assert np.isclose(energy, energy_ref, tol)
    assert np.allclose(forces, forces_numer, tol)
    assert np.allclose(stress, stress_numer, tol)


@mark.calculator
def test_lennard_jones_calculation(KIM):
    """
    Check that for a simulator model the correct energy is calculated in a
    two-atom lennard jones cell.
    """
    # Create an FCC atoms crystal

    for simulator in ["lammpsrun", "lammpslib"]:
        atoms = Atoms("Au2",
                      positions=[(0.1, 0.1, 0.1), (1.1, 0.1, 0.1)],
                      cell=(2, 2, 2))
        calc = KIM("Sim_LAMMPS_LJcut_AkersonElliott_Alchemy_PbAu",
                   simulator=simulator)
        atoms.calc = calc

        # Get energy and analytical forces/stress from KIM model
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        # stress = atoms.get_stress()

        # Calculate directly the energy of the LJ model with the parameters
        # set by KIM
        epsilon = 2.3058000
        sigma = 2.42324
        energy_ref = 4 * epsilon * (sigma**12 - sigma**6)  # eV

        # Compute forces and virial stress numerically
        forces_numer = calc.calculate_numerical_forces(atoms, d=0.0001)
        # stress_numer = calc.calculate_numerical_stress(atoms,
        #                                                d=0.0001,
        #                                                voigt=True)

        tol = 1e-6
        assert np.isclose(energy, energy_ref, tol)
        assert np.allclose(forces, forces_numer, tol)
        # the following does not work, not sure whether it should
        # assert np.allclose(stress, stress_numer, tol)
