from math import sqrt

import numpy as np
import pytest

from ase.build import fcc111
from ase import Atoms
from ase.calculators.emt import EMT
from ase.geometry import distance, find_mic
from ase.neb import NEB
from ase.optimize import BFGS
from ase.constraints import FixAtoms

pytest.importorskip('spglib')


def create_path(initial, final, calc=EMT):
    images = [initial]
    for i in range(3):
        image = initial.copy()
        image.calc = calc()
        images.append(image)

    images.append(final)
    return images


def test_Al_hexagonal():
    atoms = fcc111('Al', [2, 2, 1], 4.05, periodic=True)
    for path in [(0, 1), (1, 2)]:
        compare_rneb_w_normal_neb(atoms, path)


def compare_rneb_w_normal_neb(atoms, vacancy_path):
    from ase.rneb import RNEB
    rneb = RNEB(atoms, logfile=None)

    # deleting ions will change indices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[vacancy_path[0]]

    final_unrelaxed = atoms.copy()
    del final_unrelaxed[vacancy_path[1]]

    # RNEB symmetry identification
    sym_ops = rneb.find_symmetries(initial_unrelaxed,
                                   final_unrelaxed, log_atomic_idx=True)
    # We check here if any symmetry operations relate the initial and
    # final structures, if not then there is no point in continuing.
    # We could also do this with the SymmetryEquivalenceCheck but that
    # does not guarantee that any symmetry operations could be
    # used for path
    assert len(sym_ops) > 0

    initial_relaxed = initial_unrelaxed.copy()
    initial_relaxed.calc = EMT()
    qn = BFGS(initial_relaxed, logfile=None)
    qn.run(fmax=0.01)

    final_relaxed_n = final_unrelaxed.copy()
    final_relaxed_n.calc = EMT()
    qn = BFGS(final_relaxed_n, logfile=None)
    qn.run(fmax=0.01)

    final_relaxed_s = rneb.get_final_image(initial_unrelaxed,
                                           initial_relaxed,
                                           final_unrelaxed, rot_only=True)

    ef_n = final_relaxed_n.get_potential_energy()
    ef_s = final_relaxed_s.get_potential_energy()
    assert np.isclose(ef_n, ef_s, atol=1e-3)

    f_n = final_relaxed_n.get_forces()
    f_s = final_relaxed_s.get_forces()
    assert np.allclose(f_n, f_s, atol=1e-3)

    # Create path for symmetry detection
    images = create_path(initial_unrelaxed, final_unrelaxed)
    neb = NEB(images)
    neb.interpolate()

    # check that path has reflection symmetry for each image pair
    sym_ops = rneb.reflect_path(images, sym=sym_ops)
    # Also assert that the entire path has reflection symmetry
    # otherwise the rest is pointless
    assert len(sym_ops) > 0

    for method in ['aseneb', 'improvedtangent', 'eb']:
        # Create path for normal NEB
        images = create_path(initial_relaxed, final_relaxed_n)
        neb = NEB(images, method=method)
        neb.interpolate()

        qn = BFGS(neb, logfile=None)
        qn.run(fmax=0.05)

        # Normal NEB
        eip1 = images[1].get_potential_energy()  # e initial plus 1
        efm1 = images[-2].get_potential_energy()  # e final minus 1
        assert np.isclose(eip1, efm1, atol=1e-3)

        for S in sym_ops:
            images = create_path(initial_relaxed, final_relaxed_s)
            # images = create_path(initial_unrelaxed, final_unrelaxed)
            neb = NEB(images, reflect_ops=S, method=method)
            neb.interpolate()
            qn = BFGS(neb, logfile=None)
            qn.run(fmax=0.05)

            eip1 = images[1].get_potential_energy()
            efm1 = images[-2].get_potential_energy()
            assert np.isclose(eip1, efm1, atol=1e-3)

            # Also assert that forces are present and agree


@pytest.fixture(scope="module")  # reuse the same object for the whole script
def initial_structures_fcc111_al():
    from ase.rneb import reshuffle_positions
    atoms = fcc111('Al', [2, 2, 1], 4.05, periodic=True)
    nxy = np.sum(atoms.cell, axis=0) / 6
    nxy[2] = atoms.cell[2][2]
    atoms.cell[2] = nxy

    atoms.calc = EMT()
    qn = BFGS(atoms, logfile=None)
    qn.run(fmax=0.01)

    vacancy_path = [0, 1]
    # deleting ions will change indices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[vacancy_path[0]]

    final_unrelaxed = atoms.copy()
    del final_unrelaxed[vacancy_path[1]]

    # align indices
    final_unrelaxed = reshuffle_positions(initial_unrelaxed, final_unrelaxed)

    initial_relaxed = initial_unrelaxed.copy()
    initial_relaxed.calc = EMT()
    qn = BFGS(initial_relaxed, logfile=None)
    qn.run(fmax=0.01)

    final_relaxed_n = final_unrelaxed.copy()
    final_relaxed_n.calc = EMT()
    qn = BFGS(final_relaxed_n, logfile=None)
    qn.run(fmax=0.01)

    # also the reference NEB
    images_n = create_path(initial_relaxed, final_relaxed_n)
    neb = NEB(images_n, method='aseneb')
    neb.interpolate()

    qn = BFGS(neb, logfile=None)
    qn.run(fmax=0.01)

    return (atoms, final_unrelaxed, initial_unrelaxed,
            initial_relaxed, final_relaxed_n, images_n)


def test_rmi_rneb(initial_structures_fcc111_al):
    """ running RMI-NEB followed by reflective NEB on half the path """
    from ase.rneb import RNEB

    # accepted energy difference between normal and reflective NEB
    dft_tol = 1e-3  # 1 meV

    (atoms, final_unrelaxed, initial_unrelaxed, initial_relaxed,
     final_relaxed_n, images_n) = initial_structures_fcc111_al

    # RMI-NEB workflow
    # RNEB symmetry identification
    rneb = RNEB(atoms, logfile=None)
    sym_ops = rneb.find_symmetries(initial_unrelaxed, final_unrelaxed)
    # check path is reflective
    assert len(sym_ops) > 0
    # get final relaxed image from initial relaxed
    final_relaxed_rneb = rneb.get_final_image(initial_unrelaxed,
                                              initial_relaxed,
                                              final_unrelaxed)
    images_rmi = create_path(initial_relaxed, final_relaxed_rneb)
    # only three images
    images_rmi = [images_rmi[0], images_rmi[2], images_rmi[-1]]
    neb = NEB(images_rmi, method='aseneb')
    neb.interpolate()

    # run RMI-NEB
    qn = BFGS(neb, logfile=None)
    qn.run(fmax=0.01)

    # are energies comaparable to normal NEB?
    e_m_n = images_n[2].get_potential_energy()
    e_m_rmi = images_rmi[1].get_potential_energy()
    assert abs(e_m_n - e_m_rmi) < dft_tol

    # now the NEB on half the path
    images_half = create_path(initial_relaxed, images_rmi[1])
    images_half = [images_half[0], images_half[2], images_half[-1]]
    neb = NEB(images_half, method='aseneb')
    neb.interpolate()

    # run RMI-NEB
    qn = BFGS(neb, logfile=None)
    qn.run(fmax=0.01)

    # are energies comparable to normal NEB?
    e_first_n = images_n[1].get_potential_energy()
    e_third_n = images_n[3].get_potential_energy()
    e_first_half = images_half[1].get_potential_energy()
    assert abs(e_first_half - e_first_n) < dft_tol
    assert abs(e_first_half - e_third_n) < dft_tol


def test_number_reflection_operators(initial_structures_fcc111_al):
    """ Test to check paths separately for reflection symmetry """
    atoms = initial_structures_fcc111_al[0]

    # give path to look at as path=(init_indice, final_indice)
    # Trying path 0 -> 1
    sym = get_num_sym_operators(atoms, path=(0, 1))
    assert len(sym) == 1

    # Trying path 1 -> 2
    sym = get_num_sym_operators(atoms, path=(1, 2))
    assert len(sym) == 2


def get_num_sym_operators(atoms, path):
    from ase.rneb import RNEB

    # deleting ions will change indices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[path[0]]

    final_unrelaxed = atoms.copy()
    del final_unrelaxed[path[1]]

    images = create_path(initial_unrelaxed, final_unrelaxed)
    neb = NEB(images)
    neb.interpolate()

    rneb = RNEB(atoms, logfile=None)
    S = rneb.find_symmetries(initial_unrelaxed, final_unrelaxed)

    sym = rneb.reflect_path(images, sym=S)
    assert sym is not None  # otherwise not reflective

    return sym


def get_path_length(initial, final):
    dR, _ = find_mic(final.positions - initial.positions,
                     initial.cell, initial.pbc)
    return sqrt((dR**2).sum())


def test_reshuffling_atoms():
    from ase.rneb import reshuffle_positions

    # Create two structures
    slab = fcc111('Al', size=(3, 3, 2), a=2)
    slab.center(vacuum=5, axis=2)
    initial = slab.copy()
    final = slab.copy()
    # Make vacancies
    initial.pop(7)
    final.pop(16)
    fc = final.copy()

    eps = 1e-8

    final = reshuffle_positions(initial, final)
    assert distance(final, fc) < eps

    assert get_path_length(initial, fc) >= get_path_length(initial, final)

    assert get_path_length(initial, final) - sqrt(2) < eps


def test_reflective_images():
    from ase.calculators.lj import LennardJones
    from ase.rneb import reshuffle_positions, RNEB

    def relax(atoms):
        atoms.calc = LennardJones()
        dyn = BFGS(atoms, logfile=None)
        dyn.run(fmax=0.01)

    both = Atoms('H8',
                 positions=[(0, 0, 0),
                            (1, 0, 0),
                            (0, 1, 0),
                            (1, 1, 0),
                            (0, 2, 0),
                            (1, 2, 0),
                            (0.5, 0.5, 1),
                            (0.5, 1.5, 1)],
                 cell=[2, 3, 2],
                 constraint=[FixAtoms(range(6))])

    initial_unrelaxed = both.copy()
    initial_unrelaxed.pop(-1)
    initial_relaxed = initial_unrelaxed.copy()
    final_unrelaxed = both.copy()
    final_unrelaxed.pop(-2)
    final_relaxed = final_unrelaxed.copy()

    for atoms in [both, initial_relaxed, final_relaxed]:
        relax(atoms)

    images = create_path(initial_unrelaxed, final_unrelaxed, LennardJones)
    NEB(images).interpolate()

    rneb = RNEB(both, logfile=None)
    S = rneb.find_symmetries(initial_unrelaxed, final_unrelaxed)

    final_relaxed_s = rneb.get_final_image(initial_unrelaxed,
                                           initial_relaxed,
                                           final_unrelaxed, rot_only=True)

    ef_n = final_relaxed.get_potential_energy()
    ef_s = final_relaxed_s.get_potential_energy()
    assert np.isclose(ef_n, ef_s, atol=1e-3)

    f_n = final_relaxed.get_forces()
    f_s = final_relaxed_s.get_forces()
    assert np.allclose(f_n, f_s, atol=1e-3)

    for method in ['aseneb', 'improvedtangent', 'eb']:
        # Create path for normal NEB
        images = create_path(initial_relaxed, final_relaxed, LennardJones)
        neb = NEB(images, method=method)
        neb.interpolate()

        qn = BFGS(neb, logfile=None)
        qn.run(fmax=0.05)

        # Normal NEB
        eip1 = images[1].get_potential_energy()  # e initial plus 1
        efm1 = images[-2].get_potential_energy()  # e final minus 1
        assert np.isclose(eip1, efm1, atol=1e-3)
        
        images = create_path(initial_relaxed, final_relaxed_s, LennardJones)
        NEB(images).interpolate()
        refl_images = rneb.get_reflective_path(images, S)

        neb = NEB(refl_images, method=method)

        qn = BFGS(neb, logfile=None)
        qn.run(fmax=0.05)
        
        eip1 = refl_images[1].get_potential_energy()
        fip1_abs = np.abs(refl_images[1].get_forces())

        # First check that the energy and forces are in fact transfered
        # from the symmetric image
        assert np.isclose(eip1, refl_images[-2].get_potential_energy())
        assert np.allclose(fip1_abs, np.abs(refl_images[-2].get_forces()))

        # Then check the actual energy and forces of the symmetric image
        m2 = refl_images[-2].copy()
        m2.calc = LennardJones()
        efm2 = m2.get_potential_energy()
        ffm2_abs = np.abs(m2.get_forces())
        assert np.isclose(eip1, efm2, atol=1e-3)
        assert np.allclose(fip1_abs, ffm2_abs, atol=1e-3)

        
        
