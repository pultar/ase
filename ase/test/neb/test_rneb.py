# Only for debugging
import inspect
from math import sqrt

import numpy as np
import pytest

from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.geometry import distance, find_mic
from ase.neb import NEB
from ase.optimize import BFGS
from ase.rneb import RNEB, reshuffle_positions

pytest.importorskip('spglib')


def create_path(init, final):
    images = [init]
    for i in range(3):
        image = init.copy()
        image.calc = EMT()
        images.append(image)

    images.append(final)
    return images


def test_Al_hexagonal():
    atoms = fcc111('Al', [2, 2, 1], 4.05, periodic=True)
    for path in [(0, 1), (1, 2)]:
        compare_rneb_w_normal_neb(atoms, path)


def compare_rneb_w_normal_neb(atoms, vacancy_path):
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
    try:
        assert np.isclose(ef_n, ef_s, atol=1e-3)
    except AssertionError:
        print(inspect.stack()[1].function)
        print(vacancy_path)
        print('e final really relaxed: ', ef_n)
        print('e final symmetry:       ', ef_s)

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
    try:
        assert len(sym_ops) > 0
    except AssertionError:
        print(inspect.stack()[1].function)
        print(f'Path {vacancy_path} is not reflective')
        return

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
        try:
            assert np.isclose(eip1, efm1, atol=1e-3)
        except AssertionError:
            print(inspect.stack()[1].function)
            print(vacancy_path)
            print(f'Normal NEB (method {method}):')
            print(eip1)
            print(efm1)
            print(f'dE = {eip1 - efm1}')

        for S in sym_ops:
            images = create_path(initial_relaxed, final_relaxed_s)
            # images = create_path(initial_unrelaxed, final_unrelaxed)
            neb = NEB(images, reflect_ops=S, method=method)
            neb.interpolate()
            qn = BFGS(neb, logfile=None)
            qn.run(fmax=0.05)

            eip1 = images[1].get_potential_energy()
            efm1 = images[-2].get_potential_energy()
            try:
                assert np.isclose(eip1, efm1, atol=1e-3)
            except AssertionError:
                print(inspect.stack()[1].function)
                print(vacancy_path)
                print(S, 'not ok')
                print(f'Reflective NEB (method {method}):')
                print(eip1)
                print(efm1)
                print(f'dE = {eip1 - efm1}')

            # Also assert that forces are present and agree


@pytest.fixture(scope="module")  # reuse the same object for the whole script
def initial_structures_fcc111_al():
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
    # deleting ions will change inidices
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


def get_path_length(init, final):
    dR, _ = find_mic(final.positions - init.positions, init.cell, init.pbc)
    return sqrt((dR**2).sum())


def test_reshuffling_atoms():
    # Create two structures
    slab = fcc111('Al', size=(3, 3, 2), a=2)
    slab.center(vacuum=5, axis=2)
    init = slab.copy()
    final = slab.copy()
    # Make vacancies
    init.pop(7)
    final.pop(16)
    fc = final.copy()

    eps = 1e-8

    final = reshuffle_positions(init, final)
    assert distance(final, fc) < eps

    assert get_path_length(init, fc) >= get_path_length(init, final)

    assert get_path_length(init, final) - sqrt(2) < eps
