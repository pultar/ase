from math import sqrt

import numpy as np
import pytest

from ase import Atoms
from ase.build import fcc111
from ase.calculators.emt import EMT
from ase.calculators.lj import LennardJones
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import distance, find_mic, get_distances
from ase.neb import NEB, interpolate
from ase.optimize import BFGS
from ase.spacegroup import crystal

pytest.importorskip('spglib')


def create_path(initial, final, calc=EMT):
    images = [initial]
    for i in range(3):
        image = initial.copy()
        image.calc = calc()
        images.append(image)

    images.append(final)
    return images


def optimize(atoms_or_neb, calc=None):
    if calc is not None:
        atoms_or_neb.calc = calc()
    dyn = BFGS(atoms_or_neb, logfile=None)
    dyn.run(fmax=0.01)


@pytest.fixture(scope="module")  # reuse the same object for the whole script
def initial_structures_fcc111_al():
    from ase.rneb import reshuffle_positions
    atoms = fcc111('Al', [2, 2, 1], 4.05, periodic=True)
    nxy = np.sum(atoms.cell, axis=0) / 6
    nxy[2] = atoms.cell[2][2]
    atoms.cell[2] = nxy

    optimize(atoms, calc=EMT)

    vacancy_path = [0, 1]
    # deleting ions will change indices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[vacancy_path[0]]

    final_unrelaxed = atoms.copy()
    del final_unrelaxed[vacancy_path[1]]

    # align indices
    final_unrelaxed = reshuffle_positions(initial_unrelaxed, final_unrelaxed)

    initial_relaxed = initial_unrelaxed.copy()
    optimize(initial_relaxed, EMT)

    final_relaxed_n = final_unrelaxed.copy()
    optimize(final_relaxed_n, EMT)

    # also the reference NEB
    images_n = create_path(initial_relaxed, final_relaxed_n)
    neb = NEB(images_n, method='aseneb')
    neb.interpolate()

    optimize(neb)

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
    optimize(neb)

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
    optimize(neb)

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


@pytest.fixture(scope="module")
def al2s3():
    # Make alpha-Al2S3 in which we have equivalent positions with no
    # reflective path between
    atoms = crystal(['S', 'S', 'S', 'Al', 'Al'],
                    basis=[[0.00003, 0.33469, 0],
                           [0.3343, 1.01191, 0.00395],
                           [0.66198, 0.66843, -.00085],
                           [0.34487, 0.35791, 0.04327],
                           [-.01134, 0.32257, -.12815]],
                    cellpar=[6.430, 6.430, 17.880, 90, 90, 120],
                    spacegroup=169)
    return atoms


def test_all_equivalent_structures(al2s3):
    from ase.rneb import RNEB

    rneb = RNEB(al2s3, logfile=None)

    initial = al2s3.copy()
    initial.pop(-1)

    images = rneb.get_all_equivalent_images(initial)

    # Get the equivalent index of the popped atom
    equivalent_indices = rneb.all_sym_ops['equivalent_atoms']
    eq_idx = equivalent_indices[-1]
    indices = np.nonzero(equivalent_indices == eq_idx)[0]
    equivalent_positions = al2s3.positions[indices]

    # The structures in images should each cover all equivalent
    # positions but one
    for image in images:
        gd = get_distances(equivalent_positions, image.positions,
                           cell=al2s3.cell, pbc=True)[1]
        sorted_dists = np.sort(np.min(gd, axis=1))
        assert np.allclose(sorted_dists[:-1], np.zeros(len(sorted_dists) - 1))
        assert sorted_dists[-1] > 1e-3


def test_non_reflective_path(al2s3):
    from ase.rneb import RNEB

    rneb = RNEB(al2s3, logfile=None)

    initial = al2s3.copy()
    initial.pop(-1)

    final = al2s3.copy()
    final.pop(-2)

    all_sym_ops = rneb.find_symmetries(initial, final)
    assert len(all_sym_ops) > 0

    images = [initial.copy() for _ in range(4)] + [final]
    interpolate(images)
    assert len(rneb.reflect_path(images, all_sym_ops)) == 0
    assert rneb.get_reflective_path(images, all_sym_ops) == images


def test_reflective_images():
    from ase.rneb import RNEB

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
        optimize(atoms, calc=LennardJones)

    # RNEB symmetry identification
    rneb = RNEB(both, logfile=None)
    all_sym_ops = rneb.find_symmetries(initial_unrelaxed, final_unrelaxed,
                                       log_atomic_idx=True)
    # We check here if any symmetry operations relate the initial and
    # final structures, if not then there is no point in continuing.
    # We could also do this with the SymmetryEquivalenceCheck but that
    # does not guarantee that any symmetry operations could be
    # used for path
    assert len(all_sym_ops) > 0

    final_relaxed_s: Atoms = rneb.get_final_image(initial_unrelaxed,
                                                  initial_relaxed,
                                                  final_unrelaxed, rot_only=True)

    # Check that the final image created by symmetry operations has
    # the same energy and forces as final_relaxed
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

        optimize(neb)

        # Check that the normal NEB path is reflective by comparing
        # energies of symmetric images
        eip1 = images[1].get_potential_energy()  # e initial plus 1
        efm1 = images[-2].get_potential_energy()  # e final minus 1
        assert np.isclose(eip1, efm1, atol=1e-3)

        # Create the reflective path in order to run a NEB while
        # taking advantage of the symmetry
        images = create_path(initial_relaxed, final_relaxed_s, LennardJones)
        interpolate(images)
        refl_images = rneb.get_reflective_path(images, all_sym_ops)
        assert refl_images != images

        neb = NEB(refl_images, method=method)

        optimize(neb)

        eip1 = refl_images[1].get_potential_energy()
        fip1_abs = np.abs(refl_images[1].get_forces())

        # First check that the energy and forces are in fact transfered
        # from the symmetric image
        assert isinstance(refl_images[-2].calc, SinglePointCalculator)
        assert np.isclose(eip1, refl_images[-2].get_potential_energy())
        assert np.allclose(fip1_abs, np.abs(refl_images[-2].get_forces()))

        # Then check the actual energy and forces of the symmetric image
        m2 = refl_images[-2].copy()
        m2.calc = LennardJones()
        efm2 = m2.get_potential_energy()
        ffm2_abs = np.abs(m2.get_forces())
        assert np.isclose(eip1, efm2, atol=1e-3)
        assert np.allclose(fip1_abs, ffm2_abs, atol=1e-3)
