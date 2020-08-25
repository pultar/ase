# Only for debugging
import inspect

import numpy as np
import pytest

from ase.build import add_adsorbate, bulk, fcc100, fcc111
from ase.calculators.emt import EMT
from ase.constraints import FixAtoms
from ase.neb import NEB
from ase.optimize import BFGS
from ase.rneb import RNEB, reshuffle_positions


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
    rneb = RNEB(logfile=None)

    # deleting ions will change indices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[vacancy_path[0]]

    final_unrelaxed = atoms.copy()
    del final_unrelaxed[vacancy_path[1]]

    # align indices
    final_unrelaxed = reshuffle_positions(initial_unrelaxed, final_unrelaxed)

    # RNEB symmetry identification
    sym_ops = rneb.find_symmetries(atoms, initial_unrelaxed,
                                   final_unrelaxed)
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

    final_relaxed_s = rneb.get_final_image(atoms, initial_unrelaxed,
                                           initial_relaxed,
                                           final_unrelaxed)

    ef_n = final_relaxed_n.get_potential_energy()
    ef_s = final_relaxed_s.get_potential_energy()
    try:
        assert np.isclose(ef_n, ef_s, atol=1e-3)
    except AssertionError:
        print(inspect.stack()[1].function)
        print(vacancy_path)
        print('e final really relaxed: ', ef_n)
        print('e final symmetry:       ', ef_s)

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
            neb = NEB(images, sym=True, rotations=S, method=method)
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


@pytest.fixture(scope="module")  # reuse the same object for the whole script
def initial_Al_fcc100_slab():
    # 3x3-Al(001) surface with 3 layers and an
    # Au atom adsorbed in a hollow site:
    slab = fcc100('Al', size=(3, 3, 3))
    slab.center(axis=2, vacuum=4.0)

    # Fix second and third layers:
    mask = [atom.tag > 1 for atom in slab]
    constraint = FixAtoms(mask=mask)
    slab.set_constraint(constraint)

    add_adsorbate(slab, 'Au', 1.7, 'hollow')
    add_adsorbate(slab, 'Au', 1.7, 'hollow', offset=(1, 0))

    return slab


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

# # Superfluous test; the same (and more) is covered by test_rmi_rneb
# def test_rmi_rneb_slab(initial_Al_fcc100_slab):
#     atoms = initial_Al_fcc100_slab
#     dft_tol = 1e-3  # 1 meV

#     # deleting ions will change indices
#     initial_unrelaxed = atoms.copy()
#     del initial_unrelaxed[-1]

#     final_unrelaxed = atoms.copy()
#     del final_unrelaxed[-2]

#     # RMI-NEB workflow
#     # RNEB symmetry identification
#     rneb = RNEB(logfile=None)
#     sym_ops = rneb.find_symmetries(atoms, initial_unrelaxed, final_unrelaxed)
#     # check path is reflective
#     assert len(sym_ops) > 0

#     initial_relaxed = initial_unrelaxed.copy()
#     initial_relaxed.calc = EMT()
#     qn = BFGS(initial_relaxed, logfile=None)
#     qn.run(fmax=0.01)

#     # get final relaxed image from initial relaxed
#     final_relaxed_rneb = rneb.get_final_image(atoms, initial_unrelaxed,
#                                               initial_relaxed,
#                                               final_unrelaxed)
#     images_rmi = create_path(initial_relaxed, final_relaxed_rneb)
#     # only three images
#     images_rmi = [images_rmi[0], images_rmi[2], images_rmi[-1]]
#     neb = NEB(images_rmi, method='aseneb')
#     neb.interpolate()

#     # run RMI-NEB
#     qn = BFGS(neb, logfile=None)
#     qn.run(fmax=0.01)

#     # Normal NEB
#     final_relaxed_n = final_unrelaxed.copy()
#     final_relaxed_n.calc = EMT()
#     qn = BFGS(final_relaxed_n, logfile=None)
#     qn.run(fmax=0.01)

#     # also the reference NEB
#     images_n = create_path(initial_relaxed, final_relaxed_n)
#     neb = NEB(images_n, method='aseneb')
#     neb.interpolate()

#     qn = BFGS(neb, logfile=None)
#     qn.run(fmax=0.01)

#     # are energies/transition state comaparable to normal NEB?
#     e_m_n = images_n[2].get_potential_energy()
#     e_m_rmi = images_rmi[1].get_potential_energy()
#     assert abs(e_m_n - e_m_rmi) < dft_tol


def test_rmi_rneb(initial_structures_fcc111_al):
    """ running RMI-NEB followed by reflective NEB on half the path """
    # accepted energy difference between normal and reflective NEB
    dft_tol = 1e-3  # 1 meV

    (atoms, final_unrelaxed, initial_unrelaxed, initial_relaxed,
     final_relaxed_n, images_n) = initial_structures_fcc111_al

    # RMI-NEB workflow
    # RNEB symmetry identification
    rneb = RNEB(logfile=None)
    sym_ops = rneb.find_symmetries(atoms, initial_unrelaxed, final_unrelaxed)
    # check path is reflective
    assert len(sym_ops) > 0
    # get final relaxed image from initial relaxed
    final_relaxed_rneb = rneb.get_final_image(atoms, initial_unrelaxed,
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
    print('Trying path 0 -> 1')
    sym = get_num_sym_operators(atoms, path=(0, 1))
    assert len(sym) == 1

    print('Trying path 1 -> 2')
    sym = get_num_sym_operators(atoms, path=(1, 2))
    assert len(sym) == 2


def get_num_sym_operators(atoms, path):
    # deleting ions will change inidices
    initial_unrelaxed = atoms.copy()
    del initial_unrelaxed[path[0]]

    final_unrelaxed = atoms.copy()
    del final_unrelaxed[path[1]]

    # aling indices
    final_unrelaxed = reshuffle_positions(initial_unrelaxed,
                                          final_unrelaxed)
    images = create_path(initial_unrelaxed, final_unrelaxed)
    neb = NEB(images)
    neb.interpolate()

    rneb = RNEB(logfile=None)
    S = rneb.find_symmetries(atoms, initial_unrelaxed, final_unrelaxed)

    sym = rneb.reflect_path(images, sym=S)
    print(
        f"Allowed reflective operations for {path[0]}->{path[1]}: {len(sym)}")
    assert sym is not None  # otherwise not reflective

    return sym


def align_indices(initial, final):
    """
    move the atoms in final such that the indices match with initial.

    Parameters
    ---------
    initial: ASE atoms-object

    final: ASE atoms-object

    Return
    -------
    ASE-atoms object
    """

    sort_list = match_atoms(initial, final)
    final_sorted = final.copy()
    final_sorted = final_sorted[sort_list]

    return final_sorted


def match_atoms(initial, final):
    """
    Match the atomic indices in final to the ones in initial based on
    position

    Parameters
    ---------

    initial: ASE atoms-object
    The initial unrelaxed structure

    final: ASE atoms-object
    The final unrelaxed structure

    Returns
    -------

    sorting list: numpy array 1 x n of integers
    """
    import numpy as np
    pos_initial = initial.get_positions()
    pos_final = final.get_positions()
    symb_initial = initial.get_chemical_symbols()
    symb_final = final.get_chemical_symbols()
    sort_list = np.ones(len(initial)).astype(int) * -1
    no_match = []
    for idx_final, state in enumerate(zip(pos_final, symb_final)):
        pf, sf = state
        idx_initial = find_atom(pos_initial, symb_initial, pf, sf,
                                critical=False)
        if idx_initial is False:
            no_match.append(idx_final)
        else:
            sort_list[idx_initial] = idx_final
    missing = np.where(sort_list == -1)
    if len(missing) != len(no_match):
        if missing[0] != no_match[0]:
            msg = ("different number of atoms have moved "
                   "in the initial and final structures")
            raise RuntimeError(msg)

    if len(no_match) > 1:
        msg = ("Found more than one moving atom. "
               "Please give information about where "
               "the moving atoms start and finish")
        raise RuntimeError(msg)
    sort_list[missing[0]] = no_match[0]
    return sort_list


def find_atom(pos, symb, pos_d, symb_d, critical=True, tol=1e-3):
    """
    Find the atom matching pos_d.

    Parameters
    ---------

    pos: numpy array n x 3
    positions to be matched against

    sym: string
    chemical symbols to be matched against

    pos_d: numpy array 1 x 3
    position to be found

    sym_d: string
    chemical symbols to be found

    critical: boolean
    whether to fail upon not being able to make a match

    Return
    -------
    integer if match is found otherwise False: integer or boolean
    """
    import numpy as np
    for i, state in enumerate(zip(pos, symb)):
        p, s = state
        if s == symb_d:
            dist = np.linalg.norm(p - pos_d)
            if dist < tol:
                return i
    if critical:
        msg = ("The speified atoms was not found. "
               "Make sure you given the correct positions and symbols")
        raise RuntimeError(msg)
    else:
        return False
