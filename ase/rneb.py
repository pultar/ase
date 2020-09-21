"""The reflective NEB class.

The method is described in: Nicolai Rask Mathiesen, Hannes Jónsson,
Tejs Vegge, and Juan Maria García Lastra, J. Chem. Theory Comput.,
2019, 15 (5), pp 3215–3222, (doi: 10.1021/acs.jctc.8b01229).
"""
import logging

import numpy as np

from ase.calculators.calculator import PropertyNotImplementedError
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic, get_distances, wrap_positions
from ase.utils import atoms_to_spglib_cell


class RNEB:
    """The ASE implementation of the R-NEB method.

    Supply the original cell when instantiating the RNEB class. The
    size of the original cell should be equal to the size of the
    initial and final structures. The original cell is used for
    detecting the symmetry operations using spglib. After
    instantiation of the RNEB class the following functions are
    available:

    :func:`find_symmetries` returns the symmetry operators relating
    the initial and final structures.

    :func:`get_final_image` Given two symmetry equivalent structures
    (e.g. ``initial`` and ``final``) and a relaxed version of one of them
    (``initial_relaxed``) the relaxed version of the other structure
    (``final_relaxed``) is produced.

    Given an list of images interpolated between initial and final
    :func:`get_reflective_path` returns a list of reflective images
    that can be given to the main NEB class to utilize.

    :func:`reflect_path` checks if the full path has reflection
    symmetry by testing the given symmetry operations. Returns the
    valid reflection operations.

    Args:
        orig (Atoms): Original cell without defects, the symmetry operations
            are found using this cell.
        tol (float): Tolerance on atomic distances in Å
        logfile (str or None): filename for the log output
            If ``None`` then no output is made. Default is ``'-'`` for standard
            stream output.

    """

    def __init__(self, orig, tol=1e-3, logfile='-'):
        self.tol = tol  # tolerance on atomic positions
        self.log = self._setup_log(logfile)

        self.orig = orig
        self.all_sym_ops = self._get_symmetry(orig)

    def _get_symmetry(self, orig):
        import spglib as spg

        # Get symmetry information of orig from spglib
        self.log.info("    Tolerance: {}".format(self.tol))
        spacegroup = spg.get_spacegroup(atoms_to_spglib_cell(orig),
                                        symprec=self.tol)
        self.log.info("\n  The pristine structure belongs to spacegroup: "
                      "{}".format(spacegroup))
        spg_tuple = atoms_to_spglib_cell(orig)
        return spg.get_symmetry(spg_tuple, symprec=self.tol)

    def get_all_equivalent_images(self, atoms):
        """Get a list of all symmetry equivalent structures.

        All symmetry operations are applied to the provided
        structure. This creates all symmetry equivalent structures,
        which are returned.

        Args:
            atoms (Atoms): The structure with a defect.

        Returns:
            list:
                A list of symmetry equivalent structures.
        """

        equiv_structs = []
        spos = atoms.get_scaled_positions()
        cell = atoms.cell
        for R, T in zip(self.all_sym_ops['rotations'],
                        self.all_sym_ops['translations']):
            rot_spos = np.inner(spos, R) + T
            rot_pos = cell.cartesian_positions(rot_spos)
            pos = wrap_positions(rot_pos, cell)
            ac = atoms.copy()
            ac.positions = pos
            reshuffle_positions(atoms, ac)
            equiv_structs.append(ac)
        return equiv_structs

    def get_final_image(self, initial, initial_relaxed, final,
                        log_atomic_idx=False, return_ops=False,
                        rot_only=False):
        """Obtain a relaxed final structure from the initial relaxed.

        First translations then subsequently rotations operations are checked,
        when a valid symmetry operation is found the final image is created.

        Args:
            initial (Atoms): The initial structure (unrelaxed)
            initial_relaxed (Atoms): The relaxed version of initial
            final (Atoms): The final structure (unrelaxed)
            log_atomic_idx (bool): Specify if equivalent indices in initial and
                final should be logged.
            return_ops (bool): Return the symmetry operations used for
                creating the final relaxed structure.
            rot_only (bool): Disregard pure translation operations.

        Returns:
            Atoms:
                The final structure (relaxed)
        """
        self.log.info("Creating final image:")
        self.log.info("  Input parameters:")
        self.log.info("    Tolerance: {}".format(self.tol))

        # wrap atoms outside unit cell into the unit cell
        initial.wrap(eps=self.tol)
        final.wrap(eps=self.tol)

        if not rot_only:
            # check for simple translations
            index_swaps = self.find_translations(initial, final)

            # if translations were found use them
            if index_swaps is not None:
                # create the final relaxed image
                final_relaxed = get_relaxed_final(initial, initial_relaxed,
                                                  final,
                                                  trans=index_swaps)
                if return_ops:
                    return [final_relaxed, index_swaps]
                else:
                    return final_relaxed

        # if no translations were found, look for rotations
        all_sym_ops = self.find_symmetries(initial, final,
                                           log_atomic_idx=log_atomic_idx)

        # Check if the rot matrix is the identity matrix and rot_only
        # is True. Then that is a pure translation.
        if rot_only:
            all_sym_ops = _purge_pure_tranlation_ops(all_sym_ops)
        final_relaxed = get_relaxed_final(initial, initial_relaxed,
                                          final, rot=all_sym_ops)
        if return_ops:
            return [final_relaxed, all_sym_ops]
        else:
            return final_relaxed

    def find_symmetries(self, initial, final,
                        log_atomic_idx=False):
        """Find the relevant symmetry operations relating initial and final.

        Args:
            initial (Atoms): Initial structure in the NEB path
            final (Atoms): Final structure in the NEB path
            log_atomic_idx (bool): Specify if equivalent indices in initial and
                final should be logged.

        Returns:
            list:
                The valid symmetry operations mapping initial onto
                final. The returned list has the following form:
                [[R, T, eq_idx], ...], where R is a rotation matrix, T is the
                corresponding translation and eq_idx is a list of indices that
                maps initial onto final.

        """
        self.log.info("\n  Looking for rotations:")
        initial_temp = initial.copy()
        final_temp = final.copy()
        initial_temp.wrap()
        final_temp.wrap()
        pos = initial_temp.get_scaled_positions()
        pos_final = final_temp.get_positions()
        cell = initial_temp.get_cell()
        sym = self.all_sym_ops

        R = sym['rotations']
        T = sym['translations']
        sym = []

        # Apply symmetry operations to pos_initial and test if the
        # positions match pos_final
        for i, r in enumerate(R):
            pos_initial_scaled = np.inner(pos, r) + T[i]
            pos_initial_cartesian = cell.cartesian_positions(
                pos_initial_scaled)
            pos_initial = wrap_positions(pos_initial_cartesian, cell)
            res = compare_positions(pos_final, pos_initial, cell, tol=self.tol)
            if res[0]:
                sym.append([r.astype(int), T[i], res[1]])

        # Write to log
        if len(sym) > 0:
            for i, S in enumerate(sym):
                U = S[0]
                T = S[1]
                self.log.info("\n      Symmetry {}:".format(i))
                self._write_3x3_matrix_to_log(U, i)
                self.log.info("        T: {:2.2f} {:2.2f} {:2.2f}"
                              .format(T[0], T[1], T[2]))
                if log_atomic_idx:
                    idxs = S[2]
                    self.log.info("        Equivalent atoms:")
                    for j, idx in enumerate(idxs):
                        self.log.info("          {} -> {}".format(idx, j))
        else:
            self.log.warning("    No rotations found")

        return sym

    def reflect_path(self, images, sym):
        """Get the reflection operations valid for the entire path.

        Args:
            images (list of Atoms): The path as a list of Atoms
                objects. This is typically the output of
                :func:`ase.neb.interpolate`.
            sym (list of symmetry operations): A list of symmetry operations
                to test for their validity as reflection operators for the
                path. They are typically obtained from :func:`find_symmetries`.

        Returns:
            list:
                Valid reflection operations for the path. The returned
                list has the following form: [[R, T, eq_idx], ...], where R is
                a rotation matrix, T is the corresponding translation and
                eq_idx is a list of indices that maps the image on the first
                part of the path with its symmetric image in the final part.
        """
        self.log.info("Get valid reflections of the path:")
        # Go through each pair of images
        n = len(images)
        n_half = int(np.ceil(n / 2))
        path_flat = []
        for i, im in enumerate(images[:n_half - 1]):
            self.log.info("   Matching vector i_{} - i_{} and i_{} - i_{}:"
                          .format(i + 1, i, n - i - 2, n - i - 1))

            # Building vector from image i to images i + 1
            pos_ip1 = images[i + 1].get_scaled_positions()
            pos_i = images[i].get_scaled_positions()
            vecf = []
            for x, p in enumerate(pos_ip1):
                d = get_distances([p], [pos_i[x]],
                                  cell=np.identity(3),
                                  pbc=np.array([1, 1, 1]))
                vecf.append(d[0][0][0])

            # Building vector from image n - i - 1 to n - i - 2
            pos_nim2 = images[n - i - 2].get_scaled_positions()
            pos_nim1 = images[n - i - 1].get_scaled_positions()
            vecb = []
            for x, p in enumerate(pos_nim2):
                d = get_distances([p], [pos_nim1[x]],
                                  cell=np.identity(3),
                                  pbc=np.array([1, 1, 1]))
                vecb.append(d[0][0][0])

            # Check which symmetry operation map vecf onto vecb
            sym = self._get_valid_reflections(vecf, vecb, sym=sym)
            if len(sym) == 0:
                return sym
            else:
                self.log.info("      Found {} valid reflections:"
                              .format(len(sym)))
                for j, S in enumerate(sym):
                    self._write_3x3_matrix_to_log(S[0], j)
                    S_flat = np.concatenate((S[0], S[1], S[2]), axis=None)
                    path_flat.append(S_flat)

        # Use only symmetry operations valid for all image pairs
        sym_flat, counts = np.unique(path_flat, axis=0, return_counts=True)
        sym = []
        for S in sym_flat[np.where(counts == n_half - 1)]:
            U = np.reshape(S[:9], (3, 3)).astype(int)
            T = S[9:12]
            idx = S[12:]
            sym.append([U, T, idx.astype(int)])

        self.log.info("\n  Found {} valid reflections valid for all of the"
                      " path:".format(len(sym)))
        for i, S in enumerate(sym):
            self._write_3x3_matrix_to_log(S[0], i)
        return sym

    def get_reflective_path(self, images, sym_ops):
        refl_sym_ops = self.reflect_path(images, sym_ops)
        if len(refl_sym_ops) == 0:
            return images
        return ReflectiveImages(refl_sym_ops[0], images)

    def find_translations(self, initial, final,
                          return_translation_vec=False):
        """Find a translation that maps initial onto final.

        Args:
            initial (Atoms): The initial structure
            final (Atoms): The final structure
            return_translation_vec (bool): If True the translation vector is
                returned. Default: False.

        Returns:
            list:
                List of indices that maps initial onto final.
            *or*
            np.array:
                Vector that translates initial to final.
        """
        self.log.info("\n  Looking for translations:")
        initial_temp = initial.copy()
        # get the equivalent atoms information
        equiv_atoms = self.all_sym_ops['equivalent_atoms']

        # get unique elements in equiv_atoms to be used as reference for
        # translations
        unq = np.unique(equiv_atoms)

        pos_initial = initial_temp.get_positions()
        pos_final = final.get_positions()
        pos_sc = self.orig.get_positions()
        cell = initial_temp.get_cell()
        for u in unq:
            equiv_list = np.where(equiv_atoms == u)[0]
            for eq in equiv_list:
                dpos = pos_sc[eq] - pos_sc[u]
                initial_temp.set_positions(pos_initial + dpos)
                initial_temp.wrap(eps=self.tol)
                pos = initial_temp.get_positions()
                res, matches = compare_positions(pos_final,
                                                 pos, cell, tol=self.tol)
                if res:
                    self.log.info("    {} -> {} match found!"
                                  .format(u, eq))
                    self.log.info("    T: {:2.2f} {:2.2f} {:2.2f}"
                                  .format(dpos[0], dpos[1], dpos[2]))
                    if return_translation_vec:
                        return dpos
                    return matches
                else:
                    self.log.warning("    {} -> {} no match".format(u, eq))
        self.log.warning("    No translations found")
        return None

    def _get_valid_reflections(self, vec_initial, vec_final, sym):
        reflections = []
        for S in sym:
            U = S[0]
            idx = S[2]
            if is_reflect_op(U):
                if np.allclose(np.inner(vec_initial, U)[idx],
                               vec_final, atol=2 * self.tol):
                    reflections.append(S)
        return reflections

    def _setup_log(self, logfile):
        # Define a per-instance log
        log = logging.getLogger('{}.{}'.format(__name__, id(self)))
        if logfile is None:
            f_handler = logging.NullHandler()
        elif logfile == '-':
            f_handler = logging.StreamHandler()
        else:
            f_handler = logging.FileHandler(logfile, 'a')
        f_format = logging.Formatter('%(message)s')
        f_handler.setFormatter(f_format)
        log.addHandler(f_handler)
        log.setLevel(logging.INFO)
        return log

    def _write_3x3_matrix_to_log(self, x, i=0):
        self.log.info("\n        U_{}:".format(i))
        self.log.info("          {:2d} {:2d} {:2d}"
                      .format(x[0][0], x[0][1], x[0][2]))
        self.log.info("          {:2d} {:2d} {:2d}"
                      .format(x[1][0], x[1][1], x[1][2]))
        self.log.info("          {:2d} {:2d} {:2d}"
                      .format(x[2][0], x[2][1], x[2][2]))


class ReflectiveImages(list):
    def __init__(self, reflect_ops, iterable=()):
        super(ReflectiveImages, self).__init__(iterable)
        self.reflect_ops = reflect_ops

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[j] for j in range(*i.indices(len(self)))]
        atoms = super(ReflectiveImages, self).__getitem__(i)
        if i <= len(self) // 2:
            return atoms
        else:
            # i is over half way of the path, j is the corresponding
            # reflective image from the first half of the path
            j = i - (i - len(self) // 2) * 2

            results = get_results_dict(self[j])

            # Use symmetry operation on forces in scaled coordinates
            cell = self[j].cell
            sforces = cell.scaled_positions(results['forces'])
            rot_forces = np.inner(sforces[self.reflect_ops[2]],
                                  self.reflect_ops[0])

            # Set the rotated forces in the results dictionary. Energy
            # and magmoms are identical to the symmetric image
            results['forces'] = cell.cartesian_positions(rot_forces)
            newcalc = SinglePointCalculator(atoms=atoms, **results)
            atoms.calc = newcalc

            return atoms

    def __eq__(self, other):
        if not isinstance(other, ReflectiveImages):
            return False
        return super(ReflectiveImages, self).__eq__(other)

    def __ne__(self, other):
        if isinstance(other, ReflectiveImages):
            return super(ReflectiveImages, self).__eq__(other)
        return True


def get_relaxed_final(initial, initial_relaxed, final,
                      trans=None, rot=None):
    """Obtain a relaxed final structure from a relaxed initial structure.

    Supply symmetry operations **either** pure translations by index
    swapping in ``trans`` **or** rotations in ``rot``.

    Args:
        initial (Atoms): The initial structure (unrelaxed)
        initial_relaxed (Atoms): A relaxed version of initial
        final (Atoms): The final structure (unrelaxed)
        trans (list): Index swaps that signifies a translation symmetry
            operation. Obtained from :func:`find_symmetries`.
        rot (list): Symmetry operation as returned from
            :func:`find_symmetries`.

    Returns:
        Atoms:
            The final structure in the relaxed geometry.

    """
    if trans is not None and rot is not None:
        msg = ('Cannot specify both trans and rot. '
               'Got {} and {} respectively.').format(trans, rot)
        raise ValueError(msg)
    elif trans is None and rot is None:
        msg = 'Must specify either \'trans\' or \'rot\''
        raise ValueError(msg)
    elif rot is not None:
        # Apply rotational operator
        symop = rot[0][2]

        def f(x):
            # return np.dot(rot[0][0], x)
            return np.inner(x, rot[0][0])
    else:
        # Apply translational operator
        symop = trans

        def f(x):
            return x

    final_temp = final.copy()
    dpos = initial_relaxed.get_positions() - initial.get_positions()
    cell = initial.get_cell()
    dpos = find_mic(dpos, cell)[0]
    forces_rotated = None

    initial_results = get_results_dict(initial_relaxed)

    if initial_results['forces'] is not None:
        forces_rotated = np.zeros((len(dpos), 3))
    dpos_rotated = np.zeros((len(dpos), 3))
    magmom_rotated = np.zeros(len(dpos))

    dpos_rotated = perform_symmetry_operation(dpos[symop], cell, f)
    magmom_rotated = initial_results['magmoms'][symop]
    if initial_results['forces'] is not None:
        tmp_forces = initial_results['forces'][symop]
        forces_rotated = perform_symmetry_operation(tmp_forces, cell, f)

    # for i, at in enumerate(symop):
    #     sdpos = cell.scaled_positions(dpos[at])
    #     dpos_rotated[i] = cell.cartesian_positions(f(sdpos))
    #     # dpos_rotated[i] = f(dpos[at])
    #     magmom_rotated[i] = initial_results['magmoms'][at]
    #     if initial_results['forces'] is not None:
    #         # Why don't we use scaled forces here?
    #         sforces = cell.scaled_positions(initial_results['forces'][at])
    #         forces_rotated[i] = cell.cartesian_positions(f(sforces))
    #         # forces_rotated[i] = f(initial_results['forces'][at])

    results = {'forces': forces_rotated,
               'energy': initial_results['energy'],
               'magmoms': magmom_rotated}

    final_temp.set_initial_magnetic_moments(magmom_rotated)

    pos = final_temp.get_positions()
    final_temp.set_positions(pos + dpos_rotated)
    newcalc = SinglePointCalculator(final_temp, **results)
    final_temp.calc = newcalc

    return final_temp


def get_results_dict(atoms):
    results = {'energy': None, 'forces': None,
               'magmoms': np.zeros(len(atoms))}
    if atoms.calc:
        for prop in results.keys():
            try:
                res = atoms.calc.get_property(prop,
                                              allow_calculation=False)
            except PropertyNotImplementedError:
                res = None
            if res is not None:
                results[prop] = res
    return results


def perform_symmetry_operation(vectors, cell, sym_func):
    scaled_vectors = cell.scaled_positions(vectors)
    return cell.cartesian_positions(sym_func(scaled_vectors))


def _purge_pure_tranlation_ops(sym_ops):
    purged_ops = []
    for op in sym_ops:
        if not np.array_equal(op[0], np.identity(3)):
            purged_ops.append(op)
    return purged_ops


def compare_positions(pos1, pos2, cell, tol=1e-3):
    """Check whether two arrays contain the same positions.

    The positions need not be in the same order. A mapping for pos1 to pos2 is
    also returned.

    Args:
        pos1, pos2 (list or np.array): atomic positions of structures to
            be compared.

    Returns:
        (boolean, matches):
            True for a match and a list with the matches
            matches: i is the index an atom in structure 1 and at index i in
            matches is the index, j, of the corresponding atom in
            structure 2.
    """
    n = len(pos2)
    dists_all = get_distances(pos2, pos1, cell=cell, pbc=[1, 1, 1])
    final_to_initial = np.nonzero(np.isclose(dists_all[1], 0,
                                             atol=tol))[1]
    match = len(final_to_initial) == n
    return match, np.argsort(final_to_initial)


def is_reflect_op(R):
    """Check whether a matrix R is a reflection operation.

    Three types of reflections are possible: Reflection with respect
    to a plane, line or point.

    """
    # First test if the matrix is involutory (i.e. its own inverse)
    Q = R.T @ R
    if np.array_equal(Q, np.identity(3)):
        # Then check the eigenvalues for the type of reflection
        eig_values = np.sort(np.linalg.eig(R)[0])
        plane = np.array([-1, 1, 1])
        line = np.array([-1, -1, 1])
        point = np.array([-1, -1, -1])
        if np.array_equal(eig_values, plane):
            return True
        elif np.array_equal(eig_values, line):
            return True
        elif np.array_equal(eig_values, point):
            return True
    return False


def reshuffle_positions(initial, final, min_neb_path_length=1.2):
    """Oh no! Vacancies!

    You want to calculate the NEB barrier between two vacancies, but
    by creating vacancies the indices have been shuffled so everything
    seems to move when creating the initial NEB path. This function
    shuffles them back by checking distances between individual atoms
    in the initial and final structures. It is required that the unit
    cells are equal.

    Args:
        initial (Atoms): Initial structure
        final (Atoms): Final structure. The atoms in this structure
            are rearranged to fit with initial.
        min_neb_path_length (float): Atoms that are detected to have moved
            more than min_neb_path_length are considered to be a part of the
            desired NEB path. Default: 1.2 Å

    Returns:
        Atoms:
            Final structure as an Atoms object with shuffled positions.

    """
    final_s = final.copy()
    ml = min_neb_path_length

    cell = initial.cell
    assert np.array_equal(cell, final.cell), 'Unit cells are not equal!'

    pos1 = initial.positions
    pos2 = final_s.positions.copy()

    _, D = get_distances(pos1, pos2, cell=cell, pbc=True)
    final_idx = []
    for c, j in enumerate(np.argmin(D, axis=0)):
        if j != c:
            # Shuffled indices we need to shuffle back
            if D[j, c] < ml:  # assuming neb path length is above ml
                # this is not the correct atom moving, i.e. we shuffle
                final_s.positions[j] = pos2[c]
            else:
                final_idx.append(c)
    if len(final_idx) > 0:
        initial_idx = []
        # We need to get the atom in final furthest away from initial as well
        for c, j in enumerate(np.argmin(D, axis=1)):
            if D[c, j] > ml:
                initial_idx.append(c)
        assert len(final_idx) == len(initial_idx)
        final_s.positions[np.array(initial_idx)] = pos2[np.array(final_idx)]

    return final_s
