"""The reflective NEB class."""
import logging

import numpy as np
import spglib as spg

from ase.calculators.calculator import (PropertyNotImplementedError,
                                        PropertyNotPresent)
from ase.calculators.singlepoint import SinglePointCalculator
from ase.geometry import find_mic, get_distances, wrap_positions
from ase.utils import atoms_to_spglib_cell
from ase.io import write


class RNEB:
    """The ASE implementation of the R-NEB method.

    The method is described in:
    Nicolai Rask Mathiesen, Hannes Jónsson, Tejs Vegge,
    and Juan Maria García Lastra, J. Chem. Theory Comput., 2019, 15 (5),
    pp 3215–3222, (doi: 10.1021/acs.jctc.8b01229)

    Symmetry equivalent structures can be identified and produce the final
    relaxed structure given the initial unrelaxed, initial relaxed,
    and final unrelaxed images. (get_final_image)

    Symmetry operators relating the initial and final
    images can be given. (find_symmetries)

    It can further test whether a path is reflection symmetric
    and return the valid reflection. (reflect_path). The obtained symmetry
    operations can be given to the main NEB class to perform


    Parameters:
        tol (float): Tolerance on atomic distances in Å
        logfile (str or None): filename for the log output
            If None then no output is made. Default is '-' for standard
            stream output.

    """

    def __init__(self, tol=1e-3, logfile='-'):
        self.tol = tol  # tolerance on atomic positions
        self.log = self._setup_log(logfile)

    def get_final_image(self, orig, init, init_relaxed, final,
                        supercell=[1, 1, 1],
                        log_atomic_idx=False, return_ops=False,
                        rot_only=False, filename=None):
        """Find the symmetry operations that map init->final.

        First translations then subsequently rotations are checked,
        when a valid symmetry operation is found the final image is created.
        """
        self.log.info("Creating final image:")
        self.log.info("  Input parameters:")
        self.log.info("    Tolerance: {}".format(self.tol))
        spacegroup = spg.get_spacegroup(atoms_to_spglib_cell(orig),
                                        symprec=self.tol)
        self.log.info("\n  The pristine structure belongs to spacegroup: "
                      "{}".format(spacegroup))
        # wrap atoms outside unit cell into the unit cell
        init.wrap(eps=self.tol)
        final.wrap(eps=self.tol)
        # check for simple translations
        trans = self.find_translations(orig, init, final, supercell=supercell)
        # if translations were found
        if trans is not None and not rot_only:
            # create the final relaxed image
            final_relaxed = self.get_relaxed_final(init, init_relaxed,
                                                   final, trans=trans,
                                                   filename=filename)
            if return_ops:
                return [final_relaxed, trans]
            else:
                return final_relaxed
        else:
            # if no translations were found, look for rotations
            rot = self.find_symmetries(orig, init, final,
                                       supercell=supercell,
                                       log_atomic_idx=log_atomic_idx)
            final_relaxed = self.get_relaxed_final(init, init_relaxed,
                                                   final, rot=rot,
                                                   filename=filename)
            if return_ops:
                return [final_relaxed, rot]
            else:
                return final_relaxed

    def find_symmetries(self, orig, init, final, supercell=[1, 1, 1],
                        log_atomic_idx=False):
        """Find the relevant symmetry operations relating init and final.

        Args:
            orig (Atoms): Original structure from which all symmetry
                operations are found using spglib.
            init (Atoms): Initial structure in the NEB path
            final (Atoms): Final structure in the NEB path
            supercell (list): repeat the original structure.
                Default [1, 1, 1] (no repetitions)
            log_atomic_idx (bool): Specify if qquivalent indices in init and
                final should be logged.
        """
        self.log.info("\n  Looking for rotations:")
        init_temp = init.copy()
        final_temp = final.copy()
        init_temp.wrap()
        final_temp.wrap()
        pos = init_temp.get_scaled_positions()
        pos_final = final_temp.get_positions()
        cell = init_temp.get_cell()
        orig_temp = orig.copy()
        orig_temp = orig_temp.repeat(supercell)
        spg_tuple = atoms_to_spglib_cell(orig_temp)
        sym = spg.get_symmetry(spg_tuple, symprec=self.tol)
        R = sym['rotations']
        T = sym['translations']
        pos_temp = np.empty((len(pos), 3))
        S = []

        for i, r in enumerate(R):
            pos_init_scaled = np.inner(pos, r) + T[i]
            pos_init_cartesian = cell.cartesian_positions(pos_init_scaled)
            pos_init = wrap_positions(pos_init_cartesian, cell)
            res = self.compare_translations(pos_final,
                                            pos_init, cell)
            if res[0]:
                S.append([r.astype(int), T[i], res[1]])
        if len(S) > 0:
            for i, s in enumerate(S):
                U = s[0]
                T = s[1]
                self.log.info("\n      Symmetry {}:".format(i))
                self.log.info("        U:")
                self.write_3x3_matrix_to_log(U)
                self.log.info("        T: {:2.2f} {:2.2f} {:2.2f}"
                              .format(T[0], T[1], T[2]))
                if log_atomic_idx:
                    idxs = s[2]
                    self.log.info("        Equivalent atoms:")
                    for j, idx in enumerate(idxs):
                        self.log.info("          {} -> {}".format(idx, j))
        else:
            self.log.warning("    No rotations found")

        return S

    def reflect_path(self, images, sym=None):
        self.log.info("Get valid reflections of the path:")
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
                                  cell=np.array(
                                      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                  pbc=np.array([1, 1, 1]))
                vecf.append(d[0][0][0])
                
            # Building vector from image n - 1 to n - 2
            pos_nim2 = images[n - i - 2].get_scaled_positions()
            pos_nim1 = images[n - i - 1].get_scaled_positions()
            vecb = []
            for x, p in enumerate(pos_nim2):
                d = get_distances([p], [pos_nim1[x]],
                                  cell=np.array(
                                      [[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                  pbc=np.array([1, 1, 1]))
                vecb.append(d[0][0][0])

            sym = self.reflect_movement(vecf, vecb, sym=sym)
            if len(sym) == 0:
                return sym
            else:
                self.log.info("      Found {} valid reflections:"
                              .format(len(sym)))
                for j, S in enumerate(sym):
                    self.log.info("\n        U_{}:".format(j))
                    self.write_3x3_matrix_to_log(S[0])
                    S_flat = np.concatenate((S[0], S[1], S[2]), axis=None)
                    path_flat.append(S_flat)

        sym_flat, counts = np.unique(path_flat, axis=0, return_counts=True)
        sym_flat[np.where(counts == n_half - 1)]
        sym = []
        for S in sym_flat:
            U = np.reshape(S[:9], (3, 3)).astype(int)
            T = S[9:12]
            idx = S[12:]
            sym.append([U, T, idx.astype(int)])

        self.log.info("\n  Found {} valid reflections valid for all of the"
                      " path:".format(len(sym)))
        for i, S in enumerate(sym):
            self.log.info("\n        U_{}:".format(i))
            self.write_3x3_matrix_to_log(S[0])
        return sym

    def get_relaxed_final(self, init, init_relaxed, final,
                          trans=None, rot=None, filename=None):
        """Obtain a relaxed final structure from a relaxed initial structure.

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
                return np.dot(rot[0][0], x)
        else:
            # Apply translational operator
            symop = trans

            def f(x):
                return x

        final_temp = final.copy()
        dpos = init_relaxed.get_positions() - init.get_positions()
        cell = init.get_cell()
        dpos = find_mic(dpos, cell)[0]
        magmoms = np.zeros(len(init_relaxed))
        en = None
        forces = None
        forces_rotated = None
        # Why don't we use scaled forces here
        if init_relaxed.calc:
            try:
                en = init_relaxed.get_potential_energy()
                forces = init_relaxed.get_forces()
                forces_rotated = np.zeros((len(dpos), 3))
                magmoms = init_relaxed.get_magnetic_moments()
            except (PropertyNotImplementedError, PropertyNotPresent):
                pass

        dpos_rotated = np.zeros((len(dpos), 3))
        magmom_rotated = np.zeros(len(dpos))

        for i, at in enumerate(symop):
            dpos_rotated[i] = f(dpos[at])
            magmom_rotated[i] = magmoms[at]
            if forces is not None:
                forces_rotated[i] = f(forces[at])

        results = {'forces': forces_rotated,
                   'energy': en,
                   'magmoms': magmom_rotated}

        final_temp.set_initial_magnetic_moments(magmom_rotated)

        pos = final_temp.get_positions()
        final_temp.set_positions(pos + dpos_rotated)
        newcalc = SinglePointCalculator(final_temp, **results)
        final_temp.calc = newcalc

        if filename:
            write(filename, final_temp)
            self.log.info("    Created final relaxed image as {}"
                          .format(filename))
        return final_temp

    def find_translations(self, orig, init, final, supercell=[1, 1, 1],
                          return_vec=False):
        self.log.info("\n  Looking for translations:")
        init_temp = init.copy()
        orig_super_cell = orig.repeat(supercell)
        # get the symmetry information for the pristine structure
        spglib_tuple = atoms_to_spglib_cell(orig_super_cell)
        symmetry_super_cell = spg.get_symmetry(spglib_tuple,
                                               symprec=self.tol)
        equiv_atoms = symmetry_super_cell['equivalent_atoms']
        # get unique elements in equiv_atoms to be used as reference for
        # translations
        unq = np.unique(equiv_atoms)
        pos_init = init_temp.get_positions()
        pos_final = final.get_positions()
        pos_sc = orig_super_cell.get_positions()
        cell = init_temp.get_cell()
        for u in unq:
            equiv_list = np.where(equiv_atoms == u)[0]
            for eq in equiv_list:
                dpos = pos_sc[eq] - pos_sc[u]
                init_temp.set_positions(pos_init + dpos)
                init_temp.wrap(eps=self.tol)
                pos = init_temp.get_positions()
                res, matches = self.compare_translations(pos_final,
                                                         pos, cell)
                if res:
                    self.log.info("    {} -> {} match found!"
                                  .format(u, eq))
                    self.log.info("    T: {:2.2f} {:2.2f} {:2.2f}"
                                  .format(dpos[0], dpos[1], dpos[2]))
                    if return_vec:
                        return dpos
                    return matches
                else:
                    self.log.warning("    {} -> {} no match".format(u, eq))
        self.log.warning("    No translations found")
        return None

    def compare_translations(self, pos_final, pos, cell):
        """pos, pos_final: atomic positions of structures to be compared

        returns: [boolean, matches]
            True for a match and a list with the matches
            matches: i is the index an atom in structure 1 and at index i in
            matches is the index, j, of the corresponding atom in
            structure 2.
        """
        n = len(pos)
        dists_all = get_distances(pos, pos_final, cell=cell, pbc=[1, 1, 1])
        final_to_init = np.nonzero(np.isclose(dists_all[1], 0,
                                              atol=self.tol))[1]
        if len(final_to_init) == n:
            return [True, np.argsort(final_to_init)]
        return [False, np.argsort(final_to_init)]

    def reflect_movement(self, vec_init, vec_final, sym=None):
        reflections = []
        for S in sym:
            U = S[0]
            idx = S[2]
            vec_init_temp = np.empty((len(vec_init), 3))
            if is_reflect_op(U):
                for at2, at in enumerate(idx):
                    vec_init_temp[at2] = np.dot(U, vec_init[at])
                if np.allclose(vec_init_temp, vec_final, atol=2 * self.tol):
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

    def write_3x3_matrix_to_log(self, x):
        self.log.info("          {:2d} {:2d} {:2d}"
                      .format(x[0][0], x[0][1], x[0][2]))
        self.log.info("          {:2d} {:2d} {:2d}"
                      .format(x[1][0], x[1][1], x[1][2]))
        self.log.info("          {:2d} {:2d} {:2d}"
                      .format(x[2][0], x[2][1], x[2][2]))


def is_reflect_op(R):
    I = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    Q = np.dot(np.transpose(R), R)
    if np.array_equal(Q, I):
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


def reshuffle_positions(init, final, min_neb_path_length=1.2):
    """Oh no! Vacancies!

    You want to calculate the NEB barrier between two vacancies, but
    by creating vacancies the indices have been shuffled so everything
    seems to move when creating the initial NEB path. This function
    shuffles them back by checking distances between individual atoms
    in the initial and final structures.

    Parameters
    ----------
    init: Atoms object
        Initial structure

    final: Atoms object
        Final structure. The atoms in this structure
        are rearranged to fit with init.

    min_neb_path_length: float
        Atoms that are detected to have moved more than
        min_neb_path_length are considered to be a part of the desired
        NEB path.
        Default: 1.2 Å

    Returns
    -------
    Final structure as an Atoms object with shuffled positions.


    """
    final_s = final.copy()
    ml = min_neb_path_length
    cell = init.cell
    p1 = init.positions
    p2 = final_s.positions.copy()

    _, D = get_distances(p1, p2, cell=cell, pbc=True)
    final_idx = []
    for c, j in enumerate(np.argmin(D, axis=0)):
        if j != c:
            # Shuffled indices we need to shuffle back
            if D[j, c] < ml:  # assuming neb path length is above ml
                # this is not the correct atom moving, i.e. we shuffle
                final_s.positions[j] = p2[c]
            else:
                final_idx.append(c)
    if len(final_idx) > 0:
        init_idx = []
        # We need to get the atom in final furthest away from init as well
        for c, j in enumerate(np.argmin(D, axis=1)):
            if D[c, j] > ml:
                init_idx.append(c)
        assert len(final_idx) == len(init_idx)
        final_s.positions[np.array(init_idx)] = p2[np.array(final_idx)]

    return final_s
