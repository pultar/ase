""" Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions
    using the spread functional of Marzari and Vanderbilt
    (PRB 56, 1997 page 12847).
"""
from time import time
from math import sqrt, pi
from pickle import dump, load

import numpy as np

from ase.parallel import paropen
from ase.dft.kpoints import get_monkhorst_pack_size_and_offset
from ase.transport.tools import dagger, normalize

dag = dagger


def gram_schmidt(U):
    """Orthonormalize columns of U according to the Gram-Schmidt procedure."""
    for i, col in enumerate(U.T):
        for col2 in U.T[:i]:
            col -= col2 * np.dot(col2.conj(), col)
        col /= np.linalg.norm(col)


def lowdin(U, S=None):
    """Orthonormalize columns of U according to the Lowdin procedure.

    If the overlap matrix is know, it can be specified in S.
    """
    if S is None:
        S = np.dot(dag(U), U)
    eig, rot = np.linalg.eigh(S)
    rot = np.dot(rot / np.sqrt(eig), dag(rot))
    U[:] = np.dot(U, rot)


def search_shells(g_max, basis_c, mpgrid):
    """Compute possible combinations of vectors

       The returned array should represent directions from one k-point to every
       possible neighboring k-point, we compute this in cartesian coordinates
       as well and we evaluate the distance.

       Given a g_max integer delimiting the range, it computes every possible
       combination, with repetition. The only exception is (0,0,0) that is
       removed from the list. All the arrays are sorted in order of increasing
       distance.

       EX: g_max = 1  -> (-1,-1,-1), (-1,1,0), (1,0,-1), (0,0,1), etc.

       g_max: max value delimiting the range to use for the combinations
       basis_c: list of basis vectors (should contain three 3D vectors)
       mpgrid: Monkhorst-Pack grid size (Nx, Ny, Nz)
    """
    g_range = np.arange(- g_max, g_max + 1, dtype=int)
    comb = np.stack(np.meshgrid(g_range, g_range, g_range),
                    axis=-1).reshape(-1, len(mpgrid))
    comb = comb[np.any(comb != [0, 0, 0], axis=1)]

    # Compute cartesian coordinates and distance from the center
    b_cart_pc = np.empty((len(comb), len(basis_c)), dtype=float)
    b_dist_p = np.empty(len(comb), dtype=float)
    for i, p in enumerate(comb):
        b_cart_pc[i] = 2 * np.pi * p @ basis_c / mpgrid
        b_dist_p[i] = np.linalg.norm(b_cart_pc[i])
    sort = np.argsort(b_dist_p)
    comb = comb[sort]
    b_cart_pc = b_cart_pc[sort]
    b_dist_p = b_dist_p[sort]
    return comb, b_cart_pc, b_dist_p


def count_shells(dist_p, atol):
    """Count the equal-distance shells and the elements in each

       dist_p: sorted list of distances
       atol: absolute tolerance for equality
    """
    neighbors_per_shell = [1]
    prev_d = 0.
    for i, d in enumerate(dist_p):
        if i > 0:
            if np.abs(prev_d - d) < atol:
                neighbors_per_shell[-1] = neighbors_per_shell[-1] + 1
            else:
                neighbors_per_shell.append(1)
        prev_d = d
    neighbors_per_shell = np.array(neighbors_per_shell, dtype=int)
    return neighbors_per_shell


def check_parallel(vector1, vector2, atol=1e-6):
    """Check if two vectors are parallel to each other

       atol: absolute tolerance for equality
    """
    inn_prod = np.dot(vector1, vector2)
    norm_prod = (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return np.abs(inn_prod - norm_prod) < atol


def neighbors_complete_set(vectors_sbc, vectors_per_shell,
                           atol=1e-6, verbose=False):
    """Returns the list of shells and weights for a complete set of neighbors

       Returns the indices of vector_per_shell that correspond to a complete
       set of neighbors, i.e. the weights and the 'b' vectors that satisfy
       the following completeness equation:

       sum_s(w_s * sum_b(bs_i * bs_j)) = delta(i, j)

       Ref: https://doi.org/10.1016/j.cpc.2007.11.016

       vectors_sbc: list of 'b' vectors in each shell, in cartesian coordinates
       vectors_per_shell: list of number of 'b' vectors in each shell
       atol: absolute tolerance for (in)equality
    """
    shells_to_keep = np.arange(0, len(vectors_per_shell)).astype(int)
    rm = 0
    for n in range(0, len(vectors_per_shell)):
        # update index, some shells have been removed from the list
        n -= rm

        # Reject shell if two vectors are parallel, usually this also causes
        # small singular values, but sometimes the two are not correlated
        tot = 0
        parallel = False
        for s, Nb in enumerate(vectors_per_shell[:n]):
            for b_out in range(Nb):
                for b_in in range(vectors_per_shell[n]):
                    if parallel:
                        # we already found one, skip the rest
                        continue
                    if check_parallel(vectors_sbc[shells_to_keep][s, b_out],
                                      vectors_sbc[shells_to_keep][n, b_in],
                                      atol):
                        parallel = True
            tot += Nb

        if parallel:
            shells_to_keep = np.delete(shells_to_keep, n, axis=0)
            vectors_per_shell = np.delete(vectors_per_shell, n, axis=0)
            rm += 1
            if verbose:
                print('Found a vector parallel to a previous shell,',
                      'skipping this shell.')
            continue

        # Express completeness relation in a computationally convenient way
        # (Ad @ w_s = qd) and solve the equation with SVD, same ref. as above
        qd = np.array([1, 1, 1, 0, 0, 0], dtype=float)
        Ad = np.zeros((6, len(vectors_per_shell[:n + 1])), dtype=float)
        for s, Nb in enumerate(vectors_per_shell[:n + 1]):
            for b in range(Nb):
                Ad[0:3, s] = (Ad[0:3, s] +
                              vectors_sbc[shells_to_keep][s, b, 0:3] *
                              vectors_sbc[shells_to_keep][s, b, 0:3])
                Ad[3:6, s] = (Ad[3:6, s] +
                              vectors_sbc[shells_to_keep][s, b, 0:3] *
                              vectors_sbc[shells_to_keep][s, b, [1, 2, 0]])
        Ud, sv, Vt = np.linalg.svd(Ad, full_matrices=False)

        if (np.abs(sv) < atol).any():
            shells_to_keep = np.delete(shells_to_keep, n, axis=0)
            vectors_per_shell = np.delete(vectors_per_shell, n, axis=0)
            rm += 1
            if verbose:
                print('Found a very small singular value,',
                      'skipping this shell.')
            continue

        weights_per_shell = Vt.T @ np.diag(1 / sv) @ Ud.T @ qd

        # Manually check if the relation is satisfied
        if not (np.abs(Ad @ weights_per_shell - qd) < atol).all():
            if verbose:
                print('Completeness relation not satisfied with',
                      n + 1, 'shell(s)')
        else:
            if verbose:
                print('Completeness relation satisfied with',
                      n + 1, 'shell(s)')
                print('Weights:', weights_per_shell)
            number_of_shells = n + 1
            shells_to_keep = shells_to_keep[:number_of_shells]
            break

    return shells_to_keep, weights_per_shell


def compute_neighbors_and_G_vectors(kpt_frac, dir_per_shell,
                                    neighbors_per_shell,
                                    possible_G_vectors, atol=1e-6):
    """ Compute the list of neighbors and the G vectors

        The relation to satisfy is:  nnk = k + b + G
        nnk: neighbor
        k: original k-point
        b: vector connecting to neighbor
        G: vector to jump across PBC (3D vectors of int)

        Arguments:
        kpt_frac: list of k-points in fractional coordinates
        dir_per_shell: possible directions to neighbors per each shell
        neighbors_per_shell: obvious
        b_mill_pc: list of possible G vectors to try
        atol: absolute tolerance for equalities
    """
    nnk_kb = - np.ones((len(kpt_frac), sum(neighbors_per_shell)), dtype=int)
    G_kbc = np.empty((len(kpt_frac), sum(neighbors_per_shell), 3), dtype=int)
    for i, k in enumerate(kpt_frac):
        tot = 0
        for s in range(len(neighbors_per_shell)):
            for j in range(neighbors_per_shell[s]):
                b = dir_per_shell[s, j]
                diff = [d.all() for d in
                        np.abs(kpt_frac - k - b) < atol]
                if True in diff:
                    nnk_kb[i, tot + j] = diff.index(True)
                    G_kbc[i, tot + j] = [0, 0, 0]
                else:
                    for G in possible_G_vectors:
                        diff = [d.all() for d in
                                np.abs(kpt_frac - G - k - b) < atol]
                        if True in diff:
                            nnk_kb[i, tot + j] = diff.index(True)
                            G_kbc[i, tot + j] = G
                            break
            tot += neighbors_per_shell[s]
    if (nnk_kb == -1).any():
        raise ValueError('Unassigned data in list of neighbors.')
    return nnk_kb, G_kbc


def neighbors_and_weights(kpt_frac, recip_cell, atol=1e-6, verbose=False):
    """Compute nearest neighbors, weights and G vectors

       Returns a list of neighbors for each k-point, the G vectors to move
       across PBC and the weights.

       Relation with neighbors: nnk = k + b + G

       Reference: Wannier90 review
                  (http://dx.doi.org/10.1016/j.cpc.2007.11.016)
                  Note: the G vector here has the opposite sign

       kpt_frac:  list of k-points in a MP uniform grid in fractional coords
       recip_cell: reciprocal lattice vectors
       atol:    absolute tolerance for (in)equalities
    """
    # In the following code there are several vectors with Miller (mill)
    # notation, fractional coordinates relative to the base vectors (frac)
    # and cartesian coordinates (cart).

    # NB: even if the arrays have the size, for each shell there is a varying
    #     number of neighbors and vectors pointing to them, this number is
    #     saved in neighbors_per_shell.

    if verbose:
        t = - time()

    mpgrid = get_monkhorst_pack_size_and_offset(kpt_frac)[0]

    # Check all common directions and list distances
    g_max = 1
    _, _, b_dist_p = search_shells(g_max, recip_cell, mpgrid)

    # Increase the range of search if the distances in some directions are
    # smaller, it should prevent problems with elongated cells
    g_max = np.ceil(max(b_dist_p) / min(b_dist_p))
    b_mill_pc, b_cart_pc, b_dist_p = search_shells(g_max, recip_cell, mpgrid)

    # Count shells of neighbors and elements in each
    neighbors_per_shell = count_shells(b_dist_p, atol)

    # Split arrays based on shells
    b_mill_sbc = np.empty((len(neighbors_per_shell),
                           max(neighbors_per_shell), 3), dtype=int)
    b_cart_sbc = np.empty((len(neighbors_per_shell),
                           max(neighbors_per_shell), 3), dtype=float)
    b_dist_s = np.empty(len(neighbors_per_shell), dtype=float)
    tot = 0
    for s, Nb in enumerate(neighbors_per_shell):
        b_mill_sbc[s, :Nb] = b_mill_pc[tot:tot + Nb]
        b_cart_sbc[s, :Nb] = b_cart_pc[tot:tot + Nb]
        b_dist_s[s] = b_dist_p[tot]
        tot += Nb

    # Compute weights and number of shells to satisfy the completeness relation
    shells_to_keep, weights_per_shell = \
        neighbors_complete_set(b_cart_sbc,
                               neighbors_per_shell,
                               atol, verbose)
    b_mill_sbc = b_mill_sbc[shells_to_keep]
    neighbors_per_shell = neighbors_per_shell[shells_to_keep]

    if verbose:
        print('Number of neighbors in each shell:', neighbors_per_shell)
        print('Distances:', b_dist_s[shells_to_keep])

    b_frac_sbc = b_mill_sbc.copy().astype(float)
    b_frac_sbc[:, :] /= mpgrid

    # Compute the list of neighbors and the G vectors
    nnk_kb, G_kbc = compute_neighbors_and_G_vectors(kpt_frac, b_frac_sbc,
                                                    neighbors_per_shell,
                                                    b_mill_pc, atol)
    weights_per_neighbor = np.repeat(weights_per_shell, neighbors_per_shell)

    if verbose:
        t += time()
        print(f'Neighbors and weights computed in {t:.1f}s')

    return nnk_kb, G_kbc, weights_per_neighbor


def random_orthogonal_matrix(dim, rng=np.random, real=False):
    """Generate a random orthogonal matrix"""

    H = rng.rand(dim, dim)
    np.add(dag(H), H, H)
    np.multiply(.5, H, H)

    if real:
        gram_schmidt(H)
        return H
    else:
        val, vec = np.linalg.eig(H)
        return np.dot(vec * np.exp(1.j * val), dag(vec))


def steepest_descent(func, step=.005, tolerance=1e-6, verbose=False, **kwargs):
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        func.step(dF * step, **kwargs)
        fvalue = func.get_functional_value()
        count += 1
        if verbose:
            print('SteepestDescent: iter=%s, value=%s' % (count, fvalue))


def md_min(func, step=.25, tolerance=1e-6, verbose=False, **kwargs):
    if verbose:
        print('Localize with step =', step, 'and tolerance =', tolerance)
        t = -time()
    fvalueold = 0.
    fvalue = fvalueold + 10
    count = 0
    V = np.zeros(func.get_gradients().shape, dtype=complex)
    while abs((fvalue - fvalueold) / fvalue) > tolerance:
        fvalueold = fvalue
        dF = func.get_gradients()
        V *= (dF * V.conj()).real > 0
        V += step * dF
        func.step(V, **kwargs)
        fvalue = func.get_functional_value()
        if fvalue < fvalueold:
            step *= 0.5
        count += 1
        if verbose:
            print('MDmin: iter=%s, step=%s, value=%s' % (count, step, fvalue))
    if verbose:
        t += time()
        print('%d iterations in %0.2f seconds (%0.2f ms/iter), endstep = %s' % (
            count, t, t * 1000. / count, step))


def rotation_from_projection(proj_nw, fixed, ortho=True):
    """Determine rotation and coefficient matrices from projections

    proj_nw = <psi_n|p_w>
    psi_n: eigenstates
    p_w: localized function

    Nb (n) = Number of bands
    Nw (w) = Number of wannier functions
    M  (f) = Number of fixed states
    L  (l) = Number of extra degrees of freedom
    U  (u) = Number of non-fixed states
    """

    Nb, Nw = proj_nw.shape
    M = fixed
    L = Nw - M

    U_ww = np.empty((Nw, Nw), dtype=proj_nw.dtype)
    U_ww[:M] = proj_nw[:M]

    if L > 0:
        proj_uw = proj_nw[M:]
        eig_w, C_ww = np.linalg.eigh(np.dot(dag(proj_uw), proj_uw))
        C_ul = np.dot(proj_uw, C_ww[:, np.argsort(-eig_w.real)[:L]])
        # eig_u, C_uu = np.linalg.eigh(np.dot(proj_uw, dag(proj_uw)))
        # C_ul = C_uu[:, np.argsort(-eig_u.real)[:L]]

        U_ww[M:] = np.dot(dag(C_ul), proj_uw)
    else:
        C_ul = np.empty((Nb - M, 0))

    normalize(C_ul)
    if ortho:
        lowdin(U_ww)
    else:
        normalize(U_ww)

    return U_ww, C_ul


class Wannier:
    """Maximally localized Wannier Functions

    Find the set of maximally localized Wannier functions using the
    spread functional of Marzari and Vanderbilt (PRB 56, 1997 page
    12847).
    """

    def __init__(self, nwannier, calc,
                 file=None,
                 nbands=None,
                 fixedenergy=None,
                 fixedstates=None,
                 spin=0,
                 initialwannier='random',
                 rng=np.random,
                 verbose=False):
        """
        Required arguments:

          ``nwannier``: The number of Wannier functions you wish to construct.
            This must be at least half the number of electrons in the system
            and at most equal to the number of bands in the calculation.

          ``calc``: A converged DFT calculator class.
            If ``file`` arg. is not provided, the calculator *must* provide the
            method ``get_wannier_localization_matrix``, and contain the
            wavefunctions (save files with only the density is not enough).
            If the localization matrix is read from file, this is not needed,
            unless ``get_function`` or ``write_cube`` is called.

        Optional arguments:

          ``nbands``: Bands to include in localization.
            The number of bands considered by Wannier can be smaller than the
            number of bands in the calculator. This is useful if the highest
            bands of the DFT calculation are not well converged.

          ``spin``: The spin channel to be considered.
            The Wannier code treats each spin channel independently.

          ``fixedenergy`` / ``fixedstates``: Fixed part of Heilbert space.
            Determine the fixed part of Hilbert space by either a maximal
            energy *or* a number of bands (possibly a list for multiple
            k-points).
            Default is None meaning that the number of fixed states is equated
            to ``nwannier``.

          ``file``: Read localization and rotation matrices from this file.

          ``initialwannier``: Initial guess for Wannier rotation matrix.
            Can be 'bloch' to start from the Bloch states, 'random' to be
            randomized, or a list passed to calc.get_initial_wannier.

          ``rng``: Random number generator for ``initialwannier``.

          ``verbose``: True / False level of verbosity.
          """
        # Bloch phase sign convention.
        # May require special cases depending on which code is used.
        sign = -1

        self.nwannier = nwannier
        self.calc = calc
        self.spin = spin
        self.verbose = verbose
        self.kpt_kc = calc.get_bz_k_points()
        assert len(calc.get_ibz_k_points()) == len(self.kpt_kc)
        self.kptgrid = get_monkhorst_pack_size_and_offset(self.kpt_kc)[0]
        self.Nk = len(self.kpt_kc)
        self.unitcell_cc = calc.get_atoms().get_cell()
        self.largeunitcell_cc = (self.unitcell_cc.T * self.kptgrid).T

        # Get list of neighbors and G vectors for each k-point, number of
        # directions and weight for each shell.
        self.kklst_kd, self.G_kdc, self.weight_d = \
            neighbors_and_weights(
                kpt_frac=self.kpt_kc,
                recip_cell=self.unitcell_cc.reciprocal(),
                verbose=self.verbose)
        self.Ndir = self.kklst_kd.shape[1]

        # Compute the directions to reach the neighboring k-points
        self.bvec_dc = np.empty((self.Ndir, 3), dtype=float)
        for d in range(self.Ndir):
            self.bvec_dc[d] = (self.kpt_kc[self.kklst_kd[0, d]]
                               - self.kpt_kc[0] - self.G_kdc[0, d])
        # convert to cartesian coordinates
        self.bvec_dc = self.bvec_dc @ \
                self.unitcell_cc.reciprocal() * 2 * np.pi

        # Set the inverse list of neighboring k-points
        self.invkklst_kd = np.empty((self.Nk, self.Ndir), dtype=np.uint32)
        for k1 in range(self.Nk):
            for d in range(self.Ndir):
                self.invkklst_kd[k1, d] = self.kklst_kd[:, d].tolist().index(k1)

        # Apply sign convention, everything related to neighboring k-points
        # must be completed before this change
        self.kpt_kc *= sign

        if nbands is not None:
            self.nbands = nbands
        else:
            self.nbands = calc.get_number_of_bands()
        if fixedenergy is None:
            if fixedstates is None:
                self.fixedstates_k = np.array([nwannier] * self.Nk, int)
            else:
                if isinstance(fixedstates, int):
                    fixedstates = [fixedstates] * self.Nk
                self.fixedstates_k = np.array(fixedstates, int)
        else:
            # Setting number of fixed states and EDF from specified energy.
            # All states below this energy (relative to Fermi level) are fixed.
            fixedenergy += calc.get_fermi_level()
            print(fixedenergy)
            self.fixedstates_k = np.array(
                [calc.get_eigenvalues(k, spin).searchsorted(fixedenergy)
                 for k in range(self.Nk)], int)
        self.edf_k = self.nwannier - self.fixedstates_k
        if verbose:
            print('Wannier: Fixed states            : %s' % self.fixedstates_k)
            print('Wannier: Extra degrees of freedom: %s' % self.edf_k)

        dummy = 0  # XXX: GPAW expects this argument but it does not use it
        Nw = self.nwannier
        Nb = self.nbands
        self.Z_dkww = np.empty((self.Ndir, self.Nk, Nw, Nw), complex)
        self.V_knw = np.zeros((self.Nk, Nb, Nw), complex)
        if file is None:
            self.Z_dknn = np.empty((self.Ndir, self.Nk, Nb, Nb), complex)
            for d in range(self.Ndir):
                for k in range(self.Nk):
                    k1 = self.kklst_kd[k, d]
                    G_c = self.G_kdc[k, d]
                    self.Z_dknn[d, k] = calc.get_wannier_localization_matrix(
                        nbands=Nb, dirG=dummy, kpoint=k, nextkpoint=k1,
                        G_I=G_c, spin=self.spin)
        self.initialize(file=file, initialwannier=initialwannier, rng=rng)

    def initialize(self, file=None, initialwannier='random', rng=np.random):
        """Re-initialize current rotation matrix.

        Keywords are identical to those of the constructor.
        """
        Nw = self.nwannier
        Nb = self.nbands

        if file is not None:
            self.Z_dknn, self.U_kww, self.C_kul = load(paropen(file, 'rb'))
        elif initialwannier == 'bloch':
            # Set U and C to pick the lowest Bloch states
            self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
            self.C_kul = []
            for U, M, L in zip(self.U_kww, self.fixedstates_k, self.edf_k):
                U[:] = np.identity(Nw, complex)
                if L > 0:
                    self.C_kul.append(
                        np.identity(Nb - M, complex)[:, :L])
                else:
                    self.C_kul.append([])
        elif initialwannier == 'random':
            # Set U and C to random (orthogonal) matrices
            self.U_kww = np.zeros((self.Nk, Nw, Nw), complex)
            self.C_kul = []
            for U, M, L in zip(self.U_kww, self.fixedstates_k, self.edf_k):
                U[:] = random_orthogonal_matrix(Nw, rng, real=False)
                if L > 0:
                    self.C_kul.append(random_orthogonal_matrix(
                        Nb - M, rng=rng, real=False)[:, :L])
                else:
                    self.C_kul.append(np.array([]))
        else:
            # Use initial guess to determine U and C
            self.C_kul, self.U_kww = self.calc.initial_wannier(
                initialwannier, self.kptgrid, self.fixedstates_k,
                self.edf_k, self.spin, self.nbands)
        self.update()

    def save(self, file):
        """Save information on localization and rotation matrices to file."""
        dump((self.Z_dknn, self.U_kww, self.C_kul), paropen(file, 'wb'))

    def update(self):
        # Update large rotation matrix V (from rotation U and coeff C)
        for k, M in enumerate(self.fixedstates_k):
            self.V_knw[k, :M] = self.U_kww[k, :M]
            if M < self.nwannier:
                self.V_knw[k, M:] = np.dot(self.C_kul[k], self.U_kww[k, M:])
            # else: self.V_knw[k, M:] = 0.0

        # Calculate the Zk matrix from the large rotation matrix:
        # Zk = V^d[k] Zbloch V[k1]
        for d in range(self.Ndir):
            for k in range(self.Nk):
                k1 = self.kklst_kd[k, d]
                self.Z_dkww[d, k] = np.dot(dag(self.V_knw[k]), np.dot(
                    self.Z_dknn[d, k], self.V_knw[k1]))

        # Update the new Z matrix
        self.Z_dww = self.Z_dkww.sum(axis=1) / self.Nk

    def get_centers(self, scaled=False):
        """Calculate the Wannier centers

        Centers positions can be scaled with respect to the original unit cell.

        Eq. 13 in https://doi.org/10.1016/j.cpc.2007.11.016

        ::

                  1   --
          pos = - --  >  wb * b * phase(diag(Z(k,b)))
                  Nk  --
                      k,b
        """
        phZ_dw = np.angle(self.Z_dww.diagonal(axis1=1, axis2=2)) / (2 * np.pi)
        coord_wc = np.zeros((self.nwannier, 3), dtype=float)
        for w in range(self.nwannier):
            for d in range(self.Ndir):
                coord_wc[w] += (self.weight_d[d] * phZ_dw[d, w]
                                * self.bvec_dc[d])
        coord_wc = - coord_wc / sum(self.weight_d)
        if scaled:
            # convert from repeated cell to unit cell
            coord_wc *= self.kptgrid
        else:
            # cartesian coordinates
            coord_wc = coord_wc @ self.largeunitcell_cc

        return coord_wc

    def get_radii(self):
        r"""Calculate the spread of the Wannier functions.

        ::

                        --  /  L  \ 2       2
          radius**2 = - >   | --- |   ln |Z|
                        --d \ 2pi /
        """
        r2 = -np.dot(self.largeunitcell_cc.diagonal()**2 / (2 * pi)**2,
                     np.log(abs(self.Z_dww[:3].diagonal(0, 1, 2))**2))
        return np.sqrt(r2)

    def get_spectral_weight(self, w):
        return abs(self.V_knw[:, :, w])**2 / self.Nk

    def get_pdos(self, w, energies, width):
        """Projected density of states (PDOS).

        Returns the (PDOS) for Wannier function ``w``. The calculation
        is performed over the energy grid specified in energies. The
        PDOS is produced as a sum of Gaussians centered at the points
        of the energy grid and with the specified width.
        """
        spec_kn = self.get_spectral_weight(w)
        dos = np.zeros(len(energies))
        for k, spec_n in enumerate(spec_kn):
            eig_n = self.calc.get_eigenvalues(kpt=k, spin=self.spin)
            for weight, eig in zip(spec_n, eig_n):
                # Add gaussian centered at the eigenvalue
                x = ((energies - eig) / width)**2
                dos += weight * np.exp(-x.clip(0., 40.)) / (sqrt(pi) * width)
        return dos

    def translate(self, w, R):
        """Translate the w'th Wannier function

        The distance vector R = [n1, n2, n3], is in units of the basis
        vectors of the small cell.
        """
        for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
            U_ww[:, w] *= np.exp(2.j * pi * np.dot(np.array(R), kpt_c))
        self.update()

    def translate_to_cell(self, w, cell):
        """Translate the w'th Wannier function to specified cell"""
        # scaled_c = np.angle(self.Z_dww[:3, w, w]) * self.kptgrid / (2 * pi)
        scaled_c = self.get_centers(scaled=True)[w]
        trans = np.array(cell) - np.floor(scaled_c)
        self.translate(w, trans)

    def translate_all_to_cell(self, cell=[0, 0, 0]):
        r"""Translate all Wannier functions to specified cell.

        Move all Wannier orbitals to a specific unit cell.  There
        exists an arbitrariness in the positions of the Wannier
        orbitals relative to the unit cell. This method can move all
        orbitals to the unit cell specified by ``cell``.  For a
        `\Gamma`-point calculation, this has no effect. For a
        **k**-point calculation the periodicity of the orbitals are
        given by the large unit cell defined by repeating the original
        unitcell by the number of **k**-points in each direction.  In
        this case it is useful to move the orbitals away from the
        boundaries of the large cell before plotting them. For a bulk
        calculation with, say 10x10x10 **k** points, one could move
        the orbitals to the cell [2,2,2].  In this way the pbc
        boundary conditions will not be noticed.
        """
        # scaled_wc = (np.angle(self.Z_dww[:3].diagonal(0, 1, 2)).T *
        #              self.kptgrid / (2 * pi))
        scaled_wc = self.get_centers(scaled=True)
        trans_wc = np.array(cell)[None] - np.floor(scaled_wc)
        for kpt_c, U_ww in zip(self.kpt_kc, self.U_kww):
            U_ww *= np.exp(2.j * pi * np.dot(trans_wc, kpt_c))
        self.update()

    def distances(self, R):
        """Relative distances between centers.

        Returns a matrix with the distances between different Wannier centers.
        R = [n1, n2, n3] is in units of the basis vectors of the small cell
        and allows one to measure the distance with centers moved to a
        different small cell.
        The dimension of the matrix is [Nw, Nw].
        """
        Nw = self.nwannier
        cen = self.get_centers()
        r1 = cen.repeat(Nw, axis=0).reshape(Nw, Nw, 3)
        r2 = cen.copy()
        for i in range(3):
            r2 += self.unitcell_cc[i] * R[i]

        r2 = np.swapaxes(r2.repeat(Nw, axis=0).reshape(Nw, Nw, 3), 0, 1)
        return np.sqrt(np.sum((r1 - r2)**2, axis=-1))

    def get_hopping(self, R):
        """Returns the matrix H(R)_nm=<0,n|H|R,m>.

        ::

                                1   _   -ik.R
          H(R) = <0,n|H|R,m> = --- >_  e      H(k)
                                Nk  k

        where R is the cell-distance (in units of the basis vectors of
        the small cell) and n,m are indices of the Wannier functions.
        """
        H_ww = np.zeros([self.nwannier, self.nwannier], complex)
        for k, kpt_c in enumerate(self.kpt_kc):
            phase = np.exp(-2.j * pi * np.dot(np.array(R), kpt_c))
            H_ww += self.get_hamiltonian(k) * phase
        return H_ww / self.Nk

    def get_hamiltonian(self, k=0):
        """Get Hamiltonian at existing k-vector of index k

        ::

                  dag
          H(k) = V    diag(eps )  V
                  k           k    k
        """
        eps_n = self.calc.get_eigenvalues(kpt=k, spin=self.spin)[:self.nbands]
        return np.dot(dag(self.V_knw[k]) * eps_n, self.V_knw[k])

    def get_hamiltonian_kpoint(self, kpt_c):
        """Get Hamiltonian at some new arbitrary k-vector

        ::

                  _   ik.R
          H(k) = >_  e     H(R)
                  R

        Warning: This method moves all Wannier functions to cell (0, 0, 0)
        """
        if self.verbose:
            print('Translating all Wannier functions to cell (0, 0, 0)')
        self.translate_all_to_cell()
        max = (self.kptgrid - 1) // 2
        N1, N2, N3 = max
        Hk = np.zeros([self.nwannier, self.nwannier], complex)
        for n1 in range(-N1, N1 + 1):
            for n2 in range(-N2, N2 + 1):
                for n3 in range(-N3, N3 + 1):
                    R = np.array([n1, n2, n3], float)
                    hop_ww = self.get_hopping(R)
                    phase = np.exp(+2.j * pi * np.dot(R, kpt_c))
                    Hk += hop_ww * phase
        return Hk

    def get_function(self, index, repeat=None):
        r"""Get Wannier function on grid.

        Returns an array with the funcion values of the indicated Wannier
        function on a grid with the size of the *repeated* unit cell.

        For a calculation using **k**-points the relevant unit cell for
        eg. visualization of the Wannier orbitals is not the original unit
        cell, but rather a larger unit cell defined by repeating the
        original unit cell by the number of **k**-points in each direction.
        Note that for a `\Gamma`-point calculation the large unit cell
        coinsides with the original unit cell.
        The large unitcell also defines the periodicity of the Wannier
        orbitals.

        ``index`` can be either a single WF or a coordinate vector in terms
        of the WFs.
        """

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kptgrid
        N1, N2, N3 = repeat

        dim = self.calc.get_number_of_grid_points()
        largedim = dim * [N1, N2, N3]

        wanniergrid = np.zeros(largedim, dtype=complex)
        for k, kpt_c in enumerate(self.kpt_kc):
            # The coordinate vector of wannier functions
            if isinstance(index, int):
                vec_n = self.V_knw[k, :, index]
            else:
                vec_n = np.dot(self.V_knw[k], index)

            wan_G = np.zeros(dim, complex)
            for n, coeff in enumerate(vec_n):
                wan_G += coeff * self.calc.get_pseudo_wave_function(
                    n, k, self.spin, pad=True)

            # Distribute the small wavefunction over large cell:
            for n1 in range(N1):
                for n2 in range(N2):
                    for n3 in range(N3):  # sign?
                        e = np.exp(-2.j * pi * np.dot([n1, n2, n3], kpt_c))
                        wanniergrid[n1 * dim[0]:(n1 + 1) * dim[0],
                                    n2 * dim[1]:(n2 + 1) * dim[1],
                                    n3 * dim[2]:(n3 + 1) * dim[2]] += e * wan_G

        # Normalization
        wanniergrid /= np.sqrt(self.Nk)
        return wanniergrid

    def write_cube(self, index, fname, repeat=None, real=True):
        """Dump specified Wannier function to a cube file"""
        from ase.io import write

        # Default size of plotting cell is the one corresponding to k-points.
        if repeat is None:
            repeat = self.kptgrid
        atoms = self.calc.get_atoms() * repeat
        func = self.get_function(index, repeat)

        # Handle separation of complex wave into real parts
        if real:
            if self.Nk == 1:
                func *= np.exp(-1.j * np.angle(func.max()))
                if 0:
                    assert max(abs(func.imag).flat) < 1e-4
                func = func.real
            else:
                func = abs(func)
        else:
            phase_fname = fname.split('.')
            phase_fname.insert(1, 'phase')
            phase_fname = '.'.join(phase_fname)
            write(phase_fname, atoms, data=np.angle(func), format='cube')
            func = abs(func)

        write(fname, atoms, data=func, format='cube')

    def localize(self, step=0.25, tolerance=1e-08,
                 updaterot=True, updatecoeff=True):
        """Optimize rotation to give maximal localization"""
        md_min(self, step, tolerance, verbose=self.verbose,
               updaterot=updaterot, updatecoeff=updatecoeff)

    def get_functional_value(self):
        """Calculate the value of the spread functional.

        ::

          Tr[|ZI|^2]=sum(I)sum(n) w_i|Z_(i)_nn|^2,

        where w_i are weights."""
        a_d = np.sum(np.abs(self.Z_dww.diagonal(0, 1, 2))**2, axis=1)
        return np.dot(a_d, self.weight_d).real

    def get_gradients(self):
        # Determine gradient of the spread functional.
        #
        # The gradient for a rotation A_kij is::
        #
        #    dU = dRho/dA_{k,i,j} = sum(I) sum(k')
        #            + Z_jj Z_kk',ij^* - Z_ii Z_k'k,ij^*
        #            - Z_ii^* Z_kk',ji + Z_jj^* Z_k'k,ji
        #
        # The gradient for a change of coefficients is::
        #
        #   dRho/da^*_{k,i,j} = sum(I) [[(Z_0)_{k} V_{k'} diag(Z^*) +
        #                                (Z_0_{k''})^d V_{k''} diag(Z)] *
        #                                U_k^d]_{N+i,N+j}
        #
        # where diag(Z) is a square,diagonal matrix with Z_nn in the diagonal,
        # k' = k + dk and k = k'' + dk.
        #
        # The extra degrees of freedom chould be kept orthonormal to the fixed
        # space, thus we introduce lagrange multipliers, and minimize instead::
        #
        #     Rho_L=Rho- sum_{k,n,m} lambda_{k,nm} <c_{kn}|c_{km}>
        #
        # for this reason the coefficient gradients should be multiplied
        # by (1 - c c^d).

        Nb = self.nbands
        Nw = self.nwannier

        dU = []
        dC = []
        for k in range(self.Nk):
            M = self.fixedstates_k[k]
            L = self.edf_k[k]
            U_ww = self.U_kww[k]
            C_ul = self.C_kul[k]
            Utemp_ww = np.zeros((Nw, Nw), complex)
            Ctemp_nw = np.zeros((Nb, Nw), complex)

            for d, weight in enumerate(self.weight_d):
                if abs(weight) < 1.0e-6:
                    continue

                Z_knn = self.Z_dknn[d]
                diagZ_w = self.Z_dww[d].diagonal()
                Zii_ww = np.repeat(diagZ_w, Nw).reshape(Nw, Nw)
                k1 = self.kklst_kd[k, d]
                k2 = self.invkklst_kd[k, d]
                V_knw = self.V_knw
                Z_kww = self.Z_dkww[d]

                if L > 0:
                    Ctemp_nw += weight * np.dot(
                        np.dot(Z_knn[k], V_knw[k1]) * diagZ_w.conj() +
                        np.dot(dag(Z_knn[k2]), V_knw[k2]) * diagZ_w,
                        dag(U_ww))

                temp = Zii_ww.T * Z_kww[k].conj() - Zii_ww * Z_kww[k2].conj()
                Utemp_ww += weight * (temp - dag(temp))
            dU.append(Utemp_ww.ravel())
            if L > 0:
                # Ctemp now has same dimension as V, the gradient is in the
                # lower-right (Nb-M) x L block
                Ctemp_ul = Ctemp_nw[M:, M:]
                G_ul = Ctemp_ul - np.dot(np.dot(C_ul, dag(C_ul)), Ctemp_ul)
                dC.append(G_ul.ravel())

        return np.concatenate(dU + dC)

    def step(self, dX, updaterot=True, updatecoeff=True):
        # dX is (A, dC) where U->Uexp(-A) and C->C+dC
        Nw = self.nwannier
        Nk = self.Nk
        M_k = self.fixedstates_k
        L_k = self.edf_k
        if updaterot:
            A_kww = dX[:Nk * Nw**2].reshape(Nk, Nw, Nw)
            for U, A in zip(self.U_kww, A_kww):
                H = -1.j * A.conj()
                epsilon, Z = np.linalg.eigh(H)
                # Z contains the eigenvectors as COLUMNS.
                # Since H = iA, dU = exp(-A) = exp(iH) = ZDZ^d
                dU = np.dot(Z * np.exp(1.j * epsilon), dag(Z))
                if U.dtype == float:
                    U[:] = np.dot(U, dU).real
                else:
                    U[:] = np.dot(U, dU)

        if updatecoeff:
            start = 0
            for C, unocc, L in zip(self.C_kul, self.nbands - M_k, L_k):
                if L == 0 or unocc == 0:
                    continue
                Ncoeff = L * unocc
                deltaC = dX[Nk * Nw**2 + start: Nk * Nw**2 + start + Ncoeff]
                C += deltaC.reshape(unocc, L)
                gram_schmidt(C)
                start += Ncoeff

        self.update()
