from math import pi, sqrt
from typing import Sequence, Union, Tuple, Optional
from numbers import Real

import numpy as np
import numpy.linalg as la

import ase.units as units
from ase.atoms import Atoms
from ase.dft import monkhorst_pack
from ase.utils import lazyproperty


RealSequence5D = Sequence[Sequence[Sequence[Sequence[Sequence[Real]]]]]
RealSequence3D = Sequence[Sequence[Sequence[Real]]]


class PhononsData:
    def __init__(self,
                 atoms: Atoms,
                 force_constants: Union[RealSequence5D, np.ndarray],
                 supercell: Sequence[int],
                 center_refcell: bool,
                 indices: Union[Sequence[int], np.ndarray] = None,
                 ) -> None:

        self._indices = self._indices_or_default(atoms, indices)

        self._supercell = tuple(supercell)

        n_cells, n_atoms = self._check_dimensions(atoms, np.asarray(force_constants),
                                         indices=self._indices, supercell=supercell)
        self._atoms = atoms.copy()

        self._center_refcell = center_refcell

        self._force_constants_3d = (np.asarray(force_constants)
                                    .reshape(n_cells, 3 * n_atoms, 3 * n_atoms).copy())

        self._Z_avv: Optional[np.ndarray] = None
        self._eps_vv: Optional[np.ndarray] = None

    @classmethod
    def from_3d(cls, atoms: Atoms,
                force_constants: Union[RealSequence3D, np.ndarray],
                supercell: Sequence[int],
                center_refcell: bool,
                indices: Union[Sequence[int], np.ndarray] = None) -> 'PhononsData':
        """Instantiate PhononsData when the force constants are in a 3Nx3N format

        Args:
            atoms: Equilibrium geometry of vibrating system

            hessian: Second-derivative in energy with respect to
                Cartesian nuclear movements as a (3N, 3N) array.

            @@@@@@@@@@@@@@@@@@@@ TODO @@@@@@@@@@@@@@@@@

        """
        indices = cls._indices_or_default(atoms, indices)
        assert indices is not None  # Show Mypy that indices is valid

        force_constants_arr = np.asarray(force_constants)
        n_cells, n_atoms = cls._check_dimensions(atoms, force_constants_arr,
                                                 indices=indices,
                                                 supercell=supercell,
                                                 three_d=True)

        return cls(atoms, force_constants_arr.reshape(n_cells, n_atoms, 3, n_atoms, 3),
                   supercell=supercell, center_refcell=center_refcell, indices=indices)

    @staticmethod
    def _indices_or_default(atoms: Atoms, indices: Union[Sequence[int], np.ndarray, None]) -> np.ndarray:
        if indices is None:
            return np.arange(len(atoms), dtype=int)
        else:
            return np.array(indices, dtype=int)

    @staticmethod
    def _check_dimensions(atoms: Atoms,
                          force_constants: np.ndarray,
                          indices: np.ndarray,
                          supercell: Sequence[int],
                          three_d: bool = False) -> Tuple[int, int]:
        """Sanity check on array shapes from input data

        Args:
            atoms: Structure
            indices: Indices of atoms used in FCs
            sepercell: Number of repeats along each cell vector
            force_constants: Proposed FCs array

        Returns:
            Tuple of `(n_cells, n_atoms)` with the number of cells in the
            supercell and the number of atoms contributing to FCs.

        Raises:
            ValueError if dimensions of FC or supercell are incorrect

        """

        if len(supercell) != 3:
            raise ValueError('supercell should be a sequence of length 3')

        n_atoms = len(atoms[indices])
        n_cells = np.product(supercell)

        if three_d:
            ref_shape = [n_cells, n_atoms * 3, n_atoms * 3]
            ref_shape_txt = f'{n_cells}x{3*n_atoms}x{3*n_atoms}'

        else:
            ref_shape = [n_cells, n_atoms, 3, n_atoms, 3]
            ref_shape_txt = f'{n_cells}x{n_atoms}x3x{n_atoms}x3'

        if (isinstance(force_constants, np.ndarray)
            and force_constants.shape == tuple(ref_shape)):
            return n_cells, n_atoms
        else:
            raise ValueError("Force constants for these atoms should be a "
                             "{} numpy array.".format(ref_shape_txt))

    # XXX: TEMPORARY COPYPASTA
    @property
    def offset(self):        # Reference cell offset
        if not self._center_refcell:
            # Corner cell
            return 0
        else:
            # Center cell
            N_c = self._supercell
            return (N_c[0] // 2 * (N_c[1] * N_c[2]) +
                    N_c[1] // 2 * N_c[2] +
                    N_c[2] // 2)

    # XXX: TEMPORARY COPYPASTA
    @lazyproperty
    def lattice_vectors(self):
        """Return the integer coordinates for all cells in the supercell."""
        # Lattice vectors relevative to the reference cell
        R_cN = np.indices(self._supercell).reshape(3, -1)
        N_c = np.array(self._supercell)[:, np.newaxis]
        if self.offset == 0:
            R_cN += N_c // 2
            R_cN %= N_c
        R_cN -= N_c // 2
        return R_cN

    def get_force_constants(self):
        """ @@@ TODO @@@ """
        n_cells = self._force_constants_3d.shape[0]
        n_atoms = self._force_constants_3d.shape[1] // 3
        return self._force_constants_3d.reshape(n_cells, n_atoms, 3, n_atoms, 3).copy()

    def get_force_constants_3d(self):
        """ @@@ TODO @@@ """
        return self._force_constants_3d.copy()

    def get_band_structure(self, path, modes=False, born=False, verbose=True):
        omega_kl = self.band_structure(path.kpts, modes, born, verbose)
        if modes:
            # XXX this comes from commit b13de72e208990a263967
            assert 0
            omega_kl, modes = omega_kl

        from ase.spectrum.band_structure import BandStructure
        bs = BandStructure(path, energies=omega_kl[None])
        return bs

    def compute_dynamical_matrix(self, q_scaled: np.ndarray, D_N: np.ndarray):
        """ Computation of the dynamical matrix in momentum space D_ab(q).
            This is a Fourier transform from real-space dynamical matrix D_N
            for a given momentum vector q.

        q_scaled: q vector in scaled coordinates.

        D_N: the dynamical matrix in real-space. It is necessary, at least
             currently, to provide this matrix explicitly (rather than use
             self.D_N) because this matrix is modified by the Born charges
             contributions and these modifications are momentum (q) dependent.

        Result:
            D(q): two-dimensional, complex-valued array of
                  shape=(3 * natoms, 3 * natoms).
        """
        # Evaluate fourier sum
        R_cN = self.lattice_vectors
        phase_N = np.exp(-2.j * pi * np.dot(q_scaled, R_cN))
        D_q = np.sum(phase_N[:, np.newaxis, np.newaxis] * D_N, axis=0)
        return D_q

    def _realspace_dynmat(self):
        m_a = self._atoms.get_masses()
        m_inv_x = np.repeat(m_a[self._indices]**-0.5, 3)
        M_inv = np.outer(m_inv_x, m_inv_x)
        D_N = self._force_constants_3d.copy() * M_inv[np.newaxis, :, :]
        return D_N

    def band_structure(self, path_kc, modes=False, born=False, verbose=True):
        """Calculate phonon dispersion along a path in the Brillouin zone.

        The dynamical matrix at arbitrary q-vectors is obtained by Fourier
        transforming the real-space force constants. In case of negative
        eigenvalues (squared frequency), the corresponding negative frequency
        is returned.

        Frequencies and modes are in units of eV and Ang/sqrt(amu),
        respectively.

        Parameters:

        path_kc: ndarray
            List of k-point coordinates (in units of the reciprocal lattice
            vectors) specifying the path in the Brillouin zone for which the
            dynamical matrix will be calculated.
        modes: bool
            Returns both frequencies and modes when True.
        born: bool
            Include non-analytic part given by the Born effective charges and
            the static part of the high-frequency dielectric tensor. This
            contribution to the force constant accounts for the splitting
            between the LO and TO branches for q -> 0.
        verbose: bool
            Print warnings when imaginary frequncies are detected.

        """

        if born:
            assert self._Z_avv is not None
            assert self._eps_vv is not None

        # Dynamical matrix in real-space
        D_N = self._realspace_dynmat()

        # Lists for frequencies and modes along path
        omega_kl = []
        u_kl = []

        # Reciprocal basis vectors for use in non-analytic contribution
        reci_vc = 2 * pi * la.inv(self._atoms.cell)
        # Unit cell volume in Bohr^3
        vol = abs(la.det(self._atoms.cell)) / units.Bohr**3

        m_a = self._atoms.get_masses()
        m_inv_x = np.repeat(m_a[self._indices]**-0.5, 3)

        for q_c in path_kc:

            # Add non-analytic part
            if born:
                # q-vector in cartesian coordinates
                q_v = np.dot(reci_vc, q_c)
                # Non-analytic contribution to force constants in atomic units
                qdotZ_av = np.dot(q_v, self._Z_avv).ravel()
                C_na = (4 * pi * np.outer(qdotZ_av, qdotZ_av) /
                        np.dot(q_v, np.dot(self._eps_vv, q_v)) / vol)
                self.C_na = C_na / units.Bohr**2 * units.Hartree
                # Add mass prefactor and convert to eV / (Ang^2 * amu)
                M_inv = np.outer(m_inv_x, m_inv_x)
                D_na = C_na * M_inv / units.Bohr**2 * units.Hartree
                self.D_na = D_na
                D_N += D_na / np.prod(self._supercell)

            # if np.prod(self.N_c) == 1:
            #
            #     q_av = np.tile(q_v, len(self.indices))
            #     q_xx = np.vstack([q_av]*len(self.indices)*3)
            #     D_m += q_xx

            # Evaluate fourier sum
            D_q = self.compute_dynamical_matrix(q_c, D_N)

            if modes:
                omega2_l, u_xl = la.eigh(D_q, UPLO='U')
                # Sort eigenmodes according to eigenvalues (see below) and
                # multiply with mass prefactor
                u_lx = (m_inv_x[:, np.newaxis] *
                        u_xl[:, omega2_l.argsort()]).T.copy()
                u_kl.append(u_lx.reshape((-1, len(self.indices), 3)))
            else:
                omega2_l = la.eigvalsh(D_q, UPLO='U')

            # Sort eigenvalues in increasing order
            omega2_l.sort()
            # Use dtype=complex to handle negative eigenvalues
            omega_l = np.sqrt(omega2_l.astype(complex))

            # Take care of imaginary frequencies
            if not np.all(omega2_l >= 0.):
                indices = np.where(omega2_l < 0)[0]

                if verbose:
                    print('WARNING, %i imaginary frequencies at '
                          'q = (% 5.2f, % 5.2f, % 5.2f) ; (omega_q =% 5.3e*i)'
                          % (len(indices), q_c[0], q_c[1], q_c[2],
                             omega_l[indices][0].imag))

                omega_l[indices] = -1 * np.sqrt(np.abs(omega2_l[indices].real))

            omega_kl.append(omega_l.real)

        # Conversion factor: sqrt(eV / Ang^2 / amu) -> eV
        s = units._hbar * 1e10 / sqrt(units._e * units._amu)
        omega_kl = s * np.asarray(omega_kl)

        if modes:
            return omega_kl, np.asarray(u_kl)

        return omega_kl

    def get_dos(self, kpts=(10, 10, 10), npts=1000, delta=1e-3, indices=None):
        from ase.spectrum.dosdata import RawDOSData
        # dos = self.dos(kpts, npts, delta, indices)
        kpts_kc = monkhorst_pack(kpts)
        omega_w = self.band_structure(kpts_kc).ravel()
        dos = RawDOSData(omega_w, np.ones_like(omega_w))
        return dos

    def dos(self, kpts=(10, 10, 10), npts=1000, delta=1e-3, indices=None):
        """Calculate phonon dos as a function of energy.

        Parameters:

        qpts: tuple
            Shape of Monkhorst-Pack grid for sampling the Brillouin zone.
        npts: int
            Number of energy points.
        delta: float
            Broadening of Lorentzian line-shape in eV.
        indices: list
            If indices is not None, the atomic-partial dos for the specified
            atoms will be calculated.

        """

        # Monkhorst-Pack grid
        kpts_kc = monkhorst_pack(kpts)
        N = np.prod(kpts)
        # Get frequencies
        omega_kl = self.band_structure(kpts_kc)
        # Energy axis and dos
        omega_e = np.linspace(0., np.amax(omega_kl) + 5e-3, num=npts)
        dos_e = np.zeros_like(omega_e)

        # Sum up contribution from all q-points and branches
        for omega_l in omega_kl:
            diff_el = (omega_e[:, np.newaxis] - omega_l[np.newaxis, :])**2
            dos_el = 1. / (diff_el + (0.5 * delta)**2)
            dos_e += dos_el.sum(axis=1)

        dos_e *= 1. / (N * pi) * 0.5 * delta

        return omega_e, dos_e
