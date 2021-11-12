from numpy import unique, log, sqrt, exp
from numpy import zeros, ones, absolute, diagflat, flatnonzero
from numpy.linalg import eigh, norm, pinv
from scipy.linalg import lstsq  # performs better than numpy.linalg.lstsq

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.calculator import CalculatorSetupError, CalculationFailed


class Harmonic(Calculator):
    """Class for calculations with a Hessian-based harmonic force-field.

    Energy and forces of this calculator are based on the Cartesian Hessian
    for a local reference configuration, i.e. if desired, on the Hessian
    matrix transformed to a user-defined coordinate system.
    The required Hessian has to be passed as an argument, e.g. predetermined
    numerically via central finite differences in Cartesian coordinates.
    Note that a potential being harmonic in Cartesian coordinates **x** is not
    necessarily equivalently harmonic in another coordinate system **q**,
    e.g. when the transformation between the coordinate systems is non-linear.
    By default, calculations are performed in Cartesian coordinates in which
    energy and forces are not rotationally and translationally invariant.
    Systems with variable orientation, require rotationally and translationally
    invariant calculations for which a set of appropriate coordinates has to
    be defined. This can be a set of (redundant) internal coordinates (bonds,
    angles, dihedrals, coordination numbers, ...) or any other user-defined
    coordinate system.

    The :class:`Harmonic` calculator can be used to compute Anharmonic
    Corrections to the Harmonic Approximation. [1]_
    """
    # Amsler, J. et al., J. Chem. Theory Comput. 2021, 17 (2), 1155-1169.

    implemented_properties = ['energy', 'forces']
    default_parameters = {
            'ref_atoms': None,
            'ref_energy': 0.0,
            'hessian_x': None,
            'hessian_limit': 0.0,
            'get_q_from_x': None,
            'get_jacobian': None,
            'cartesian': True,
            'variable_orientation': False,
            'constrained_q': None,
            'rcond': 1e-7,
            'zero_thresh': 0.0,
    }
    nolabel = True

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        ref_atoms: :class:`~ase.Atoms` object
            Reference structure for which energy (``ref_energy``) and Hessian
            matrix in Cartesian coordinates (``hessian_x``) are provided.

        ref_energy: float, optional, default: 0.0
            Energy of the reference structure ``ref_atoms``, typically in `eV`.

        hessian_x: numpy array
            Cartesian Hessian matrix for the reference structure ``ref_atoms``.
            If a user-defined coordinate system is provided via
            ``get_q_from_x`` and ``get_jacobian``, the Cartesian Hessian matrix
            is transformed to the user-defined coordinate system and back to
            Cartesian coordinates, thereby eliminating rotational and
            translational traits from the Hessian. The Hessian matrix
            obtained after this double-transformation is then used as
            the reference Hessian matrix to evaluate energy and forces for
            ``cartesian = True``. For ``cartesian = False`` the reference
            Hessian matrix transformed to the user-defined coordinates is used
            to compute energy and forces.

        hessian_limit: float, optional, default: 0.0
            Reconstruct the reference Hessian matrix with a lower limit for the
            eigenvalues, typically in `eV/A^2`. Eigenvalues in the interval
            [``zero_thresh``, ``hessian_limit``] are set to ``hessian_limit``
            while the eigenvectors are left untouched.

        get_q_from_x: python function, optional, default: None (Cartesians)
            Function that returns a vector of user-defined coordinates **q** for
            a given :class:`~ase.Atoms` object 'atoms'. The signature should be:
            :obj:`get_q_from_x(atoms)`.
            If not provided, ``cartesian`` is forcefully set to True.

        get_jacobian: python function, optional, default: None (Cartesians)
            Function that returns the geometric Jacobian matrix of the
            user-defined coordinates **q** w.r.t. Cartesian coordinates **x**
            defined as `dq/dx` (Wilson B-matrix) for a given :class:`~ase.Atoms`
            object 'atoms'. The signature should be: :obj:`get_jacobian(atoms)`.
            If not provided, ``cartesian`` is forcefully set to True.

        cartesian: bool, optional, default: True
            Set to True to evaluate energy and forces based on the reference
            Hessian (system harmonic in Cartesian coordinates).
            Set to False to evaluate energy and forces based on the reference
            Hessian transformed to user-defined coordinates (system harmonic in
            user-defined coordinates).

        variable_orientation: bool, optional, default: False
            Set to True if the orientation of the attached :class:`~ase.Atoms`
            object with respect to ``ref_atoms`` might be different (typically
            for molecules).
            Set to False to speed up the calculation when ``cartesian = True``.

        constrained_q: list, optional, default: None
            A list of indices 'i' of constrained coordinates `q_i` to be
            projected out from the Hessian matrix
            (e.g. remove forces along imaginary mode of a transition state).

        rcond: float, optional, default: 1e-7
            Cutoff for singular value decomposition in the computation of the
            Moore-Penrose pseudo-inverse during transformation of the Hessian
            matrix. Equivalent to the rcond parameter in scipy.linalg.lstsq.

        zero_thresh: float, optional, default: 0.0
            Reconstruct the reference Hessian matrix with absolute eigenvalues
            below this threshold set to zero.
        """
        super().__init__(**kwargs)

    def set(self, **kwargs):
        changed_parameters = super().set(**kwargs)
        changes = ['ref_atoms', 'hessian_x', 'hessian_limit', 'get_q_from_x',
                   'get_jacobian', 'variable_orientation', 'constrained_q',
                   'rcond', 'zero_thresh']  # almost any change -> self.update()
        if [change for change in changes if change in changed_parameters]:
            self.update()  # should be called during super().__init__(...)
        return changed_parameters

    def update(self):
        coord_fs = [self.parameters.get_q_from_x, self.parameters.get_jacobian]
        if None in coord_fs:
            if not all([func is None for func in coord_fs]):
                msg = ('A user-defined coordinate system requires both '
                       '`get_q_from_x` and `get_jacobian`.')
                raise CalculatorSetupError(msg)
            if self.parameters.variable_orientation:
                msg = ('The use of `variable_orientation` requires a '
                       'user-defined, translationally and rotationally '
                       'invariant coordinate system.')
                raise CalculatorSetupError(msg)

            # Cartesian coordinates
            self.parameters.cartesian = True
            self.get_q_from_x = lambda x: x.get_positions()
            self.get_jacobian = lambda x: diagflat(ones(3 * len(x)))
        else:  # user-defined coordinates
            self.get_q_from_x = self.parameters.get_q_from_x
            self.get_jacobian = self.parameters.get_jacobian

        # reference Cartesian coords. x0; reference user-defined coords. q0
        self.x0 = self.parameters.ref_atoms.get_positions().ravel()
        self.q0 = self.get_q_from_x(self.parameters.ref_atoms).ravel()
        self.setup_reference_hessians()  # self.hessian_x and self.hessian_q

        # store number of zero eigenvalues of G-matrix for redundancy check
        jac0 = self.get_jacobian(self.parameters.ref_atoms)
        Gmat = jac0.T @ jac0
        self.Gmat_eigvals, _ = eigh(Gmat)  # stored for inspection purposes
        self.zero_eigvals = len(flatnonzero(absolute(self.Gmat_eigvals) <
                                            self.parameters.zero_thresh))

    def setup_reference_hessians(self):
        """Prepare projector to project out constrained user-defined coordinates
        **q** from Hessian. Then do transformation to user-defined coordinates
        and back. Relevant literature:
        * Peng, C. et al. J. Comput. Chem. 1996, 17 (1), 49-56.
        * Baker, J. et al. J. Chem. Phys. 1996, 105 (1), 192–212."""
        jac0 = self.get_jacobian(self.parameters.ref_atoms)  # Jacobian (dq/dx)
        jac0 = self.constrain_jac(jac0)  # for reference Cartesian coordinates
        ijac0 = self.get_ijac(jac0, self.parameters.rcond)
        self.transform2reference_hessians(jac0, ijac0)  # perform projection

    def constrain_jac(self, jac):
        """Procedure by Peng, Ayala, Schlegel and Frisch adjusted for redundant
        coordinates.
        Peng, C. et al. J. Comput. Chem. 1996, 17 (1), 49–56.
        """
        proj = jac @ jac.T  # build non-redundant projector
        constrained_q = self.parameters.constrained_q or []
        Cmat = zeros(proj.shape)  # build projector for constraints
        Cmat[constrained_q, constrained_q] = 1.0
        proj = proj - proj @ Cmat @ pinv(Cmat @ proj @ Cmat) @ Cmat @ proj
        jac = pinv(jac) @ proj  # come back to redundant projector
        return jac.T

    def transform2reference_hessians(self, jac0, ijac0):
        """Transform Cartesian Hessian matrix to user-defined coordinates
        and back to Cartesian coordinates. For suitable coordinate systems
        (e.g. internals) this removes rotational and translational degrees of
        freedom. Furthermore, apply the lower limit to the force constants
        and reconstruct Hessian matrix."""
        hessian_x = self.parameters.hessian_x
        hessian_x = 0.5 * (hessian_x + hessian_x.T)  # guarantee symmetry
        hessian_q = ijac0.T @ hessian_x @ ijac0  # forward transformation
        hessian_x = jac0.T @ hessian_q @ jac0  # backward transformation
        hessian_x = 0.5 * (hessian_x + hessian_x.T)  # guarantee symmetry
        w, v = eigh(hessian_x)  # rot. and trans. degrees of freedom are removed
        w[absolute(w) < self.parameters.zero_thresh] = 0.0  # noise-cancelling
        w[(0.0 < w) &  # substitute small eigenvalues by lower limit
          (w < self.parameters.hessian_limit)] = self.parameters.hessian_limit
        # reconstruct Hessian from new eigenvalues and preserved eigenvectors
        hessian_x = v @ diagflat(w) @ v.T  # v.T == inv(v) due to symmetry
        self.hessian_x = 0.5 * (hessian_x + hessian_x.T)  # guarantee symmetry
        self.hessian_q = ijac0.T @ self.hessian_x @ ijac0

    @staticmethod
    def get_ijac(jac, rcond):  # jac is the Wilson B-matrix
        """Compute Moore-Penrose pseudo-inverse of Wilson B-matrix."""
        jac_T = jac.T  # btw. direct Jacobian inversion is slow, hence form Gmat
        Gmat = jac_T @ jac   # avoid: numpy.linalg.pinv(Gmat, rcond) @ jac_T
        ijac = lstsq(Gmat, jac_T, rcond, lapack_driver='gelsy')
        return ijac[0]  # [-1] would be eigenvalues of Gmat

    def calculate(self, atoms=None, properties=['energy', 'forces'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        if self.calculation_required(atoms, properties):
            super().calculate(atoms, properties, system_changes)

            q = self.get_q_from_x(atoms).ravel()

            if self.parameters.cartesian:
                x = atoms.get_positions().ravel()
                x0 = self.x0
                hessian_x = self.hessian_x

                if self.parameters.variable_orientation:
                    # determine x0 for present orientation
                    x0 = self.back_transform(x, q, self.q0, atoms.copy())
                    ref_atoms = atoms.copy()
                    ref_atoms.set_positions(x0.reshape(int(len(x0) / 3), 3),
                                         apply_constraint=False)
                    # determine jac0 for present orientation
                    jac0 = self.get_jacobian(ref_atoms)
                    self.check_redundancy(jac0)  # check for coordinate failure
                    # determine hessian_x for present orientation
                    hessian_x = jac0.T @ self.hessian_q @ jac0

                xdiff = x - x0
                forces_x = -hessian_x @ xdiff
                energy = (self.parameters.ref_energy
                          - 0.5 * (forces_x * xdiff).sum())

            else:
                jac = self.get_jacobian(atoms)
                self.check_redundancy(jac)  # check for coordinate failure
                qdiff = q - self.q0
                forces_q = -self.hessian_q @ qdiff
                forces_x = forces_q @ jac
                energy = (self.parameters.ref_energy
                          - 0.5 * (forces_q * qdiff).sum())

            forces_x = forces_x.reshape(int(forces_x.size / 3), 3)

            self.results['energy'] = energy
            self.results['forces'] = forces_x

    def back_transform(self, x, q, q0, atoms_copy):
        """Find the right orientation in Cartesian reference coordinates."""
        xk = 1 * x
        qk = 1 * q
        dq = qk - q0
        err = abs(dq).max()
        count = 0
        atoms_copy.set_constraint()  # helpful for back-transformation
        while err > 1e-7:  # back-transformation tolerance for convergence
            count += 1
            if count > 99:  # maximum number of iterations during back-transf.
                msg = ('Back-transformation from user-defined to Cartesian '
                       'coordinates failed.')
                raise CalculationFailed(msg)
            jac = self.get_jacobian(atoms_copy)
            ijac = self.get_ijac(jac, self.parameters.rcond)
            dx = ijac @ dq
            xk = xk - dx
            atoms_copy.set_positions(xk.reshape(int(len(xk) / 3), 3))
            qk = self.get_q_from_x(atoms_copy).ravel()
            dq = qk - q0
            err = abs(dq).max()
        return xk

    def check_redundancy(self, jac):
        """Compare number of zero eigenvalues of G-matrix to initial number."""
        Gmat = jac.T @ jac
        self.Gmat_eigvals, _ = eigh(Gmat)
        zero_eigvals = len(flatnonzero(absolute(self.Gmat_eigvals) <
                                       self.parameters.zero_thresh))
        if zero_eigvals != self.zero_eigvals:
            raise RuntimeError('Suspected coordinate failure: '
                               + 'G-matrix has got {} '.format(zero_eigvals)
                               + 'zero eigenvalues, but had '
                               + '{} during '.format(self.zero_eigvals)
                               + 'setup')

    def copy(self):
        """Create a new instance of the :class:`Harmonic` calculator with the
        same input parameters."""
        return Harmonic(**self.parameters)

    def todict(self):
        d = super().todict()  # when self.parameters is serialized, ...
        d.update(get_q_from_x=repr(self.parameters.get_q_from_x))  # functions
        d.update(get_jacobian=repr(self.parameters.get_jacobian))  # raise errs
        return d


class SpringCalculator(Calculator):
    """
    Spring calculator corresponding to independent oscillators with a fixed
    spring constant.


    Energy for an atom is given as

    E = k / 2 * (r - r_0)**2

    where k is the spring constant and, r_0 the ideal positions.


    Parameters
    ----------
    ideal_positions : array
        array of the ideal crystal positions
    k : float
        spring constant in eV/Angstrom
    """
    implemented_properties = ['forces', 'energy', 'free_energy']

    def __init__(self, ideal_positions, k):
        Calculator.__init__(self)
        self.ideal_positions = ideal_positions.copy()
        self.k = k

    def calculate(self, atoms=None, properties=['energy'],
                  system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(atoms)
        self.results['energy'], self.results['forces'] = energy, forces

    def compute_energy_and_forces(self, atoms):
        disps = atoms.positions - self.ideal_positions
        forces = - self.k * disps
        energy = sum(self.k / 2.0 * norm(disps, axis=1)**2)
        return energy, forces

    def get_free_energy(self, T, method='classical'):
        """Get analytic vibrational free energy for the spring system.

        Parameters
        ----------
        T : float
            temperature (K)
        method : str
            method for free energy computation; 'classical' or 'QM'.
        """
        F = 0.0
        masses, counts = unique(self.atoms.get_masses(), return_counts=True)
        for m, c in zip(masses, counts):
            F += c * SpringCalculator.compute_Einstein_solid_free_energy(self.k, m, T, method)
        return F

    @staticmethod
    def compute_Einstein_solid_free_energy(k, m, T, method='classical'):
        """ Get free energy (per atom) for an Einstein crystal.

        Free energy of a Einstein solid given by classical (1) or QM (2)
        1.    F_E = 3NkbT log( hw/kbT )
        2.    F_E = 3NkbT log( 1-exp(hw/kbT) ) + zeropoint

        Parameters
        -----------
        k : float
            spring constant (eV/A^2)
        m : float
            mass (grams/mole or AMU)
        T : float
            temperature (K)
        method : str
            method for free energy computation, classical or QM.

        Returns
        --------
        float
            free energy of the Einstein crystal (eV/atom)
        """
        assert method in ['classical', 'QM']

        hbar = units._hbar * units.J  # eV/s
        m = m / units.kg              # mass kg
        k = k * units.m**2 / units.J  # spring constant J/m2
        omega = sqrt(k / m)        # angular frequency 1/s

        if method == 'classical':
            F_einstein = 3 * units.kB * T * log(hbar * omega / (units.kB * T))
        elif method == 'QM':
            log_factor = log(1.0 - exp(-hbar * omega / (units.kB * T)))
            F_einstein = 3 * units.kB * T * log_factor + 1.5 * hbar * omega

        return F_einstein
