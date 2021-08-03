import warnings

import numpy as np
from ase.optimize.optimize import Optimizer
from numpy.linalg import eigh


def update_hessian(
    iterate: np.ndarray,
    iterate_previous: np.ndarray,
    gradient: np.ndarray,
    gradient_previous: np.ndarray,
    hessian: np.ndarray,
) -> np.ndarray:
    """Update hessian by BFGS algorithm

    REMARK: `gradient` currently means negative gradient, i.e., force

    Args:
        iterate: current state (e.g. positions)
        tterate_previous: state before last update (e.g. prev. positions)
        gradient: current gradient (e.g. forces)
        gradient_previous: last gradient
        hessian: current hessian

    Returns:
        new_hessian: update hessian
    """
    dx = iterate - iterate_previous
    df = gradient - gradient_previous

    a = np.dot(dx, df)
    dg = np.dot(hessian, dx)
    b = np.dot(dx, dg)
    new_hessian = hessian - np.outer(df, df) / a + np.outer(dg, dg) / b

    return new_hessian


class BFGSState:
    """dataclass handling the state of an BFGS algorithm incl. initial cond."""

    def __init__(
        self, iterate: np.ndarray, gradient: np.ndarray, hessian: np.ndarray,
    ) -> None:
        """initialize with a iterate (e.g. positions), gradient (e.g. forces),
           optionally hessian
        """
        self.iterate = iterate
        self.gradient = gradient
        self.hessian = hessian
        self._initial_state = self.todict()

    def todict(self) -> dict:
        d = {k: getattr(self, k) for k in ("iterate", "gradient", "hessian")}
        return d

    @property
    def initial_state(self):
        return self._initial_state.copy()


class BFGSOptimizer:
    default_configuration = {"alpha": 70.0}

    def __init__(
        self,
        iterate: np.ndarray,
        gradient: np.ndarray,
        alpha: float = None,
        hessian: np.ndarray = None,
    ) -> None:
        """initialize with a iterate (e.g. positions), gradient (e.g. forces),
           optionally hessian
        """
        ndim = np.size(iterate)
        if hessian is not None:
            assert np.shape(hessian) == (ndim, ndim), (np.shape(hessian), ndim)
        elif alpha is not None:
            hessian = np.eye(ndim) / alpha
        else:
            hessian = np.eye(ndim) / self.default_configuration["alpha"]

        # initialize state
        self.state = BFGSState(
            iterate=iterate, gradient=gradient, hessian=hessian
        )

    def get_step_and_update(self, iterate, gradient) -> np.ndarray:
        """Take iterate and gradient, update state, return predicted step"""
        # update hessian
        self.state.hessian = update_hessian(
            iterate=iterate,
            iterate_previous=self.state.iterate,
            gradient=gradient,
            gradient_previous=self.state.gradient,
            hessian=self.state.hessian,
        )
        # predict new iterate and update internal state
        omega, V = eigh(self.state.hessian)
        dx = np.dot(V, np.dot(gradient, V) / np.fabs(omega)).reshape((-1, 3))
        # update state
        self.state.iterate = iterate
        self.state.gradient = gradient

        return dx


# fmt: off
class BFGS(Optimizer):
    # default parameters
    defaults = {**Optimizer.defaults, 'alpha': 70.0}

    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None):
        """BFGS optimizer.

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        restart: string
            Pickle file used to store hessian matrix. If set, file with
            such a name will be searched and hessian matrix stored will
            be used, if the file exists.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        maxstep: float
            Used to set the maximum distance an atom can move per
            iteration (default value is 0.2 Å).

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.
        """
        if maxstep is None:
            self.maxstep = self.defaults['maxstep']
        else:
            self.maxstep = maxstep

        if self.maxstep > 1.0:
            warnings.warn('You are using a *very* large value for '
                          'the maximum step size: %.1f Å' % maxstep)

        if alpha is None:
            self.alpha = self.defaults['alpha']
        else:
            self.alpha = alpha

        Optimizer.__init__(self, atoms, restart, logfile, trajectory, master)

    def todict(self):
        d = Optimizer.todict(self)
        if hasattr(self, 'maxstep'):
            d.update(maxstep=self.maxstep)
        return d

    def initialize(self):
        # initial hessian
        self.H0 = np.eye(3 * len(self.atoms)) * self.alpha

        self.H = None
        self.r0 = None
        self.f0 = None

    def read(self):
        self.H, self.r0, self.f0, self.maxstep = self.load()

    def step(self, f=None):
        atoms = self.atoms

        if f is None:
            f = atoms.get_forces()

        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)
        omega, V = eigh(self.H)

        # FUTURE: Log this properly
        # # check for negative eigenvalues of the hessian
        # if any(omega < 0):
        #     n_negative = len(omega[omega < 0])
        #     msg = '\n** BFGS Hessian has {} negative eigenvalues.'.format(
        #         n_negative
        #     )
        #     print(msg, flush=True)
        #     if self.logfile is not None:
        #         self.logfile.write(msg)
        #         self.logfile.flush()

        dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)
        atoms.set_positions(r + dr)
        self.r0 = r.flat.copy()
        self.f0 = f.copy()
        self.dump((self.H, self.r0, self.f0, self.maxstep))

    def determine_step(self, dr, steplengths):
        """Determine step to take according to maxstep

        Normalize all steps as the largest step. This way
        we still move along the eigendirection.
        """
        maxsteplength = np.max(steplengths)
        if maxsteplength >= self.maxstep:
            scale = self.maxstep / maxsteplength
            # FUTURE: Log this properly
            # msg = '\n** scale step by {:.3f} to be shorter than {}'.format(
            #     scale, self.maxstep
            # )
            # print(msg, flush=True)

            dr *= scale

        return dr

    def update(self, r, f, r0, f0):
        if self.H is None:
            self.H = self.H0
            return
        dr = r - r0

        if np.abs(dr).max() < 1e-7:
            # Same configuration again (maybe a restart):
            return

        df = f - f0
        a = np.dot(dr, df)
        dg = np.dot(self.H, dr)
        b = np.dot(dr, dg)
        self.H -= np.outer(df, df) / a + np.outer(dg, dg) / b

    def replay_trajectory(self, traj):
        """Initialize hessian from old trajectory."""
        if isinstance(traj, str):
            from ase.io.trajectory import Trajectory
            traj = Trajectory(traj, 'r')
        self.H = None
        atoms = traj[0]
        r0 = atoms.get_positions().ravel()
        f0 = atoms.get_forces().ravel()
        for atoms in traj:
            r = atoms.get_positions().ravel()
            f = atoms.get_forces().ravel()
            self.update(r, f, r0, f0)
            r0 = r
            f0 = f

        self.r0 = r0
        self.f0 = f0


class oldBFGS(BFGS):
    def determine_step(self, dr, steplengths):
        """Old BFGS behaviour for scaling step lengths

        This keeps the behaviour of truncating individual steps. Some might
        depend of this as some absurd kind of stimulated annealing to find the
        global minimum.
        """
        dr /= np.maximum(steplengths / self.maxstep, 1.0).reshape(-1, 1)
        return dr
