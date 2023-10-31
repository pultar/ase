from typing import IO, Optional, Union

import numpy as np
import scipy.optimize as opt

from ase import Atoms
from ase.optimize.optimize import Optimizer


class Converged(Exception):
    pass


class OptimizerConvergenceError(Exception):
    pass


class SciPyOptimizer(Optimizer):
    """General interface for SciPy optimizers

    Only the call to the optimizer is still needed
    """

    def __init__(
        self,
        atoms: Atoms,
        logfile: Union[IO, str] = '-',
        trajectory: Optional[str] = None,
        callback_always: bool = False,
        alpha: float = 70.0,
        master: Optional[bool] = None,
        force_consistent: Optional[bool] = None,
    ):
        """Initialize object

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or string
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        callback_always: boolean
            Should the callback be run after each force call (also in the
            linesearch)

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        
        method: string or callable object or None
            Defaults to 'BFGS'. If a string is supplied, it must correspond
            to a valid value for the `method` parameter of 
            `scipy.optimize.minimize`. If a callable object is supplied, it
            will be called as `method(fun, x0, args, **kwargs, **options)`
            where `fun = self.f`, `x0 = self.x0()`, `args = ()`,
            `options = self.options`, and `kwargs` contains the remaining
            keyword arguments passed to `scipy.optimizer.minimize` in
            `SciPyOptimizer.run`.
        options: dictionary or None
            Defaults to an empty dictionary. Specify any additional keyword
            arguments to pass to the function specified by `method`.
        """
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)
        self.force_calls = 0
        self.callback_always = callback_always
        self.H0 = alpha
        self.method = method or 'BFGS'
        self.options = options or {}

    def x0(self):
        """Return x0 in a way SciPy can use

        This class is mostly usable for subclasses wanting to redefine the
        parameters (and the objective function)"""
        return self.optimizable.get_positions().reshape(-1)

    def f(self, x):
        """Objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        # Scale the problem as SciPy uses I as initial Hessian.
        return (self.optimizable.get_potential_energy(
                force_consistent=self.force_consistent) / self.H0)

    def fprime(self, x):
        """Gradient of the objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.force_calls += 1

        if self.callback_always:
            self.callback(x)

        # Remember that forces are minus the gradient!
        # Scale the problem as SciPy uses I as initial Hessian.
        return - self.optimizable.get_forces().reshape(-1) / self.H0

    def callback(self, x):
        """Callback function to be run after each iteration by SciPy

        This should also be called once before optimization starts, as SciPy
        optimizers only calls it after each iteration, while ase optimizers
        call something similar before as well.

        :meth:`callback`() can raise a :exc:`Converged` exception to signal the
        optimisation is complete. This will be silently ignored by
        :meth:`run`().
        """
        f = self.optimizable.get_forces()
        self.log(f)
        self.call_observers()
        if self.converged(f):
            raise Converged
        self.nsteps += 1

    def run(self, fmax=0.05, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.fmax = fmax
        try:
            # As SciPy does not log the zeroth iteration, we do that manually
            self.callback(None)
            # Scale the problem as SciPy uses I as initial Hessian.
            self.call_fmin(fmax / self.H0, steps)
        except Converged:
            pass
        return self.converged()

    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, fmax, steps):
        if self.method != 'TNC':
            self.options['maxiter'] = steps
        if 'gtol' in self.options and self.options['gtol'] is None:
            self.options['gtol'] = fmax * 0.1
            opt.fmin_cg

        output = opt.minimize(
            fun=self.f,
            x0=self.x0(),
            method=self.method,
            jac=self.fprime,
            callback=self.callback,
            options=self.options
        )
        if not output.success:
            if output.message != 'Maximum number of iterations has been exceeded.':
                raise OptimizerConvergenceError(
                    f'Warning: Desired error not necessarily achieved: {output.message}'
                    )


class SciPyFminCG(SciPyOptimizer):
    """Non-linear (Polak-Ribiere) conjugate gradient algorithm"""

    def __init__(self,
                 atoms,
                 logfile='-',
                 trajectory=None,
                 callback_always=False,
                 alpha=70,
                 master=None,
                 force_consistent=None,
                 gtol = None,
                 disp = False,
                 norm = np.inf):
        options = {
            'gtol': gtol,
            'norm': norm,
            'disp': disp,
        }
        super().__init__(atoms,
                         logfile,
                         trajectory,
                         callback_always,
                         alpha,
                         master,
                         force_consistent,
                         'CG',
                         options)


class SciPyFminBFGS(SciPyOptimizer):
    """Quasi-Newton method (Broydon-Fletcher-Goldfarb-Shanno)"""

    def __init__(self,
                 atoms,
                 logfile='-',
                 trajectory=None,
                 callback_always=False,
                 alpha=70,
                 master=None,
                 force_consistent=None,
                 gtol = None,
                 disp = False,
                 norm = np.inf):
        options = {
            'gtol': gtol,
            'norm': norm,
            'disp': disp,
        }
        super().__init__(atoms,
                         logfile,
                         trajectory,
                         callback_always,
                         alpha,
                         master,
                         force_consistent,
                         'BFGS',
                         options)


class SciPyGradientlessOptimizer(Optimizer):
    """General interface for gradient less SciPy optimizers

    Only the call to the optimizer is still needed

    Note: If you redefine x0() and f(), you don't even need an atoms object.
    Redefining these also allows you to specify an arbitrary objective
    function.

    XXX: This is still a work in progress
    """

    def __init__(self, atoms, logfile='-', trajectory=None,
                 callback_always=False, master=None,
                 force_consistent=None):
        """Initialize object

        Parameters:

        atoms: Atoms object
            The Atoms object to relax.

        trajectory: string
            Pickle file used to store trajectory of atomic movement.

        logfile: file object or str
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        callback_always: book
            Should the callback be run after each force call (also in the
            linesearch)

        alpha: float
            Initial guess for the Hessian (curvature of energy surface). A
            conservative value of 70.0 is the default, but number of needed
            steps to converge might be less if a lower value is used. However,
            a lower value also means risk of instability.

        master: boolean
            Defaults to None, which causes only rank 0 to save files.  If
            set to true,  this rank will save files.

        force_consistent: boolean or None
            Use force-consistent energy calls (as opposed to the energy
            extrapolated to 0 K).  By default (force_consistent=None) uses
            force-consistent energies if available in the calculator, but
            falls back to force_consistent=False if not.
        """
        restart = None
        Optimizer.__init__(self, atoms, restart, logfile, trajectory,
                           master, force_consistent=force_consistent)
        self.function_calls = 0
        self.callback_always = callback_always

    def x0(self):
        """Return x0 in a way SciPy can use

        This class is mostly usable for subclasses wanting to redefine the
        parameters (and the objective function)"""
        return self.optimizable.get_positions().reshape(-1)

    def f(self, x):
        """Objective function for use of the optimizers"""
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.function_calls += 1
        # Scale the problem as SciPy uses I as initial Hessian.
        return self.optimizable.get_potential_energy(
            force_consistent=self.force_consistent)

    def callback(self, x):
        """Callback function to be run after each iteration by SciPy

        This should also be called once before optimization starts, as SciPy
        optimizers only calls it after each iteration, while ase optimizers
        call something similar before as well.
        """
        # We can't assume that forces are available!
        # f = self.optimizable.get_forces()
        # self.log(f)
        self.call_observers()
        # if self.converged(f):
        #    raise Converged
        self.nsteps += 1

    def run(self, ftol=0.01, xtol=0.01, steps=100000000):
        if self.force_consistent is None:
            self.set_force_consistent()
        self.xtol = xtol
        self.ftol = ftol
        # As SciPy does not log the zeroth iteration, we do that manually
        self.callback(None)
        try:
            # Scale the problem as SciPy uses I as initial Hessian.
            self.call_fmin(xtol, ftol, steps)
        except Converged:
            pass
        return self.converged()

    def dump(self, data):
        pass

    def load(self):
        pass

    def call_fmin(self, xtol, ftol, steps):
        raise NotImplementedError


class SciPyFmin(SciPyGradientlessOptimizer):
    """Nelder-Mead Simplex algorithm

    Uses only function calls.

    XXX: This is still a work in progress
    """

    def call_fmin(self, xtol, ftol, steps):
        opt.fmin(self.f,
                 self.x0(),
                 # args=(),
                 xtol=xtol,
                 ftol=ftol,
                 maxiter=steps,
                 # maxfun=None,
                 # full_output=1,
                 disp=0,
                 # retall=0,
                 callback=self.callback)


class SciPyFminPowell(SciPyGradientlessOptimizer):
    """Powell's (modified) level set method

    Uses only function calls.

    XXX: This is still a work in progress
    """

    def __init__(self, *args, **kwargs):
        """Parameters:

        direc: float
            How much to change x to initially. Defaults to 0.04.
        """
        direc = kwargs.pop('direc', None)
        SciPyGradientlessOptimizer.__init__(self, *args, **kwargs)

        if direc is None:
            self.direc = np.eye(len(self.x0()), dtype=float) * 0.04
        else:
            self.direc = np.eye(len(self.x0()), dtype=float) * direc

    def call_fmin(self, xtol, ftol, steps):
        opt.fmin_powell(self.f,
                        self.x0(),
                        # args=(),
                        xtol=xtol,
                        ftol=ftol,
                        maxiter=steps,
                        # maxfun=None,
                        # full_output=1,
                        disp=0,
                        # retall=0,
                        callback=self.callback,
                        direc=self.direc)
