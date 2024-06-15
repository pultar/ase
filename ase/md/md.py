"""Molecular Dynamics."""
import warnings
from typing import IO, Optional, Union

import numpy as np

from ase import Atoms, units
from ase.io.trajectory import Trajectory
from ase.md.logger import MDLogger
from ase.optimize.optimize import Dynamics

class MolecularDynamics(Dynamics):
    """Base-class for all MD classes."""

    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        trajectory: Optional[str] = None,
        logfile: Optional[Union[IO, str]] = None,
        loginterval: int = 1,
        append_trajectory: bool = False,
    ):
        """Molecular Dynamics object.

        Parameters:

        atoms: Atoms object
            The Atoms object to operate on.

        timestep: float
            The time step in ASE time units.

        trajectory: Trajectory object or str
            Attach trajectory object.  If *trajectory* is a string a
            Trajectory will be constructed.  Use *None* for no
            trajectory.

        logfile: file object or str (optional)
            If *logfile* is a string, a file with that name will be opened.
            Use '-' for stdout.

        loginterval: int (optional)
            Only write a log line for every *loginterval* time steps.
            Default: 1

        append_trajectory: boolean (optional)
            Defaults to False, which causes the trajectory file to be
            overwriten each time the dynamics is restarted from scratch.
            If True, the new structures are appended to the trajectory
            file instead.
        """
        if timestep is None:
            raise TypeError('Missing timestep argument')

        self.dt = timestep

        super().__init__(atoms, logfile=None, trajectory=None)

        # Some codes (e.g. Asap) may be using filters to
        # constrain atoms or do other things.  Current state of the art
        # is that "atoms" must be either Atoms or Filter in order to
        # work with dynamics.
        #
        # In the future, we should either use a special role interface
        # for MD, or we should ensure that the input is *always* a Filter.
        # That way we won't need to test multiple cases.  Currently,
        # we do not test /any/ kind of MD with any kind of Filter in ASE.
        self.atoms = atoms
        self.masses = self.atoms.get_masses()

        if 0 in self.masses:
            warnings.warn('Zero mass encountered in atoms; this will '
                          'likely lead to errors if the massless atoms '
                          'are unconstrained.')

        self.masses.shape = (-1, 1)

        if not self.atoms.has('momenta'):
            self.atoms.set_momenta(np.zeros([len(self.atoms), 3]))

        # Trajectory is attached here instead of in Dynamics.__init__
        # to respect the loginterval argument.
        if trajectory is not None:
            if isinstance(trajectory, str):
                mode = "a" if append_trajectory else "w"
                trajectory = self.closelater(
                    Trajectory(trajectory, mode=mode, atoms=atoms)
                )
            self.attach(trajectory, interval=loginterval)

        if logfile:
            logger = self.closelater(
                MDLogger(dyn=self, atoms=atoms, logfile=logfile))
            self.attach(logger, loginterval)

    def todict(self):
        return {'type': 'molecular-dynamics',
                'md-type': self.__class__.__name__,
                'timestep': self.dt}

    def irun(self, steps=50):
        """Run molecular dynamics algorithm as a generator.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of molecular dynamics steps to be run.

        Yields
        ------
        converged : bool
            True if the maximum number of steps are reached.
        """
        return Dynamics.irun(self, steps=steps)

    def run(self, steps=50):
        """Run molecular dynamics algorithm.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of molecular dynamics steps to be run.

        Returns
        -------
        converged : bool
            True if the maximum number of steps are reached.
        """
        return Dynamics.run(self, steps=steps)

    def get_time(self):
        return self.nsteps * self.dt

    def converged(self):
        """ MD is 'converged' when number of maximum steps is reached. """
        return self.nsteps >= self.max_steps

    def _get_com_velocity(self, velocity):
        """Return the center of mass velocity.
        Internal use only. This function can be reimplemented by Asap.
        """
        return np.dot(self.masses.ravel(), velocity) / self.masses.sum()
