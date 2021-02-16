import numpy as np
from numpy.linalg import eigh

from ase.optimize.bfgs import BFGS
from ase.constraints import FixInternals
from ase.visualize import view


class ClimbFixInternals(BFGS):
    def __init__(self, atoms, restart=None, logfile='-', trajectory=None,
                 maxstep=None, master=None, alpha=None,
                 climb_coordinate=None,
                 optB=BFGS, optB_log=None, optB_kwargs=None, optB_fmax=0.05):
                 #auto_thresh=True, fixed_conv_ratio=0.8, max_interval_steps=3,
                 #interval_step=0.5, adaptive_thresh=0.6, linear_interpol=False,
                 #cubic=None):

        """Class for transition state optimization

        Climbs the 1D reaction coordinate defined as constrained internal
        coordinate via the :class:`~ase.constraints.FixInternals` class while
        minimizing all remaining degrees of freedom.

        Two optimizers are applied orthogonal to each other.
        Optimizer 'A' climbs the constrained coordinate while optimizer 'B'
        optimizes the remaining degrees of freedom after each climbing step.

        Other constraints are considered.

        Parameters:
        -----------
        atoms: :class:`~ase.Atoms` object
            The Atoms object to be optimized with a
            :class:`~ase.constraints.FixInternals` constraint attached.

        climb_coordinate: list
            Specifies which subconstraint of the
            :class:`~ase.constraints.FixInternals` should be climbed.
            Provide a list with 'constraint name' and corresponding
            indices (without coefficient in case of combo constraints).

     
        optB: any ASE optimizer
            Optimizer 'B' used for the optimization of all degrees of
            freedom but the constrained coordinate to be climbed.
            Default: :class:`~ase.optimize.BFGS`

        optB_log:
            Specifies logging of optimizer 'B'.

        optB_kwargs: dict
            Specifies arguments to be passed to optimizer 'B'.

        optB_fmax:
            Specifies `fmax` of optimizer 'B' during optimization of the
            remaining degrees of freedom.


        For further parameters see the parent class
        :class:`~ase.optimize.Optimizer`.
        """
        BFGS.__init__(self, atoms, restart, logfile, trajectory,
                      maxstep, master, alpha)

        self.constr2climb = self.get_constr2climb(self.atoms, climb_coordinate)
        self.targetvalue = self.constr2climb.targetvalue

        self.optB = optB
        self.optB_log = optB_log or '/optB_{}.log'.format(self.targetvalue)
        self.optB_kwargs = optB_kwargs or {}
        self.optB_fmax = optB_fmax

    def get_constr2climb(self, atoms, climb_coordinate):
        atoms.set_positions(atoms.get_positions())  # initialize FixInternals
        available_constraint_types = list(map(type, atoms.constraints))
        index = available_constraint_types.index(FixInternals)
        for subconstr in atoms.constraints[index].constraints:
            if repr(subconstr).startswith(climb_coordinate[0]):
                if subconstr.indices == climb_coordinate[1]:
                    return subconstr
        raise ValueError('Given `climb_coordinate` not found on Atoms.')

    def initialize(self):
        BFGS.initialize(self)
        self.projected_forces = None

    def read(self):
        (self.H, self.r0, self.f0, self.maxstep,
         self.projected_forces, self.targetvalue) = self.load()
        self.constr2climb.targetvalue = self.targetvalue  # update constr.

    def step(self, f=None):
        atoms = self.atoms
        f = self.get_projected_forces()  # initially computed during self.log()

        # similar to BFGS.step()
        r = atoms.get_positions()
        f = f.reshape(-1)
        self.update(r.flat, f, self.r0, self.f0)
        omega, V = eigh(self.H)
        dr = np.dot(V, np.dot(f, V) / np.fabs(omega)).reshape((-1, 3))
        steplengths = (dr**2).sum(1)**0.5
        dr = self.determine_step(dr, steplengths)

        self.constr2climb.adjust_positions(r, r+dr)  # update constr2climb.sigma
        self.targetvalue += self.constr2climb.sigma  # climb the constraint
        self.constr2climb.targetvalue = self.targetvalue  # adjust positions...
        atoms.set_positions(atoms.get_positions())        # ...to targetvalue

        self.r0 = r.flat.copy()
        self.f0 = f.copy()

        optB = self.optB(atoms, **self.optB_kwargs)  # optimize remaining...
        optB.run(self.optB_fmax)                     # ...degrees of freedom

        self.projected_forces = self.get_projected_forces()

        self.dump((self.H, self.r0, self.f0, self.maxstep,
                   self.projected_forces, self.targetvalue))

    def get_projected_forces(self):
        f = self.constr2climb.projected_force * self.constr2climb.jacobian
        f = f.reshape(self.atoms.get_positions().shape)
        return f

    def converged(self):  # converge projected_forces
        forces = self.projected_forces
        return BFGS.converged(self, forces=forces)

    def log(self):
        forces = self.projected_forces
        if forces is None:  # always log fmax(projected_forces)
            self.atoms.get_forces()  # compute projected_forces
            forces = self.get_projected_forces()
        BFGS.log(self, forces=forces)
