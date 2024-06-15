"""Berendsen NPT dynamics class."""
import warnings
from typing import IO, Optional, Union

import numpy as np

from ase import Atoms, units
from ase.md.nvtberendsen import NVTBerendsen


class NPTBerendsen(NVTBerendsen):
    def __init__(
        self,
        atoms: Atoms,
        timestep: float,
        temperature: float,
        pressure: float,
        taut: float = 0.5e3 * units.fs,
        taup: float = 1e3 * units.fs,
        compressibility: Optional[float] = None,
        fixcm: bool = True,
        **md_kwargs,
    ):
        """Berendsen (constant N, P, T) molecular dynamics.

        This dynamics scale the velocities and volumes to maintain a constant
        pressure and temperature.  The shape of the simulation cell is not
        altered, if that is desired use Inhomogenous_NPTBerendsen.

        Parameters
        ----------

        atoms: Atoms object
            The list of atoms.

        timestep: float
            The time step in ASE time units.

        temperature: float
            The desired temperature, in Kelvin.

        pressure: float
            The desired pressure, in atomic units (eV/Å^3).

        taut: float
            Time constant for Berendsen temperature coupling in ASE
            time units.  Default: 0.5 ps.

        taup: float
            Time constant for Berendsen pressure coupling.  Default: 1 ps.

        compressibility: float
            The compressibility of the material, in atomic units (Å^3/eV).

        fixcm: bool (optional)
            If True, the position and momentum of the center of mass is
            kept unperturbed.  Default: True.
        """

        NVTBerendsen.__init__(self, atoms, timestep, temperature=temperature,
                              taut=taut, fixcm=fixcm, **md_kwargs)

        self.taup = taup
        self.pressure = pressure
        self.compressibility = compressibility

    def set_taup(self, taup):
        self.taup = taup

    def get_taup(self):
        return self.taup

    def set_pressure(self, pressure):
        self.pressure = pressure

    def get_pressure(self):
        return self.pressure

    def set_compressibility(self, compressibility):
        self.compressibility = compressibility

    def get_compressibility(self):
        return self.compressibility

    def set_timestep(self, timestep):
        self.dt = timestep

    def get_timestep(self):
        return self.dt

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""

        taupscl = self.dt / self.taup
        stress = self.atoms.get_stress(voigt=False, include_ideal_gas=True)
        old_pressure = -stress.trace() / 3
        scl_pressure = (1.0 - taupscl * self.compressibility / 3.0 *
                        (self.pressure - old_pressure))

        cell = self.atoms.get_cell()
        cell = scl_pressure * cell
        self.atoms.set_cell(cell, scale_atoms=True)

    def step(self, forces=None):
        """ move one timestep forward using Berenden NPT molecular dynamics."""

        NVTBerendsen.scale_velocities(self)
        self.scale_positions_and_cell()

        # one step velocity verlet
        atoms = self.atoms

        if forces is None:
            forces = atoms.get_forces(md=True)

        p = self.atoms.get_momenta()
        p += 0.5 * self.dt * forces

        if self.fix_com:
            # calculate the center of mass
            # momentum and subtract it
            psum = p.sum(axis=0) / float(len(p))
            p = p - psum

        self.atoms.set_positions(
            self.atoms.get_positions() +
            self.dt * p / self.atoms.get_masses()[:, np.newaxis])

        # We need to store the momenta on the atoms before calculating
        # the forces, as in a parallel Asap calculation atoms may
        # migrate during force calculations, and the momenta need to
        # migrate along with the atoms.  For the same reason, we
        # cannot use self.masses in the line above.

        self.atoms.set_momenta(p)
        forces = self.atoms.get_forces(md=True)
        atoms.set_momenta(self.atoms.get_momenta() + 0.5 * self.dt * forces)

        return forces

class Inhomogeneous_NPTBerendsen(NPTBerendsen):
    """Berendsen (constant N, P, T) molecular dynamics.

    This dynamics scale the velocities and volumes to maintain a constant
    pressure and temperature.  The size of the unit cell is allowed to change
    independently in the three directions, but the angles remain constant.

    Usage: NPTBerendsen(atoms, timestep, temperature, taut, pressure, taup)

    Parameters
    ----------
    mask : tuple[int]
        Specifies which axes participate in the barostat.  Default (1, 1, 1)
        means that all axes participate, set any of them to zero to disable
        the barostat in that direction.
    """

    def __init__(self, *args, mask=(1, 1, 1), **kwargs):
        NPTBerendsen.__init__(self, *args, **kwargs)
        self.mask = mask

    def scale_positions_and_cell(self):
        """ Do the Berendsen pressure coupling,
        scale the atom position and the simulation cell."""

        taupscl = self.dt * self.compressibility / self.taup / 3.0
        stress = - self.atoms.get_stress(include_ideal_gas=True)
        if stress.shape == (6,):
            stress = stress[:3]
        elif stress.shape == (3, 3):
            stress = [stress[i][i] for i in range(3)]
        else:
            raise ValueError('Cannot use a stress tensor of shape ' +
                             str(stress.shape))
        pbc = self.atoms.get_pbc()
        scl_pressurex = 1.0 - taupscl * (self.pressure - stress[0]) \
            * pbc[0] * self.mask[0]
        scl_pressurey = 1.0 - taupscl * (self.pressure - stress[1]) \
            * pbc[1] * self.mask[1]
        scl_pressurez = 1.0 - taupscl * (self.pressure - stress[2]) \
            * pbc[2] * self.mask[2]
        cell = self.atoms.get_cell()
        cell = np.array([scl_pressurex * cell[0],
                         scl_pressurey * cell[1],
                         scl_pressurez * cell[2]])
        self.atoms.set_cell(cell, scale_atoms=True)
