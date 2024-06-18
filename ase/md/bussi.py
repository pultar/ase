"""Bussi NVT dynamics class."""

import math

import numpy as np

from ase import units
from ase.md.md import MolecularDynamics
from ase.parallel import world


class Bussi(MolecularDynamics):
    """Bussi stochastic velocity rescaling (NVT) molecular dynamics.
    Based on the paper from Bussi et al. (https://arxiv.org/abs/0803.4060)

    Parameters
    ----------
    atoms: Atoms object
        The atoms object.
    timestep: float
        The time step in ASE time units.
    temperature_K: float
        The desired temperature, in Kelvin.
    taut: float
        Time constant for Bussi temperature coupling in ASE time units.
    rng: numpy.random.Generator (default np.random.default_rng())
        Random number generator.
    md_kwargs: dict
        Additional arguments passed to MolecularDynamics base class.
    """

    def __init__(
        self,
        atoms,
        timestep,
        temperature_K,
        taut,
        rng=np.random.default_rng(),
        **md_kwargs,
    ):
        super().__init__(
            atoms,
            timestep,
            **md_kwargs,
        )

        self.temp = temperature_K * units.kB
        self.taut = taut
        self.communicator = world
        self.rng = rng

        self.ndof = self.atoms.get_number_of_degrees_of_freedom()

        self.target_kinetic_energy = 0.5 * self.temp * self.ndof

        if np.isclose(
            self.atoms.get_kinetic_energy(), 0.0, rtol=0, atol=1e-12
        ):
            raise ValueError(
                "Initial kinetic energy is zero. "
                "Please set the initial velocities before running Bussi NVT."
            )

        self._exp_term = math.exp(-self.dt / self.taut)
        self._masses = self.atoms.get_masses()[:, np.newaxis]

        self.transferred_energy = 0.0

    def scale_velocities(self):
        """Do the NVT Bussi stochastic velocity scaling."""
        kinetic_energy = self.atoms.get_kinetic_energy()
        alpha = self.calculate_alpha(kinetic_energy)

        momenta = self.atoms.get_momenta()
        self.atoms.set_momenta(alpha * momenta)

        self.transferred_energy += (alpha**2 - 1.0) * kinetic_energy

    def calculate_alpha(self, kinetic_energy):
        """Calculate the scaling factor alpha using equation (A7)
        from the Bussi paper."""

        energy_scaling_term = (
            (1 - self._exp_term)
            * self.target_kinetic_energy
            / kinetic_energy
            / self.ndof
        )

        normal_noise = self.rng.standard_normal()
        sum_of_noises = 2.0 * self.rng.standard_gamma(0.5 * (self.ndof - 1))

        return math.sqrt(
            self._exp_term
            + energy_scaling_term * (sum_of_noises + normal_noise**2)
            + 2
            * normal_noise
            * math.sqrt(self._exp_term * energy_scaling_term)
        )

    def step(self, forces=None):
        """Move one timestep forward using Bussi NVT molecular dynamics."""
        if forces is None:
            forces = self.atoms.get_forces(md=True)

        self.atoms.set_momenta(
            self.atoms.get_momenta() + 0.5 * self.dt * forces
        )
        momenta = self.atoms.get_momenta()

        self.atoms.set_positions(
            self.atoms.positions
            + self.dt * momenta / self._masses
        )

        forces = self.atoms.get_forces(md=True)

        self.atoms.set_momenta(momenta + 0.5 * self.dt * forces)

        self.scale_velocities()

        return forces
