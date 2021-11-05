.. module:: ase.calculators.harmonic

Harmonic calculator
===================

.. autoclass:: Harmonic

   .. automethod:: copy

.. note::

   The reference Hessians in **x** and **q** can be inspected on the calculator
   attributes ``hessian_x`` and ``hessian_q``.

Examples
========
Prerequisites: :class:`~ase.Atoms` object (``ref_atoms``),
its energy (``ref_energy``) and Hessian (``hessian_x``).

Example 1: Cartesian coordinatates
----------------------------------
In Cartesian coordinates, forces and energy are not invariant with respect
to rotations and translations of the system.
>>> from ase.calculators.harmonic import Harmonic
>>> calc_harmonic = Harmonic(ref_atoms=ref_atoms,
...                          ref_energy=ref_energy,
...                          hessian_x=hessian_x)
>>> atoms = ref_atoms.copy()
>>> atoms.calc = calc_harmonic

.. note::

   Forces and energy can be computed via :meth:`~ase.Atoms.get_forces` and
   :meth:`~ase.Atoms.get_potential_energy` for any configuration that does
   not involve rotations with respect to the configuration of ``ref_atoms``.
   In case of system rotations, Cartesian coordinates return incorrect values
   and thus cannot be used without an additional suitable coordinate system
   as demonstrated in the Supporting Information of [1]_.

Example 2: Internal Coordinates
-------------------------------
To compute forces and energy correctly even for rotated systems,
a user-defined coordinate system must be provided.
Within this coordinate system, energy and forces must be invariant with
respect to rotations and translations of the system.
For this purpose internal coordinates (distances, angles, dihedrals,
coordination numbers and linear combinations thereof, etc.) are widely used.
The following example deals with the water molecule (H2O).

>>> import numpy as np
>>> from ase.geometry.geometry import get_distances_derivatives
>>> dist_defs = [[0, 1], [1, 2], [2, 0]]  # define three distances by atom indices
>>> def water_get_q_from_x(atoms):
...     """Simple internal coordinates to describe water with only distances."""
...     q_vec = [atoms.get_distance(i, j) for i, j in dist_defs]
...     return np.asarray(q_vec)  # returns a vector with internal coordinates
>>> def water_get_jacobian(atoms):
...     """Function to return the Jacobian for the water molecule, i.e. the
...     Cartesian derivatives of the above defined internal coordinates."""
...     pos = atoms.get_positions()
...     dist_vecs = [pos[j] - pos[i] for i, j in dist_defs]
...     derivs = get_distances_derivatives(dist_vecs)
...     jac = []
...     for i, defin in enumerate(dist_defs):
...         dqi_dxj = np.zeros(ref_pos.shape)
...         for j, deriv in enumerate(derivs[i]):
...             dqi_dxj[defin[j]] = deriv
...         jac.append(dqi_dxj.flatten())
...     return np.asarray(jac)  # returns a matrix with Cartesian derivatives
>>> calc_harmonic = Harmonic(ref_atoms=ref_atoms,
...                          ref_energy=ref_energy,
...                          hessian_x=hessian_x,
...                          get_q_from_x=water_get_q_from_x,
...                          get_jacobian=water_get_jacobian,
...                          cartesian=False)
>>> atoms = ref_atoms.copy()
>>> atoms.calc = calc_harmonic

Example 3
---------
A transformation of the coordinate system may transform the force
field. The change in free energy due to this transformation can be computed via
thermodynamic (`\lambda`-path) integration, see [1]_.

>>> from ase.calculators.mixing import MixedCalculator
>>> from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution
...                                          Stationary, ZeroRotation)
>>> from ase.md.andersen import Andersen
>>> calc_harmonic_1 = Harmonic(ref_atoms=ref_atoms,
...                            ref_energy=ref_energy,
...                            hessian_x=hessian_x,
...                            get_q_from_x=water_get_q_from_x,
...                            get_jacobian=water_get_jacobian,
...                            cartesian=True, variable_orientation=True)
>>> calc_harmonic_0 = calc_harmonic_one.copy()
>>> calc.harmonic_0.set('cartesian'=False)
>>> ediffs = {}  # collect energy difference for varying lambda coupling
>>> for lamb in [0.00, 0.25, 0.50, 0.75, 1.00]:  # integration grid
...     ediffs[lamb] = []
...     calc_linearCombi = MixedCalculator(calc_harmonic_0, calc_harmonic_1,
...                                        1 - lamb, lamb)
...     atoms = ref_atoms.copy()
...     atoms.calc = calc_linearCombi
...     MaxwellBoltzmannDistribution(atoms, temperature_K=300, force_temp=True)
...     Stationary(atoms)
...     ZeroRotation(atoms)
...     dyn = Andersen(atoms, 0.5 * ase.units.fs, temperature_K=300,
...                    andersen_prob=0.05, fixcm=False)
...     for _ in dyn.irun(100000):
...         e0, e1 = calc.get_energy_contributions(atoms)
...         ediffs[lamb].append(float(e1) - float(e0))
...     ediffs[lamb] = sum(ediffs[lamb]) / len(ediffs[lamb])  # mean

Now integrate the mean energy differences ('ediffs') over the integration grid
(`\lambda` path) to obtain the change in free energy due to the coordinate
transformation.

.. note::

   Remember to compute the Harmonic Approximation from the reference Hessian
   which is generated during initialization of the :class:`Harmonic`
   calculator. In other words, evaluate the frequencies obtained from

    >>> vib = Vibrations(atoms, nfree=2, delta=1e-8)
    >>> vib.run()

   The total free energy is the sum of the contributions from the Harmonic
   Approximation for the reference Hessian (frequencies) and from the
   `\lambda`-path integration.
   Compare this total free energy to the free energy contribution by the
   standard Harmonic Approximation obtained from the unmodified Hessian.

Example 4
---------
Compute the Anharmonic Correction to the Harmonic Approximation.

>>> calc_harmonic = Harmonic(ref_atoms=ref_atoms,
...                          ref_energy=ref_energy,
...                          hessian_x=hessian_x,
...                          get_q_from_x=ic.create,
...                          get_jacobian=ic.get_jacobian,
...                          cartesian=False)
>>> calc_anharmonic = Vasp()  # can do interactions beyong Harm. Approx.

Now perform thermodynamic integration over the `\lambda`-path similar to
Example 3 using the :class:`~ase.calculators.mixing.MixedCalculator` with
'calc_harmonic' and 'calc_anharmonic' to obtain the Anharmonic Correction.

Theory for Anharmonic Correction via Thermodynamic Integration (TI)
===================================================================
Thermodynamic integration (TI), i.e. `\lambda`-path integration,
connects two thermodynamic states via a `\lambda`-path.
We begin the TI from a reference system '0' with known free energy (here the
Harmonic Approximation) and obtain the Anharmonic Correction via integration
over the `\lambda`-path to the target system '1' (the fully interacting
anharmonic system).
Hence, the free energy of the target system can be written as

.. math::
    A_1 = A_0 + \Delta A_{0 \rightarrow 1}

where the second term corresponds to the integral over the `\lambda`-path

.. math::

    \Delta A_{0 \rightarrow 1} = \int_0^1 d \lambda
    \langle H_1 - H_0 \rangle_\lambda

The term `\langle ... \rangle_\lambda` represents the NVT ensemble
average of the system driven by the classical Hamiltonian
`\mathcal{H}_\lambda` determined by the coupling parameter
`\lambda \in [0,1]`

.. math::

    \mathcal{H}_\lambda = \lambda \mathcal{H}_1 + (1 - \lambda) \mathcal{H}_0

Since the Hamiltonians differ only in their potential energy contributions
`V_1` and `V_0`, the free energy change can be computed from the
potentials

.. math::

    \Delta A_{0 \rightarrow 1} = \int_0^1 d \lambda
    \langle V_1 - V_0 \rangle_\lambda

The Cartesian coordinates **x** used in the common Harmonic Approximation are
not insensitive to overall rotations and translations that must leave the total
energy invariant.
This limitation can be overcome by transformation of the Hessian in **x**
to a suitable coordinate system **q** (e.g. internal coordinates).
Since the force field of that Hessian which is harmonic in **x** is not
necessarily equivalently harmonic in **q**, the free energy correction can be
rewritten to

.. math::
    A_1 = A_{0,\mathbf{x}} + \Delta A_{0,\mathbf{x} \rightarrow 0,\mathbf{q}}
    + \Delta A_{0,\mathbf{q} \rightarrow 1}

The terms in this equation correspond to the free energy from the Harmonic
Approximation with the reference Hessian (`A_{0,\mathbf{x}}`), the free
energy change due to the coordinate transformation
(`\Delta A_{0,\mathbf{x} \rightarrow 0,\mathbf{q}}`) obtained via TI
(see Example 3) and the free energy change from the harmonic to the fully
interacting system (`\Delta A_{0,\mathbf{q} \rightarrow 1}`) obtained via
TI (see Example 4).
Please see Amsler, J. et al. for details. [1]_

.. note::

    Anharmonicity is quantified by comparison of the total free energy
    `A_1` to the free energy contributions by the standard Harmonic
    Approximation with the unmodified Hessian.
    The reference Hessian and its free energy contribution
    `A_{0,\mathbf{x}}` have no meaning outside the TI procedure.
