.. module:: ase.calculators.harmonic

Harmonic calculator
===================

.. autoclass:: Harmonic

   .. automethod:: copy

Examples
========
Prerequisites: :class:`~ase.Atoms` object (``ref_atoms``),
its energy (``ref_energy``) and Hessian (``hessian_x``).

Example 1: Use Cartesian coordinatates
--------------------------------------
>>> from ase.calculators.harmonic import Harmonic
>>> calc_harmonic = Harmonic(atoms0=atoms0,
...                          energy0=energy0,
...                          hessian_x=hessian_x)
>>> atoms = atoms0.copy()
>>> atoms.calc = calc_harmonic

.. note::

   Forces and energy can be computed via :meth:`~ase.Atoms.get_forces` and
   :meth:`~ase.Atoms.get_potential_energy` for any configuration that does
   not involve rotations with respect to the configuration of ``atoms0``.
   In case of system rotations, Cartesian coordinates return incorrect values
   and thus cannot be used as demonstrated in the Supporting Information of
   [1]_.
