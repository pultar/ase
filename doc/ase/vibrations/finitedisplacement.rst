=========================
Finite-displacement tools
=========================

While :mod:`~ase.vibrations.vibrations` provides an object-oriented
approach to Vibration calculations,
:mod:`~ase.vibrations.finite_diff` provides tools for
performing similar workflows in separate steps that may be customised,
automated or task-farmed.

Fundamentally the approach is always:

  - From an optimised Atoms, generate a set of distorted structures by
    perturbing each Atom in x, y, z directions.

    - These are generally represented as a collection of Atoms
      objects, but may also be stored as a collection of structure
      files or added to a database.
  - Calculate forces for each distorted structure and store these with
    the structures.

    - In a Python session this is naturally represented as a
      collection of Atoms objects with attached Calculators. Where the
      "live" Calculator is unavailable (e.g. the corresponding DFT
      code lives on another computer) the
      :class:`ase.calculators.singlepointcalculator.SinglePointCalculator`
      can be used to attach precomputed forces to an Atoms.

  - Fit these displacements with forces to a model Hessian, using the
    optimised reference structure (and, optionally, the residual forces on the
    reference structure).

    - The Hessian is represented by
      :class:`ase.vibrations.data.VibrationsData` and can be saved to
      JSON if necessary.

    - Each structure is matched to the displacement of one atom along
      a Cartesian axis by comparison with the reference
      structure. This means that it is not necessary to keep track of
      displacement labels, or the original displacement parameters:
      it can all be inferred from the Atoms objects.

    - Least-squares fitting is performed over the available structures
      for each degree of freedom (i.e. [atom, direction]). This means
      that forward-difference, central-difference and
      multiple-distance schemes can be accommodated equally. For
      e.g. a 4-point sampling scheme, generate two sets of
      central-difference displacements at different distances, then
      fit them simultaneously.

Example 1: Interactive workflow
-------------------------------

This example uses the :class:`ase.calculators.mopac.MOPAC`
Calculator. You can use another calculator if MOPAC is unavailable,
but it is Free Software, runs quickly and produces reasonable
qualitative results for these organic molecules.

.. literalinclude:: displacements_interactive.py

      
.. automodule:: ase.vibrations.finite_diff
