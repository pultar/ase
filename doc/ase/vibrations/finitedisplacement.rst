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

The output *ethanol_mopac_vibs.xyz* file can be visualised with JMOL.
Note that broadly the vibrations around 1000-1500 cm-1 are bending
modes whereas the modes above 2500 cm-1 are characterised by bond
stretching.

Example 2: File-based workflow
------------------------------

In this example, displacements are written to files, forces are
calculated at the user's leisure, and then the results are read back
for analysis. This workflow may be better suited to slow calculations
performed on a batch system. For this tutorial example we use GPAW.

.. literalinclude:: displacements_files_write.py

After optimizing the structure and writing displaced files, we use a
Bash script to iterate over the files, run a DFT calculation and save
the outputs somewhere useful. In this case it is a simple ``for`` loop
over all displacements, but in practice one might e.g. run in small
batches on different machines, run several displacements in parallel,
add more logging...

.. literalinclude:: displacements_files_run.sh

Finally the results are analyzed. Note that it isn't really important
for the filenames to reflect the displacements or match the input file
names; this is only done here to avoid overwriting files and assist
troubleshooting.

.. literalinclude:: displacements_files_read.py

The output frequencies and *ethanol_gpaw_vibs.xyz* file should look
pretty similar to the MOPAC results in Example 1. With the input files
above, the low frequencies do not come out especially close to zero;
this should improve with finer GPAW calculation parameters!
                    
.. automodule:: ase.vibrations.finite_diff
