=========
Calculate
=========


Set calculator
--------------

Allows :mod:`ase.gui` to choose a calculator for internal computations (see
below). Currently available the GPAW density functional backend and 
several force fields (LJ, EMT, EAM, Brenner). 
For some of the choices a Python code can be generated for use in scripts.


Energy and forces
-----------------

Invokes the currently set calculator and provides energies and
optional forces for all atoms.


Energy minimization
-------------------

Runs an ASE relaxation using the currently selected calculator with a
choice of relaxation algorithm and convergence criteria. Great for
quickly (pre-)relaxing a molecule before placing it into a bigger
system.
