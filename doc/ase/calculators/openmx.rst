.. module:: ase.calculators.openmx

======
OpenMX
======

Introduction
============

OpenMX_ (Open source package for Material eXplorer) is a software
package for nano-scale material simulations based on density functional
theories (DFT), norm-conserving pseudopotentials, and pseudo-atomic
localized basis functions. This interface makes it possible to use
OpenMX_ as a calculator in ASE, and also to use ASE as a post-processor
for an already performed OpenMX_ calculation.

You should import the OpenMX calculator when writing ASE code.
To import into your python code,

.. code-block:: python

  from ase.calculators.openmx import OpenMX

Then you can define a calculator object and set it as the calculator of an
atoms object:

.. code-block:: python

  calc = OpenMX(**kwargs)
  atoms.calc = calc

Environment variables
=====================

The environment variable :envvar:`ASE_OPENMX_COMMAND` must point to that file.

Set both environment variables in your shell configuration file:

.. highlight:: bash

::

  $ export ASE_OPENMX_COMMAND='openmx'

.. highlight:: python

Keyword Arguments of OpenMX objects
===================================

User must specify two keywords `data_path` and `definition_of_atomic_species`
manually. `data_path` is the location where pseudo potential is located,
`definition_of_atomic_species` is orbital and xc functional setting for atoms.
For example,

.. code-block:: python

  calc = OpenMX(data_path='/Where/openmx3.9/DFT_DATA19',
                definition_of_atomic_species=[['C', 'C5.0-s1p3', 'C_PBE19'],
                                              ['H', 'H5.0-s1', 'H_PBE19']],
                                              ...)

There is helper function for the getting `definition_of_atomic_species` just
in case you need it. You can get example orbitals using a helper funtion.

.. code-block:: python

  from ase.atoms import Atoms
  from ase.calculators.openmx import get_definition_of_atomic_species
  atoms = Atoms('CH4')
  get_definition_of_atomic_species(atoms, data_year='19', scf_xctype='gga-pbe')
  [['H', 'H6.0-s3p2', 'H_PBE19'], ['C', 'C6.0-s2p2d1', 'C_PBE19']]


In python, common argument format is to use lowercase alphabet. To follow
this rule, every OpenMX keyword has to be changed. For example,::

  scf.criterion -> scf_criterion
  Scf.Kgrid -> scf_kgrid
  MD.maxIter  -> md_maxiter

.. code-block:: python

    calc = OpenMX(scf_criterion = 1e-6, md_maxiter = 100, scf_kgrid = (4, 4, 4)
                  ...)

Some variables like `atoms_number` or `species_number`, which can be
guessed easily from `atoms`, are automatically generated, if not explicitly
given.

Matrix keywords are keyword that have special (complex) format in OpenMX.
For example,::

  <Definition.of.Atomic.Species
    H   H5.0-s2p2d1      H_CA13
    C   C5.0-s2p2d2      C_CA13
  Definition.of.Atomic.Species>

This is typical example of matrix keyword. User can specify explicitly this
argument using python list object. For example,

.. code-block:: python

  calc = OpenMX(definition_of_atomic_species=[['H','H5.0-s2p2d1','H_CA13'],
                                              ['C','C5.0-s2p2d2','C_CA13']])

Some keyword like `Atoms.SpeciesAndCoordinates` is generated automatically
if not explicitly given. To know about details about keywords,
see the official `OpenMX`_ manual.

.. _OpenMX: http://www.openmx-square.org

Examples
========


.. code-block:: python


.. autoclass:: OpenMX
