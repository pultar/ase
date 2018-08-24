.. module:: ase.calculators.kim

=====
 KIM
=====
The Open Knowledgebase of Interatomic Models (KIM) is an online framework at
openkim.org_ for making molecular simulations reliable, reproducible, and
portable. Computer implementations of interatomic models are archived on
openkim.org_, verified for coding integrity (using ``Verification Checks``),
and tested (using ``KIM Tests``) for their predictions for a variety of
material properties. Models from openkim.org_ can be used seemlessly with major
simluation codes that support the KIM application programming interface (API)
standard.

The KIM Calculator makes it possible to use any interatomic model archived in
openkim.org_ from within ASE. Each model is identified by a unique ``KIM ID``,
which is passed to the KIM Calculator when it is initialized.

Instructions
------------

The KIM calculator requires the following packages:

- The KIM API package, which enables a KIM model to communicate with a
  simulation code that uses it. The KIM API is a C++ package that must be
  installed on your machine in order to be able to use models archived in
  openkim.org_. In addition, any models to be used must be downloaded and
  installed.  Instructions on how to do this are given on the `KIM API page
  <https://openkim.org/kim-api/>`_.


- The ``kimpy`` Python package. This package provides a wrapper to the KIM API
  to allow models archived in openkim.org_ to be used by Python applications,
  such as ASE. The ``kimpy`` package is available through PyPI. To install it do:

  .. code-block:: bash

    $ pip install kimpy

Example
-------

Here is an example of how to use the KIM calculator.

.. code-block:: python

    from ase.calculators.kim import KIM
    from ase.lattice.cubic import Diamond

    modelname = 'SW_Si__MO_405512056662_004'
    calc = KIM(modelname)

    atoms = Diamond(size=(1, 1, 1), latticeconstant=5.43095, symbol='Si', pbc=True)

    atoms.set_calculator(calc)

    energy = atoms.get_potential_energy()

    print('Cohesive Energy:', energy/len(atoms))

This example computes the cohesive energy of a diamond periodic
lattice using the Stillinger-Weber potential archived in openkim.org_ as
`SW_Si__MO_405512056662_004 <https://openkim.org/cite/SW_Si__MO_405512056662_004>`_.

Running this example gives the result: ``Cohesive Energy: -4.336399999917175``


.. _openkim.org: https://openkim.org/
