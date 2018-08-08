.. module:: ase.calculators.kim

=====
 KIM
=====

The Open Knowledgebase of Interatomic Models (OpenKIM_) is a repository containing
computer implementations of interatomic models (potentials and force fields).

TODO (Ellad) Difference between normal models and simulator models


Instructions
------------

The KIM calculator requires ``KIM API``, ``kimpy``, and ``kimsm`` to be installed
beforehead.

KIM API
*******

To create an installed-(standard)-build and install to the default directory
``/usr/local`` do the below.  Here we assume that ``/usr/local/bin`` is included as
part of the system's standard PATH setting.

.. code-block:: bash

    $ cd ${HOME}
    $ wget https://s3.openkim.org/kim-api/kim-api-vX.Y.Z.txz  # replace X.Y.Z with the current version number
    $ tar Jxvf kim-api-vX.Y.Z.txz
    $ cd kim-api-vX.Y.Z
    $ ./configure
    $ make
    $ sudo make install
    $ sudo ldconfig  # on Redhat-like systems you may need to first add /usr/local/lib to /etc/ld.so.conf

See `here <https://openkim.org/kim-api/>`_ more information about ``KIM API`` and
alternative ways to install it.

kimpy & kimsm
*************

.. code-block:: bash

    $ pip install kimpy
    $ pip install kimsm


Example
-------

Here is an example of how to use the KIM calculator.

.. code-block:: python

    from ase.calculators.kim import KIM
    from ase.lattice.cubic import FaceCenteredCubic


    modelname = 'ex_model_Ar_P_Morse_07C'
    calc = KIM(modelname)

    argon = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], size=(1, 1, 1),
                              symbol='Ar', pbc=(1, 0, 0), latticeconstant=3.0)

    argon.set_calculator(calc)

    energy = argon.get_potential_energy()
    forces = argon.get_forces()
    stress = argon.get_stress()

    print('Energy:', energy)
    print('forces:', forces)
    print('stress:', stress)


The example performs a single-point calculation for energy, forces, and stress
of an FCC argon that is periodic in the x direction within ASE using the KIM
potential model ``ex_model_Ar_P_Morse_07C``.


.. _OpenKIM: https://openkim.org/
