.. module:: ase.crystallography.xrdebye

===========================
X-ray scattering simulation
===========================


The module for simulation of X-ray scattering properties from the atomic
level. The approach works only for finite systems, so that periodic boundary
conditions and cell shape are ignored.

Large parts of the derivations presented here can also be found in the `Debyer documentation`_.

Theory
======
The scattering can be calculated using Debye formula [Debye1915]_, and a rederivation can be found in [Farrow2009]_. For completeness, we do the full derivation out here. We assume that each photon is scattered only once, the amplitude of the scattered wave is given by:

.. math::

    \Psi(\vec{q}) = \sum_i \psi_i = \sum_i f_i exp(-i \vec{q} \cdot \vec{r})

The intensity is the absolute squared of this function, giving us:

.. math::

    I(\vec{q}) = |\Psi(\vec{q})|^2 &= \Psi^*(\vec{q}) \Psi(\vec{q}) \\ 
              &= \sum_i \sum_j f_i f_j exp(-i \vec{q} \cdot (\vec{r_i} - \vec{r_j}))

The Debye scattering equation gives the spherically averaged intensity. To take this average, we first need the area of the sphere of a radius `r`, which is `4 \pi r^2`.

Now, we let `\gamma` be the angle between `\vec{q}` and `\vec{r_i} - \vec{r_j}`. This allows us to average the oscillating contribution in the intensity sum:

.. math::

    \langle exp(-iqr_{ij} cos \gamma) \rangle &= \frac{1}{4 \pi r_{ij}^2} \int_0^\pi d \gamma exp(-iqr_{ij} cos \gamma) 2 \pi r_{ij}^2 sin \gamma \\
    &= \frac{1}{2} \int_0^\pi d \gamma exp(-iqr_{ij} cos \gamma) sin \gamma \\
    &= \frac{1}{2} \left[ \frac{exp(-iqr_{ij} cos \gamma)}{iqr_{ij}} \right]_0^\pi \\
    &= \frac{exp(iqr_{ij}) - exp(-iqr_{ij})}{2iqr_{ij}} \\
    &= \frac{sin(qr_{ij})}{qr_{ij}}

where:

- `i` and `j` -- atom indexes;
- `f_i(q)` -- `i`-th atomic scattering factor;
- `r_{ij}` -- distance between atoms `i` and `j`;
- `q` is a scattering vector length defined using scattering angle
  (`\theta`) and wavelength (`\lambda`) as
  `q = 4\pi \cdot \sin(\theta)/\lambda`.

The final expression above corresponds to the *unnormalized* sinc function. This is different from say, Numpy's implementation of sinc, which returns the *normalized* sinc function.

The Histogram Approximation
===========================

The above formula involves an `O(N^2)` calculation of distances (and the relevant sine function). This can get extremely expensive. To alleviate this situation, distances can be binned by creating a histogram of the distance magnitudes, and the relevant contributions to the diffraction pattern needs to only be evaluated for distances which have a contribution of 1 or more pairs of atoms. The derivation for a monoatomic system is as follows:

.. math::
    I(q) &= f^2 \sum_i^N \sum_j^N \frac{sin(q r_{ij}) }{q r_{ij}} \\
    &= f^2 \left(N +  2 \sum_i^N \sum_{j>i}^N \frac{sin(q r_{ij}) }{q r_{ij}} \right) \\
    &= f^2 \left(N +  2 \sum_i^{N_{bins}} n_k \frac{sin(q r_k) }{q r_k} \right)

In the final expression above `n_k` and `r_k` are the number of pairs in the `k`-th bin, and the distance, respectively. For large systems (1000-10000+ atoms), the histogram approximation with a bin width of 1e-3 Å (the default value) can speed up calculations by thousands of times compared to a naive summation. The `l_2`-norm of error in the diffraction pattern due to this approximation order of 0.05%.

For pairs of atoms of different species say `a` and `b`, we have:

.. math::
    I(q) &= f_i f_j \sum_i^{N_a} \sum_j^{N_b} \frac{sin(q r_{ij}) }{q r_{ij}} \\
    &= f_i f_j \left(2 \sum_i^{N_a} \sum_{j}^{N_b} \frac{sin(q r_{ij}) }{q r_{ij}} \right)

where we can now proceed to bin the distances exactly as in the single-species case.

The thermal vibration of atoms can be accounted by introduction of damping
exponent factor (Debye-Waller factor) written as `\exp(-B \cdot q^2 / 2)`.
The angular dependency of geometrical and polarization factors are expressed
as [Iwasa2007]_ `\cos(\theta)/(1 + \alpha \cos^2(2\theta))`, where `\alpha
\approx 1` if incident beam is not polarized.

Units
-----

The following measurement units are used:

- scattering vector `q` -- inverse Angstrom (1/Å),
- thermal damping parameter `B` -- squared Angstrom (Å\ :sup:`2`).


Example
=======

The considered system is a nanoparticle of silver which is built using
``FaceCenteredCubic`` function (see :mod:`ase.cluster`) with parameters
selected to produce approximately 2 nm sized particle::

  from ase.cluster.cubic import FaceCenteredCubic
  from ase.crystallography import XrDebye
  import numpy as np

  surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
  atoms = FaceCenteredCubic('Ag', [(1, 0, 0), (1, 1, 0), (1, 1, 1)],
                            [6, 8, 8], 4.09)

Next, we need to specify the wavelength of the X-ray source::

  xrd = XrDebye(atoms=atoms, wavelength=0.50523, histogram_approximation=False)

The X-ray diffraction pattern on the `2\theta` angles ranged from 15 to 30
degrees can be simulated as follows::

  xrd.calc_pattern(x=np.arange(15, 30, 0.1), mode='XRD')
  xrd.plot_pattern('xrd.png')

The resulted X-ray diffraction pattern shows (220) and (311) peaks at 20 and
~24 degrees respectively.

.. image:: xrd.png

The small-angle scattering curve can be simulated too. Assuming that
scattering vector is ranged from `10^{-2}=0.01` to `10^{-0.3}\approx 0.5` 1/Å
the following code should be run: ::

  xrd.calc_pattern(x=np.logspace(-2, -0.3, 50), mode='SAXS')
  xrd.plot_pattern('saxs.png')

The resulted SAXS pattern:

.. image:: saxs.png


Further details
===============

The module contains wavelengths dictionary with X-ray wavelengths for copper
and wolfram anodes::

  from ase.crystallography.xrddata import wavelengths
  print('Cu Kalpha1 wavelength: %f Angstr.' % wavelengths['CuKa1'])


The dependence of atomic form-factors from scattering vector is calculated
based on coefficients given in ``waasmaier`` dictionary according
[Waasmaier1995]_ if method of calculations is set to 'Iwasa'. In other case,
the atomic factor is equal to atomic number and angular damping factor is
omitted.


XrayDebye class members
-----------------------

.. autoclass:: XrayDebye
   :members:


References
==========

.. [Debye1915] P. Debye  Ann. Phys. **351**, 809–823 (1915)
.. _Debyer documentation: https://debyer.readthedocs.io/en/latest/
.. [Farrow2009] C.L. Farrow, and S.J.L. Billinge Acta Cryst. **A65**: 232-239 (2009) :doi:`10.1107/S0108767309009714`
.. [Iwasa2007] T. Iwasa, K. Nobusada J. Phys. Chem. C, **111**, 45-49 (2007) :doi:`10.1021/jp063532w`
.. [Waasmaier1995] D. Waasmaier, A. Kirfel Acta Cryst. **A51**, 416-431 (1995)
