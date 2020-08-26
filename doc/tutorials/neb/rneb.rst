.. _rneb tutorial:

=============================================
 Reflective Nudged Elastic Band method (RNEB)
=============================================

The :class:`RNEB <ase.rneb.RNEB>` can take advantage of symmetry in the
diffusion path to limit the computational cost of a NEB calculation.

The implementation is based on:

   | Mathiesen, N. R., Jonsson, H., Vegge, T., & García Lastra, J. M.
   | `R-NEB: Accelerated Nudged Elastic Band Calculations by Use of Reflection Symmetry`__
   | J. Chem. Theory Comput. 2019, 15(5), 3215-3222.
   
   __ https://doi.org/10.1021/acs.jctc.8b01229

In a bulk or slab system the diffusion paths can be symmetric, this is
not always obvious to the eye, so here we use the functionality in the
RNEB class.  Using symmetry considerations, the reflective Nudged
Elastic Band method reduce the number of images to be considered in a
path by reflecting symmetry equivalent images onto each other. For
instance, a path with 5 intermediate images possessing reflection
symmetry has two images which can be reflected onto each other by the
use of unit cell dependent symmetry operations. The implemented R-NEB
class helps to identify paths that possess reflection symmetry. It
will then return appropriate symmetry operators that are used to run
ASEs NEB method.

In this tutorial we will look at the surface diffusion of Cu on a fcc100 
surface. Due to the high symmetry of the crystal structure, there exists only one
symmetry equivalent path. Additionally, this path is reflective, meaning 
intermediate images can be reflected onto each other.

|initial| |final|

Example 1: Bulk diffusion in Cu 
================================

.. literalinclude:: rneb_reflective.py

The barrier is obtained by not calculating, but reflecting the last
two intermediate images. When visualizing the path, the calculator on
these images is not :mod:`EMT <ase.calculators.emt>` (when opening the
info ``Ctrl + I``), but ``unknown`` since these images have been
assigned the energy values of the corresponding images from the first
half of the path.

|rneb|

Example 2: Accelerate through reflective middle image NEB (RMI-NEB)
===================================================================
Another way of accelerating the NEB can be done by using only a single 
intermediate image if the path is reflective (reflective middle image NEB or 
RMI-NEB). That way only the energy of the image at half of the path
length is probed. This can come in handy, if only the magnitude of the barrier
is of interest, but not the energies of all intermediate images.
Here we start by creating a path using only one intermediate image. The final
image is created by symmetry operations using the ``get_final_image`` function
from the RNEB class. Using the relaxed initial image and appropriate symmetry
operators, the RNEB class creates the final relaxed image without the need of
any additional relaxation.
Although a single image gives a first guess on the barrier, it is not given
that the transition state lies at exactly half of the path. Therefore, in a
second step, we use the middle image from the RMI-NEB calculation as the final
image in a subsequent NEB. That way we are running a NEB calculation with only
half of the intermediate images. Note that for running a reflective NEB 
calculation on the full path, we would also have to calculate the intermediate
image again (Example 1) which we assume to be already in its relaxed state.

.. literalinclude:: rneb_rmineb.py

|rmineb|

* Are the energy barriers comparable to the example above?

In this case it was obviously enough to only use a single intermediate image
to find the transition state as the barrier is "bell-shaped". Further examples
on how to use the RNEB tools for bulk materials can for example be found in

Reference:

   | Bölle, F. T., Mathiesen, N. R., Nielsen, A. J., Vegge, T., Garcia‐Lastra, J. M., & Castelli, I. E.
   | `Autonomous discovery of materials for intercalation electrodes`__
   | Batteries & Supercaps, 2020, 3, 488-498.
   
   __ https://doi.org/10.1002/batt.201900152

.. hint::

	In bulk materials vacancy creation can cause atom indices to change. This 
	can be annoying when creating a path as the interpolation relies on all 
	indices being aligned. For this the rneb module has the 
	``rneb.reshuffle_positions`` function that might come in handy!

.. |rneb| image:: reflective_path.png
.. |rmineb| image:: rmineb.png
.. |initial| image:: rneb-I.png
.. |final| image:: rneb-F.png

.. autoclass:: ase.rneb.RNEB
   :members: find_symmetries, get_final_image, reflect_path
