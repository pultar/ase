.. _rneb tutorial:

=======================================
 Reflective Nudged Elastic Band (RNEB)
=======================================
Reference: Mathiesen, N. R., Jonsson, H., Vegge, T., & García Lastra, J. M. 
(2019). R-neb: Accelerated nudged elastic band calculations by use of 
reflection symmetry. Journal of chemical theory and computation, 15(5),
3215-3222.

In a bulk system diffusion paths can be symmetric, this is not always obvious to 
the eye, so here we use the functionality in the RNEB class.
Using symmetry considerations, the reflective Nudged Elastic Band method allows
to reduce the number of images to be considered in a path by reflecting symmetry
equivalent images onto each other. For instance, a path with 5 intermediate 
images possessing reflection symmetry has two images than can be reflected onto
each other by the use of unit cell dependent symmetry operations. The R-NEB 
class implemented helps to identify paths that possess reflection symmetry. It 
will then return appropriate symmetry operators that are used to run ASEs NEB
method.

In this tutorial we will look at the bulk diffusion of Cu. Due to the high 
symmetry of the crystal structure, there exist only one symmetry equivalent 
path. Additionally, this path is reflective, meaning intermediate images can
be reflected onto each other.

Example 1: Bulk diffusion in Cu 
===============================


