.. _sort:

=========================================================
Sorting atoms by distance from geometric objects
=========================================================

Let's try to sort the atoms in a molecule by their distance from a geometric object. First we construct the molecule and the supercell it will be inside, and center the molecule in the supercell:

.. literalinclude:: sort.py
   :lines: 1-5

We then import the relevant functions for calculating the distances of atoms from :mod:`ase.geometry.distance`

.. literalinclude:: sort.py
   :lines: 7-8

The first and simplest case is to calculate the distances of atoms from a point, in this case the center of the cell:

.. literalinclude:: sort.py
   :lines: 10-14

The function is just a wrapper for taking the norm of the difference. Then we create a list of pointers which can be sorted using the calculated distances, and view the sorted atoms:

.. literalinclude:: sort.py
  :lines: 15-18

The tags can be visualized under the ``View>Show Labels`` tool bar, and for a molecule, it may also be good to show bonds by ``View>Show Bonds``. As expected, the atoms closer to the center of the molecule, and therefore the cell since the molecule was centered in the cell, have lower index numbers.

A non-trivial example is sorting by the distance from a line segment, in this case going through the z-axis of the molecule:

.. literalinclude:: sort.py
  :lines: 20-24
  
This distance is defined as the minimum distance from a position to one of the points on the line. This is effectively a projection operation (see :mod:`ase.geometry.distance`).

Finally one may sort by distance from a plane:

.. literalinclude:: sort.py
  :lines: 26-31

One may also want to sort hierarchically, meaning to sort with respect to one set of distances, then another, and so on. For this example, a more interesting molecule is taken which has atoms in the same plane so the hierarchical sort will be different from the non-hierarchical sort. The hierarchical sort is done with tuple comparison. Try this to see how it works:

>>> c1 = (1,0) < (0,1)
>>> c2 = (0,1) < (1,0)
>>> c3 = (1,0) < (1,1)
>>> print(c1, c2, c3)

The code is:

.. literalinclude:: sort.py
  :lines: 33-47

The numpy ``roll`` is used to make the hierarchically ordered yz-, zx-, and xy-planes bisecting the centered molecule. The distances are rounded to make sure the tuples evaluate to something different than the non-hierarchical sort. Try this to see why:

>>> (1.e-14,0) < (0,1)

The full script is at :download:`sort.py`. A LaTex document deriving the equations used to calculate the distances is at :download:`nearest-dist.pdf`.
