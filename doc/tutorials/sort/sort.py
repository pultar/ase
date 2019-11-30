from ase.build import molecule
atoms = molecule('CH3CH2OCH3')
cell = [[10, 0, 0],[0, 10, 0],[0, 0, 10]]
atoms.set_cell(cell)
atoms.center()

from ase.geometry.distance import dist_from_point, \
        dist_from_line_segment, dist_from_plane

#distance from point
import numpy as np
point = np.array([5, 5, 5])
positions = atoms.positions
d = dist_from_point(positions, point)
n = len(atoms)
pointers = np.arange(n)
from ase.visualize import view
view(atoms[sorted(pointers,key=lambda x:d[x])])

#distance from line segment
b1 = np.array([5, 5, 0])
b2 = np.array([5, 5, 10])
d = dist_from_line_segment(positions, b1, b2)
view(atoms[sorted(pointers,key=lambda x:d[x])])

#distance from a plane
m1 = np.array([0, 10, 0])
m2 = np.array([0, 0, 10])
b = np.array([5, 0, 0])
d = dist_from_plane(positions, m1, m2, b)
view(atoms[sorted(pointers,key = lambda x:d[x])])

#lexicographical sorting for the distance from each bisecting plane
atoms = molecule('cyclobutene')
atoms.set_cell(cell)
atoms.center()
positions = atoms.positions
n = len(atoms)
pointers = np.arange(n)
mat = np.array([[0, 10, 0],[0, 0, 10],[5, 0, 0]])
d1 = dist_from_plane(positions, *mat)
d2 = dist_from_plane(positions, *np.roll(mat,1,axis=1))
d3 = dist_from_plane(positions, *np.roll(mat,2,axis=1))
view(atoms[sorted(pointers,key = lambda x:(round(d1[x], 3),
    round(d2[x], 3),
    round(d3[x], 3),)
    )])
