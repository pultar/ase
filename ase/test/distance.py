import itertools
import numpy as np

from ase import Atoms, Atom
from ase.geometry import distance

# artificial structure
org = Atoms('COPNS',
            [[-1.75072, 0.62689, 0.00000],
             [0.58357, 2.71652, 0.00000],
             [-5.18268, 1.36522, 0.00000],
             [-1.86663, -0.77867, 2.18917],
             [-1.80586, 0.20783, -2.79331]])

maxdist = 3.0e-13

# translate
for dx in range(3, 10, 2):
    new = org.copy()
    new.translate([dx / np.sqrt(2), -dx / np.sqrt(2), 0])
    dist = distance(org, new, True)
    dist2 = distance(org, new, False)
    print('translation', dx, '-> distance', dist)
    assert dist < maxdist
    assert dist == dist2

# rotate
for axis in ['x', '-y', 'z', np.array([1, 1, 1] / np.sqrt(3))]:
    for rot in [20, 200]:
        new = org.copy()
        new.translate(-new.get_center_of_mass())
        new.rotate(rot, axis)
        dist = distance(org, new, True)
        dist2 = distance(org, new, False)
        print('rotation', axis, ', angle', rot, '-> distance', dist)
        assert dist < maxdist
        assert dist == dist2

if 0:
    # reflect
    new = Atoms()
    cm = org.get_center_of_mass()
    for a in org:
        new.append(Atom(a.symbol, -(a.position - cm)))
    dist = distance(org, new)
    print('reflected -> distance', dist)

# permute
for i, a in enumerate(org):
    if i < 3:
        a.symbol = 'H'

for indxs in itertools.permutations(range(3)):
    new = org.copy()
    for c in range(3):
        new[c].position = org[indxs[c]].position
    dist = distance(org, new)
    print('permutation', indxs, '-> distance', dist)
    assert dist < maxdist

from ase.geometry.distance import dist_from_point,dist_from_line,\
        dist_from_line_segment,dist_from_plane,repeats,dist_from_plane_normal

# 2-D testing
visual = False
positions = np.array([
    [1, 1],
    [2, 2],
    [3, 3],
])

m = np.array([1, 1])
b = np.array([0, 1])
d = dist_from_line(positions, m, b)
print('distances from line m={} b={} is d={}'.format(m,b,d))
assert((abs(d - np.roll(d,1)) < 0.05).all())
d = dist_from_line_segment(positions, b, m+b)
print('distances from line segment b1={} b2={} is d={}'.format(b, m+b, d))

if visual:
    import matplotlib.pyplot as plt
    plt.plot(*positions.transpose(),'o')
    plt.plot(*np.vstack((5*m+b,b)).transpose(),'k-')
    plt.plot(*np.vstack((m+b,b)).transpose(),'ko-')
    plt.show()

cell = np.array([[3, 0], [0, 3]])
ds = [dist_from_line(tp,m,b) for tp in repeats(positions,cell)]
ds = np.min(ds, axis=0)
ds_2 = [dist_from_line(positions,m,tp) for tp in repeats(np.array([b]), cell)]
ds_2 = np.min(ds_2,axis=0)

assert(sorted(ds) == sorted(ds_2))

# 3-D testing
positions = np.array([
    [1, 1, 1],
    [2, 2, 2],
    [3, 3, 3],
    [4, 4, 4]
])
m = np.array([1, 1, 1])
b = np.array([1, 0, 0])
d1 = dist_from_line(positions, m, b)
print('distances from line m={} b={} is d={}'.format(m,b,d1))
d2=dist_from_line_segment(positions,b,b+m)
print('distances from line segment b1={} b2={} is d={}'.format(b,b+5*m,d2))

assert(sum(d1) < sum(d2))

if visual:
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot3D(*np.vstack((b,b+5*m)).transpose(),'k-')
    ax.plot3D(*np.vstack((b,b+m)).transpose(),'ko-')
    ax.scatter(*positions.transpose())
    plt.show()

# plane
m1 = np.array([-2, 1, 1])
m2 = np.array([1, -2, 1])
b = np.array([1, 1, 1])
cell = np.array([[5, 0, 0], [0, 5, 0], [0, 0, 5]], dtype=np.float64)

ds = [dist_from_plane(tp, m1, m2, b) for tp in repeats(positions,cell)]
ds = np.min(ds, axis=0)
ds_2 = [dist_from_plane(positions, m1, m2, tp) for tp in repeats(np.array([b]),cell)]
ds_2 = np.min(ds_2, axis=0)

assert(sorted(ds) == sorted(ds_2))

atoms = Atoms('Cu4',positions=positions)
atoms.set_cell(cell)
ds = dist_from_plane_normal(atoms,np.array([0,0,1]),b)
print('layer sort numbers = {}'.format(ds))
