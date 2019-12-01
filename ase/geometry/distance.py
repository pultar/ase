import numpy as np
from ase.geometry import get_layers
from itertools import product
from numpy.linalg import norm


def distance(s1, s2, permute=True):
    """Get the distance between two structures s1 and s2.
    
    The distance is defined by the Frobenius norm of
    the spatial distance between all coordinates (see
    numpy.linalg.norm for the definition).

    permute: minimise the distance by 'permuting' same elements
    """

    s1 = s1.copy()
    s2 = s2.copy()
    for s in [s1, s2]:
        s.translate(-s.get_center_of_mass())
    s2pos = 1. * s2.get_positions()
    
    def align(struct, xaxis='x', yaxis='y'):
        """Align moments of inertia with the coordinate system."""
        Is, Vs = struct.get_moments_of_inertia(True)
        IV = list(zip(Is, Vs))
        IV.sort(key=lambda x: x[0])
        struct.rotate(IV[0][1], xaxis)
        
        Is, Vs = struct.get_moments_of_inertia(True)
        IV = list(zip(Is, Vs))
        IV.sort(key=lambda x: x[0])
        struct.rotate(IV[1][1], yaxis)

    align(s1)

    def dd(s1, s2, permute):
        if permute:
            s2 = s2.copy()
            dist = 0
            for a in s1:
                imin = None
                dmin = np.Inf
                for i, b in enumerate(s2):
                    if a.symbol == b.symbol:
                        d = np.sum((a.position - b.position)**2)
                        if d < dmin:
                            dmin = d
                            imin = i
                dist += dmin
                s2.pop(imin)
            return np.sqrt(dist)
        else:
            return np.linalg.norm(s1.get_positions() - s2.get_positions())

    dists = []
    # principles
    for x, y in zip(['x', '-x', 'x', '-x'], ['y', 'y', '-y', '-y']):
        s2.set_positions(s2pos)
        align(s2, x, y)
        dists.append(dd(s1, s2, permute))
   
    return min(dists)


def dist_from_point(positions, b):
    """Returns the distance from an array of points to a point

    Parameters:

    positions: float ndarray of shape (n,d)
        Positions of the atoms
    b: fload ndarray of shape (d)
        The point

    Returns:

    d: vector of distances from positions to point
    """
    return norm(positions - b, axis=1)


def dist_from_line(positions, m, b):
    """Returns the minimum distance from an array of points to a line defined by a point and direction.

    Parameters:

    positions: float ndarray of shape (n,d)
        Positions of the atoms
    m: float ndarray of shape (d)
        A vector along which the line travels
    b: float ndarray of shape (d)
        One point on the line
            """

    u = positions - b
    x = u @ m / norm(m)**2
    return norm(u - np.outer(x, m), axis=1)


def dist_from_line_segment(positions, b1, b2):
    """Returns the minimum distance from an array of points to a line segment defined between two points.

    Parameters:

    positions: float ndarray of shape (n,d)
        Positions of the atoms
    b1: float ndarray of shape (d)
        One end point of the line
    b2: float ndarray of shape (d)
        The other end point of the line

    Returns:

    d: vector of distances from positions to point
     """

#    x = -((b2 - positions)@(b1+b2)) / (norm(b1)**2 - norm(b2)**2)
    den = norm(b1)**2 - norm(b2)**2
    x = np.array([(b2-p)@(b1+b2)/den for p in positions])
    c1 = norm(positions - np.outer(x, b1) - np.outer(1 - x, b2), axis=1)
    c2 = norm(positions - b1, axis=1)
    c3 = norm(positions - b2, axis=1)
    return np.min((c1, c2, c3), axis=0)


def dist_from_plane(positions, m1, m2, b): 
    """Returns the minimum distance from an array of points to a plane
    defined by one point and two spanning vectors.

    Parameters:

    positions: float ndarray of shape (n,3)
        Positions of the atoms
    m1: float ndarray of shape (3)
        A vector in the plane
    m2: float ndarray of shape (3)
        A vector in the plane not parallel to m1
    b: float ndarray of shape (3)
        A point on the plane

    Returns:

    d: vector of distances from positions to point
    """
    u = positions - b
    x = ((u@m1) * norm(m2)**2 - (u@m2) * (m1@m2)) / \
        (norm(m1)**2 * norm(m2)**2 - (m1@m2)**2)
    y = ((u@m2) * norm(m1)**2 - (u@m1) * (m1@m2)) / \
        (norm(m1)**2 * norm(m2)**2 - (m1@m2)**2)
    return norm(u - np.outer(x, m1) - np.outer(y, m2), axis=1)


def dist_from_plane_segment(positions, m1, m2, m3): 
    """Returns the minimum distance from an array of points to a plane
    segment defined by 3 points.

    Parameters:

    positions: float ndarray of shape (n,3)
        Positions of the atoms
    m1: float ndarray of shape (3)
        A vertex of the plane
    m2: float ndarray of shape (3)
        Another vertex of the plane
    m3: float ndarray of shape (3)
        The final vertex of the plane

    Returns:

    d: vector of distances from positions to point
    """
    A = np.vstack((m1, m2, m3))  # 3-by-3
    # rows are vertices
    # columns are coordinate values
    M = A.T @ A
    MM = np.zeros((4, 4))
    MM[:3, :3] = M
    MM[0:3, 3] = 1
    MM[3, 0:3] = -1
    d = []
    z = np.zeros(4)
    z[3] = -1
    for pos in positions:
        b = A@pos
        z[:3] = b
        xyzl = np.linalg.solve(MM, z)
        xyz = xyzl[:3]
        p = A@xyz
        d.append(norm(pos - p))
    return np.array(d)


def dist_from_plane_normal(atoms, s, b):
    # note: use ase.geometry get layers to sort by distance from miller plane
    # with any intercept
    a = atoms.copy()
    a.positions = a.positions - b
    l, _ = get_layers(a, s)
    return l


def repeats(positions, cell): 
    """Returns an array containing the shifted positions according to boundary
    conditions. Since

    .. math:: |(\mathbf{r}_1 + n\mathbf{a} + m\mathbf{b} + p\mathbf{c})
    - \mathbf{r}_2| = | (\mathbf{r}_2 - n\mathbf{a} +m\mathbf{b} +
    p\mathbf{c}) - \mathbf{r}_1 |

    it is equivalent to move either the positions of atoms or of a
    geometry.

    Parameters:

    positions: float ndarray of shape (n,d)
        Positions of the atoms or of the vectors defining a geometry
    cell: cell type object or float ndarray of shape (n,d)
        Periodic (super)cell of the atoms

    Returns:

    translated_positions: float ndarray of shape (3**d,n,d)
        The positions of the atoms in the original cell plus its 3**d
        touching neighbors (touching may be just at a corner) or the
        translated positions of the vectors defining a geometry

    """
    assert(positions.shape[1] == cell.shape[0] == cell.shape[1])
    indices = product(*[[-1, 0, 1]] * positions.shape[1])
    translated_positions = []
    for i in indices:
        translated_positions.append(positions - cell @ i)
    return np.array(translated_positions)
