import numpy as np
import math
from numpy.linalg import matrix_rank
from copy import deepcopy
from itertools import permutations, combinations, combinations_with_replacement
from ase.visualize import view
from ase.build.rotate import rotation_matrix_from_points
from ase import Atoms
from collections import deque

def index_by_position(atoms):
    # add zero to avoid negative zeros
    tags = atoms.get_positions().round(decimals=6) + 0
    tags = tags.tolist()
    deco = sorted([(tag, i) for i, tag in enumerate(tags)])
    indices = [i for tag, i in deco]
    return indices

def sort_by_position(atoms):
    # Return a new Atoms object with sorted atomic order.
    # The default is to order according to chemical symbols,
    # but if *tags* is not None, it will be used instead.
    # A stable sorting algorithm is used.
    indices = index_by_position(atoms)
    return atoms[indices]

def wrap_and_sort_by_position(atoms):
    atoms.wrap()
    atoms = sort_by_position(atoms)
    return atoms

def nCr(n, r):
    """Compute and return combination"""
    f = math.factorial
    return f(n)/f(r)/f(n-r)

def reduce_matrix(matrix):
    matrix = matrix[:, ~np.all(matrix == 0., axis=0)]
    offset = 0
    rank = matrix_rank(matrix)
    while matrix.shape[1] > rank:
        temp = np.delete(matrix, -1 - offset, axis=1)
        if matrix_rank(temp) < rank:
            offset += 1
        else:
            matrix = temp
            offset = 0
    return matrix

def create_cluster(atoms, indices):
    """Create a cluster centered in the unitcell"""
    #at_cpy = atoms.copy()
    cluster = atoms[list(indices)]
    cell = cluster.get_cell()
    center = 0.5*(cell[0,:] + cell[1,:] + cell[2,:])
    min_max_dist = 1E10
    minimal_cluster = None
    for ref_indx in range(len(indices)):
        cluster_cpy = cluster.copy()
        sub_indx = [i for i in range(len(indices)) if i != ref_indx]
        mic_dists = cluster_cpy.get_distances(ref_indx, sub_indx, mic=True, vector=True)
        com = cluster_cpy[ref_indx].position + np.sum(mic_dists, axis=0)/len(indices)
        cluster_cpy.translate(center-com)
        cluster_cpy.wrap()
        pos = cluster_cpy.get_positions()
        lengths = []

        for comb in combinations(range(len(indices)), r=2):
            dist = pos[comb[0],:] - pos[comb[1],:]
            length = np.sqrt(np.sum(dist**2))
            lengths.append(length)
        max_dist = np.max(lengths)
        if max_dist < min_max_dist:
            min_max_dist = max_dist
            minimal_cluster = cluster_cpy.copy()
    return minimal_cluster

def shift(array):
    ref = array[-1]
    array[1:] = array[:-1]
    array[0] = ref
    return array

def sorted_internal_angles(atoms, mic=False):
    """Get sorted internal angles of a
    """
    if len(atoms) <= 2:
        return [0]

    angles = []
    indx_comb = list(combinations(range(len(atoms)), r=3))

    # Have to include all cyclic permutations in addition
    n = len(indx_comb)
    for i in range(n):
        new_list = list(indx_comb[i])
        for _ in range(2):
            new_list = shift(new_list)
            indx_comb.append(tuple(new_list))

    angles = atoms.get_angles(indx_comb, mic=mic).round(decimals=0)+0
    angles = angles.tolist()
    for i,angle in enumerate(angles):
        if math.isnan(angle):
            angles[i] = 0
        elif angle == 180.0:
            angles[i] = 0
    angles.sort(reverse=True)
    return angles

def distance_center_to_faces(atoms):
    cell = atoms.get_cell()
    a1 = cell[0,:]
    a2 = cell[1,:]
    a3 = cell[2,:]
    center = 0.5*(a1 + a2 + a3)

    n12 = np.cross(a1, a2)
    n13 = np.cross(a1, a3)
    n23 = np.cross(a2, a3)
    n12 /= np.sqrt(np.sum(n12**2))
    n13 /= np.sqrt(np.sum(n13**2))
    n23 /= np.sqrt(np.sum(n23**2))

    proj12 = np.abs(center.dot(n12))
    proj13 = np.abs(center.dot(n13))
    proj23 = np.abs(center.dot(n23))
    dists = [proj12, proj13, proj23]
    return dists

def sort_by_internal_distances(atoms, indices, template, num_decimals=3):
    """Sort the indices according to the distance to the other elements"""
    if len(indices) <= 1:
        return range(len(indices)), []
    elif len(indices) == 2:
        return range(len(indices)), [(0,1)]

    cluster = create_cluster(atoms, indices)
    pos_templ = template.get_positions().T
    pos_cluster = cluster.get_positions().T

    # Set the centroid to the origin
    com_templ = np.sum(pos_templ, axis=1)/len(indices)
    com_cluster = np.sum(pos_cluster, axis=1)/len(indices)
    for i in range(pos_templ.shape[1]):
        pos_templ[:,i] -= com_templ
        pos_cluster[:,i] -= com_cluster

    # Check all permutations of the three atoms to find one that match
    perm = permutations(range(len(indices)))
    sort_order = None
    pos_difference = []
    for p in perm:
        pos = np.zeros_like(pos_cluster)
        for i,indx in enumerate(p):
            pos[:,i] = pos_cluster[:,indx]

        R = rotation_matrix_from_points(pos, pos_templ)
        new_pos = R.dot(pos)
        pos_difference.append(new_pos-pos_templ)
        if np.allclose(new_pos, pos_templ):
            sort_order = p
            break

    #if sort_order is None:
    #    #temp = Atoms(positions=pos_templ.T)
    #    #cl = Atoms(positions=pos_cluster.T)
    #    #view(temp)
    #    #view(cl)
    #    print(pos_difference)
    #    view(cluster)
    #    view(template)

    #assert sort_order is not None

    mic_dists = []
    for indx in indices:
        mic_distances = atoms.get_distances(indx, list(indices), mic=True).round(decimals=3)+0
        mic_distances = mic_distances.tolist()
        #mic_distances *= 10**num_decimals
        #mic_distances = mic_distances.astype(np.int32).tolist()

        mic_distances.sort()
        mic_dists.append(mic_distances)

    sort_order = [indx for _,indx in sorted(zip(mic_dists,range(len(indices))))]
    mic_dists.sort()
    equivalent_sites = [[i] for i in range(len(indices))]
    site_types = [i for i in range(len(indices))]
    for i in range(len(sort_order)):
        for j in range(i+1,len(sort_order)):
            if mic_dists[i] == mic_dists[j]:
                if  site_types[j] > i:
                    # This site has not been assigned to another category yet
                    site_types[j] = i
                st = site_types[j]
                if j not in equivalent_sites[st]:
                    equivalent_sites[st].append(j)

    # Remove empty lists from equivalent_sites
    equivalent_sites = [entry for entry in equivalent_sites if len(entry) > 1]
    return sort_order, equivalent_sites

def is_rotation_reflection_matrix(matrix):
    """Returns true if the matrix is a rotation reflection matrix"""
    det = np.abs(np.linalg.det(matrix))
    return np.abs(det-1.0) < 1E-5

def hashable(array):
    """Creates a hashable key based on the entries in a list"""
    separator = "_"
    return separator.join((str(entry) for entry in array))

def best_rot_matrix(source, target):
    """Determines a transformation matrix from source coordinates to target"""
    mat = target.dot(source.T)
    U,S,V = np.linalg.svd(mat)
    return U.dot(V)
