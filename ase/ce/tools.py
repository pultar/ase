import math
from itertools import permutations, combinations

import numpy as np
from numpy.linalg import matrix_rank
from ase.build.rotate import rotation_matrix_from_points


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
    return f(n) / f(r) / f(n - r)


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
    center = 0.5 * (cell[0, :] + cell[1, :] + cell[2, :])
    min_max_dist = 1E10
    minimal_cluster = None
    for ref_indx in range(len(indices)):
        cluster_cpy = cluster.copy()
        sub_indx = [i for i in range(len(indices)) if i != ref_indx]
        mic_dists = cluster_cpy.get_distances(
            ref_indx, sub_indx, mic=True, vector=True)
        com = cluster_cpy[ref_indx].position + \
            np.sum(mic_dists, axis=0) / len(indices)
        cluster_cpy.translate(center - com)
        cluster_cpy.wrap()
        pos = cluster_cpy.get_positions()
        lengths = []

        for comb in combinations(range(len(indices)), r=2):
            dist = pos[comb[0], :] - pos[comb[1], :]
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

    angles = atoms.get_angles(indx_comb, mic=mic).round(decimals=0) + 0
    angles = angles.tolist()
    for i, angle in enumerate(angles):
        if math.isnan(angle):
            angles[i] = 0
        elif angle == 180.0:
            angles[i] = 0
    angles.sort(reverse=True)
    return angles


def sort_by_internal_distances(atoms, indices):
    """Sort the indices according to the distance to the other elements"""
    if len(indices) <= 1:
        return range(len(indices)), []
    elif len(indices) == 2:
        return range(len(indices)), [(0, 1)]

    mic_dists = []
    for indx in indices:
        mic_distances = atoms.get_distances(
            indx, list(indices), mic=True).round(decimals=3) + 0
        mic_distances = sorted(mic_distances.tolist())
        mic_dists.append(mic_distances)

    sort_order = [indx for _, indx in sorted(
        zip(mic_dists, range(len(indices))))]
    mic_dists.sort()
    equivalent_sites = [[i] for i in range(len(indices))]
    site_types = [i for i in range(len(indices))]
    for i in range(len(sort_order)):
        for j in range(i + 1, len(sort_order)):
            if mic_dists[i] == mic_dists[j]:
                if site_types[j] > i:
                    # This site has not been assigned to another category yet
                    site_types[j] = i
                st = site_types[j]
                if j not in equivalent_sites[st]:
                    equivalent_sites[st].append(j)

    # Remove empty lists from equivalent_sites
    equivalent_sites = [entry for entry in equivalent_sites if len(entry) > 1]
    return sort_order, equivalent_sites


def ndarray2list(data):
    """Converts nested lists of a combination of lists and numpy arrays
    to list of lists"""
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        return data

    data = list(data)
    for i in range(len(data)):
        data[i] = ndarray2list(data[i])
    return list(data)
