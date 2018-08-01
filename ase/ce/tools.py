import math
from itertools import permutations, combinations, product
import numpy as np
from numpy.linalg import matrix_rank
import collections


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
    """Compute and return combination."""
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
    """Create a cluster centered in the unit cell."""
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


def distances_and_angles(atoms, ref_indx, float_obj_dist, float_obj_angle):
    """Get sorted internal angles of a"""
    indices = [a.index for a in atoms if a.index != ref_indx]
    if len(atoms) < 2:
        raise ValueError("distances and angles cannot be called for"
                         "{} body clusters".format(len(atoms)))
    if len(atoms) == 2:
        dist = atoms.get_distance(ref_indx, indices[0])
        classifier = float_obj_dist.get(dist)
        return [classifier]

    angles = []
    dists = []

    for comb in combinations(indices, r=2):
        angle = atoms.get_angle(comb[0], ref_indx, comb[1])
        ang_classifier = float_obj_angle.get(angle)
        angles.append(ang_classifier)

    dists = atoms.get_distances(ref_indx, indices, mic=True)
    dists = sorted(dists.tolist(), reverse=True)
    dists = [float_obj_dist.get(dist) for dist in dists]

    return dists + sorted(angles, reverse=True)


def get_cluster_descriptor(cluster, float_obj_dist, float_obj_angle):
    """Create a unique descriptor for each cluster."""
    dist_ang_tuples = []
    for ref_indx in range(len(cluster)):
        dist_ang_list = distances_and_angles(cluster, ref_indx,
                                             float_obj_dist, float_obj_angle)
        dist_ang_tuples.append(dist_ang_list)
    return dist_ang_tuples


def sort_by_internal_distances(atoms, indices, float_obj_dist, float_obj_ang):
    """Sort the indices according to the distance to the other elements."""
    if len(indices) <= 1:
        return list(range(len(indices))), "point"

    cluster = create_cluster(atoms, indices)
    if len(indices) == 2:
        dist_ang = get_cluster_descriptor(cluster, float_obj_dist, float_obj_ang)
        order = list(range(len(indices)))
        eq_sites = [(0, 1)]
        descr =  "{}_0".format(dist_ang[0][0])
        # return list(range(len(indices))), [(0, 1)], "{}_0".format(dist_ang[0][0])
        return order, eq_sites, descr

    dist_ang = get_cluster_descriptor(cluster, float_obj_dist, float_obj_ang)

    sort_order = [ind for _, ind in sorted(zip(dist_ang, range(len(indices))))]
    # mic_dists.sort()
    dist_ang.sort()
    equivalent_sites = [[i] for i in range(len(indices))]
    site_types = [i for i in range(len(indices))]
    for i in range(len(sort_order)):
        for j in range(i + 1, len(sort_order)):
            # if np.allclose(dist_ang[i], dist_ang[j], atol=0.002):
            if dist_ang[i] == dist_ang[j]:
                if site_types[j] > i:
                    # This site has not been assigned to another category yet
                    site_types[j] = i
                st = site_types[j]
                if j not in equivalent_sites[st]:
                    equivalent_sites[st].append(j)

    # Remove empty lists from equivalent_sites
    equivalent_sites = [entry for entry in equivalent_sites if len(entry) > 1]

    # Create a string descriptor of the clusters
    dist_ang_strings = []
    for item in dist_ang:
        strings = [str(x) for x in item]
        dist_ang_strings.append("_".join(strings))
    string_description = "-".join(dist_ang_strings)
    return sort_order, equivalent_sites, string_description

def ndarray2list(data):
    """Converts nested lists of a combination of lists and numpy arrays
    to list of lists"""
    if not isinstance(data, list) and not isinstance(data, np.ndarray):
        return data

    data = list(data)
    for i in range(len(data)):
        data[i] = ndarray2list(data[i])
    return list(data)


def dec_string(deco, equiv_sites):
    """Create the decoration string based on equiv sites."""
    equiv_dec = sorted(equivalent_deco(deco, equiv_sites))
    return ''.join(str(i) for i in equiv_dec[0])


def equivalent_deco(deco, equiv_sites):
    """Generate equivalent decoration numbers based on equivalent sites."""
    if not equiv_sites:
        return [deco]

    perm = []
    for equiv in equiv_sites:
        perm.append(list(permutations(equiv)))

    equiv_deco = []
    for comb in product(*perm):
        order = []
        for item in comb:
            order += list(item)

        orig_order = list(range(len(deco)))
        for i, srt_indx in enumerate(sorted(order)):
            orig_order[srt_indx] = order[i]
        equiv_deco.append([deco[indx] for indx in orig_order])

    unique_deco = []
    for eq_dec in equiv_deco:
        if eq_dec not in unique_deco:
            unique_deco.append(eq_dec)
    return unique_deco


def flatten(x):
    if isinstance(x, collections.Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def get_unique_name(size, max_dia, fam_id):
    name = "c{}_{}_{}".format(size, max_dia, fam_id)
    return name
