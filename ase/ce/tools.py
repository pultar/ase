import numpy as np
import math
from numpy.linalg import matrix_rank
from copy import deepcopy

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

def sort_by_internal_distances(atoms, indices, num_decimals=3):
    """Sort the indices according to the distance to the other elements"""
    if len(indices) <= 1:
        return range(len(indices)), []
    elif len(indices) == 2:
        return range(len(indices)), [(0,1)]
    
    mic_dists = []
    for indx in indices:
        all_indx = deepcopy(list(indices))
        all_indx.remove(indx)
        mic_distances = atoms.get_distances(indx, all_indx, mic=True)
        mic_distances *= 10**num_decimals
        mic_distances = mic_distances.astype(np.int32).tolist()
        mic_distances = sorted(mic_distances)
        mic_dists.append(mic_distances)

    sort_order = [indx for _,indx in sorted(zip(mic_dists,range(len(indices))))]
    equivalent_sites = []
    for i in range(len(sort_order)):
        for j in range(i+1,len(sort_order)):
            if mic_dists[i] == mic_dists[j]:
                equivalent_sites.append((i,j))
    return sort_order, equivalent_sites
