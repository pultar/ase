import numpy as np
import math
from numpy.linalg import matrix_rank

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

def nCr(n,r):
    #Compute and return combination
    f = math.factorial
    return f(n)/f(r)/f(n-r)

def reduce_matrix(matrix):
    matrix = matrix[:, ~np.all(matrix == 0., axis=0)]
    offset = 0
    rank = matrix_rank(matrix)
    # print("rank:{}".format(rank))
    while matrix.shape[1] > rank:
        temp = np.delete(matrix,-1-offset,axis=1)
        if matrix_rank(temp) < rank:
            offset += 1
        else:
            matrix = temp
            offset = 0
    return matrix
