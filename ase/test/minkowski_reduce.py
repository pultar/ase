import numpy as np
from ase.geometry import minkowski_reduce

tol = 1E-14
rng = np.random.RandomState(0)

for i in range(40):
    B = rng.uniform(-1, 1, (3, 3))
    R, H = minkowski_reduce(B)
    assert np.allclose(H @ B, R, atol=tol)
    assert np.sign(np.linalg.det(B)) == np.sign(np.linalg.det(R))

    norms = np.linalg.norm(R, axis=1)
    assert (np.argsort(norms) == range(3)).all()

    # Test idempotency
    _, _H = minkowski_reduce(R)
    assert (_H == np.eye(3).astype(np.int)).all()


cell = np.array([[1, 1, 1], [0, 1, 4], [0, 0, 1]])
unimodular = np.array([[1, 2, 2], [0, 1, 2], [0, 0, 1]])
assert np.linalg.det(unimodular) == 1
lcell = unimodular.T @ cell

# test 3D
rcell, op = minkowski_reduce(lcell)
assert np.linalg.det(rcell) == 1

for pbc in [1, True, (1, 1, 1)]:
    rcell, op = minkowski_reduce(lcell, pbc=pbc)
    assert np.linalg.det(rcell) == 1
    assert np.sign(np.linalg.det(rcell)) == np.sign(np.linalg.det(lcell))

# test 0D
rcell, op = minkowski_reduce(lcell, pbc=[0, 0, 0])
assert (rcell == lcell).all()

# test 1D
for i in range(3):
    rcell, op = minkowski_reduce(lcell, pbc=np.roll([1, 0, 0], i))
    assert (rcell == lcell).all()

# test 2D
for i in range(3):
    rcell, op = minkowski_reduce(lcell, pbc=np.roll([0, 1, 1], i))
    assert (rcell[i] == lcell[i]).all()
    assert np.sign(np.linalg.det(rcell)) == np.sign(np.linalg.det(lcell))
