from numpy.testing import assert_allclose
import pytest

from ase.dft.kpoints import RegularGridKPoints


@pytest.mark.parametrize(
    'size, offset, expected_kpts',
    [([2, 2, 2],
      [0.0, 0.0, 0.0],
      [[-0.25, -0.25, -0.25],
       [-0.25, -0.25, 0.25],
       [-0.25, 0.25, -0.25],
       [-0.25, 0.25, 0.25],
       [0.25, -0.25, -0.25],
       [0.25, -0.25, 0.25],
       [0.25, 0.25, -0.25],
       [0.25, 0.25, 0.25]]),
     ([2, 2, 2],
      [0.25, 0.25, 0.25],
      [[0., 0., 0.],
       [0., 0., 0.5],
       [0., 0.5, 0.],
       [0., 0.5, 0.5],
       [0.5, 0., 0.],
       [0.5, 0., 0.5],
       [0.5, 0.5, 0.],
       [0.5, 0.5, 0.5]])])
def test_regular_grid_k_points(size, offset, expected_kpts):
    grid_kpts = RegularGridKPoints(size, offset=offset)

    assert_allclose(grid_kpts.kpts, expected_kpts)
    assert grid_kpts.size == size
    assert_allclose(grid_kpts.offset, offset)
