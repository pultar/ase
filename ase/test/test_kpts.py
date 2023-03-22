import numpy as np
from numpy.testing import assert_allclose
import pytest

from ase.calculators.calculator import WeightedKPoints
from ase.dft.kpoints import bandpath


def test_bandpath():
    print(bandpath('GX,GX', np.eye(3), 6))

def test_weighted_kpts():
    kpts = np.random.random([4, 3])
    weights = np.random.random(4)

    weighted_kpts = WeightedKPoints(kpts=kpts, weights=weights)
    assert_allclose(weighted_kpts.kpts, kpts)
    assert_allclose(weighted_kpts.weights, weights)

    with pytest.raises(IndexError):
        WeightedKPoints(kpts=kpts, weights=weights[:-1])

    combined_data = np.random.random([3, 4])

    weighted_kpts = WeightedKPoints.from_array(combined_data)
    assert_allclose(weighted_kpts.kpts, combined_data[:, :-1])
    assert_allclose(weighted_kpts.weights, combined_data[:, -1])

    with pytest.raises(IndexError):
        WeightedKPoints.from_array(np.random.random([3, 3]))
