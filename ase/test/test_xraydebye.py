"""Tests for XrDebye class"""

from pathlib import Path

import numpy as np
import pytest

from ase.utils.xrdebye import XrDebye
from ase.crystallography import XrayDebye
from ase.crystallography.xrddata import wavelengths, waasmaier
from ase.crystallography.xrayfunctions import construct_atomic_form_factor_spline
from ase.crystallography.xrdebye import one_species_contribution, two_species_contribution
from ase import Atoms
from ase.cluster import Icosahedron


@pytest.fixture
def xrd():
    # test system -- cluster of 587 silver atoms
    atoms = Atoms('Au', positions=[[0.0, 0.0, 0.0]])
    xrd = XrayDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.00,
                   method='Iwasa', alpha=1.00, histogram_approximation=False)

    return xrd


@pytest.fixture
def xrd_tuple():
    # test system -- cluster of 587 silver atoms
    atoms = Icosahedron('Au', noshells=4)
    xrd1 = XrayDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.00,
                   method='Iwasa', alpha=1.00, histogram_approximation=False)

    xrd2 = XrayDebye(atoms=atoms, wavelength=wavelengths['CuKa1'], damping=0.00,
                   method='Iwasa', alpha=1.00, histogram_approximation=True)

    return xrd1, xrd2


def test_formfactor_correctness(xrd):
    x = np.linspace(0, 1, 10)

    xrddata = xrd.calc_pattern(x[:1])
    formfactor = construct_atomic_form_factor_spline('Au', x)
    evaluated_formfactor = formfactor(np.array([0]))

    # The extra factor of 2 is because method='Iwasa' introduces
    # a factor of 0.5 at s=0.0.
    assert np.allclose(np.sqrt(xrddata * 2.0), evaluated_formfactor)


def test_contributions_correctness():
    rng = np.random.default_rng(42)
    random_positions = rng.random((2, 3))

    form_factor = construct_atomic_form_factor_spline('Au', np.linspace(0, 1, 100))
    x = np.array([0.1])

    c1 = one_species_contribution(random_positions, form_factor, x)
    c2 = two_species_contribution(random_positions, random_positions, form_factor, form_factor, x)

    # The two-species contribution function leads to double-counting if the position arrays
    # are the same array.
    assert np.isclose(c1, c2 / 2.0)


def test_histogram_approximation(xrd_tuple):
    xrd1, xrd2 = xrd_tuple
    two_theta = np.linspace(15, 105, 200)
    pattern_without_histogram = xrd1.calc_pattern(two_theta)

    pattern_with_histogram = xrd2.calc_pattern(two_theta)

    assert pattern_without_histogram == pytest.approx(pattern_with_histogram, rel=5e-2)
