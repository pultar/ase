import pytest
from ase.spacegroup import (get_bravais_class,
                            get_point_group,
                            polar_space_group)


functions = [get_bravais_class, get_point_group, polar_space_group]


@pytest.mark.parametrize("func", functions)
def test_valid_spacegroup(func):
    func(1)


@pytest.mark.parametrize("func", functions)
def test_nonpositive_spacegroup(func):
    with pytest.raises(ValueError, match="positive"):
        func(0)


@pytest.mark.parametrize("func", functions)
def test_bad_spacegroup(func):
    with pytest.raises(ValueError, match="Bad"):
        func(400)
