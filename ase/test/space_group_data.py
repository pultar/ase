import pytest
from ase.spacegroup import (get_bravais_class,
                            get_point_group,
                            polar_space_group)
import ase.lattice


functions = [get_bravais_class, get_point_group, polar_space_group]


@pytest.mark.parametrize("func,answer",
                         zip(functions,
                             [ase.lattice.FCC, '4/m -3 2/m', False]))
def test_valid_spacegroup(func, answer):
    assert func(225) == answer


@pytest.mark.parametrize("func", functions)
def test_nonpositive_spacegroup(func):
    with pytest.raises(ValueError, match="positive"):
        func(0)


@pytest.mark.parametrize("func", functions)
def test_bad_spacegroup(func):
    with pytest.raises(ValueError, match="Bad"):
        func(400)
