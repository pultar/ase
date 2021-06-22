"""
Check the many ways of reading KPOINTS
"""
from collections import namedtuple
import os
import tempfile

from numpy.testing import assert_array_almost_equal
import pytest

from ase.calculators.vasp.create_input import GenerateVaspInput

KpointTestData = namedtuple('KpointTestData',
                            ['name', 'text', 'gamma', 'reciprocal', 'kpts'])

all_tests = [KpointTestData("MP grid",
                            """Any Comment On This Line
                            0
                            Monkhorst-Pack
                            4 4 4
                            0 0 0
                            """, False, None, [4, 4, 4]),
             KpointTestData("Gamma-centered grid",
                            """Also ANY Comment On This Line
                            0
                            g
                            2 2 2
                            0 0 0
                            """, True, None, [2, 2, 2]),
             KpointTestData("Auto",
                            """Any Comment On This Line
                            0
                            Auto
                            20
                            """, False, None, [20]),
             KpointTestData("Reciprocal points",
                            """Any Comment On This Line
                            5
                            Reciprocal
                            0.  0. 0.
                            0.1 0. 0.
                            0.2 0. 0.
                            0.5 0. 0.5
                            0.  0. 0.5
                            """, None, True, [[0., 0., 0.],
                                              [0.1, 0., 0.],
                                              [0.2, 0., 0.],
                                              [0.5, 0., 0.5],
                                              [0., 0., 0.5]]),
             KpointTestData("Cartesian points",
                            """Any Comment On This Line
                            4
                            k
                            0.1 0. 0.
                            0.2 0. 0.
                            0.5 0. 0.5
                            0.  0. 0.5

                            """, None, False, [[0.1, 0., 0.],
                                               [0.2, 0., 0.],
                                               [0.5, 0., 0.5],
                                               [0., 0., 0.5]]),
             KpointTestData("Cartesian lines",
                            """Any Comment On This Line
                            3
                            line
                            CART
                            0. 0. 0.
                            1. 0. 0.
                            1. 0. 0.
                            1.  0. 1.
                            0. 1. 0.
                            0. 0. 0.

                            """, None, False,
                            [[0., 0., 0.], [0.5, 0., 0.], [1., 0., 0.],
                             [1., 0., 0.], [1., 0., 0.5], [1., 0., 1.],
                             [0., 1., 0.], [0., 0.5, 0.], [0., 0., 0.]]),
             KpointTestData("Reciprocal lines",
                            """Any Comment On This Line
                            3
                            line
                            rec
                            0. 0. 0.
                            1. 0. 0.
                            1. 0. 0.
                            1.  0. 1.
                            0. 1. 0.
                            0. 0. 0.

                            """, None, True,
                            [[0., 0., 0.], [0.5, 0., 0.], [1., 0., 0.],
                             [1., 0., 0.], [1., 0., 0.5], [1., 0., 1.],
                             [0., 1., 0.], [0., 0.5, 0.], [0., 0., 0.]])
             ]


@pytest.mark.parametrize('test_data', all_tests)
def test_read_kpoints(test_data):
    with tempfile.NamedTemporaryFile(mode='wt', delete=False) as fd:
        fd.write(test_data.text)
        tmpfilename = fd.name

    try:
        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(tmpfilename)
    finally:
        os.remove(tmpfilename)

    if test_data.gamma is not None:
        assert vaspinput.input_params['gamma'] == test_data.gamma

    if test_data.reciprocal is not None:
        assert vaspinput.input_params['reciprocal'] == test_data.reciprocal

    assert_array_almost_equal(vaspinput.input_params['kpts'], test_data.kpts)
