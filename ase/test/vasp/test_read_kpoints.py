"""
Check the many ways of reading KPOINTS
"""
import os
import pytest
from numpy.testing import assert_array_almost_equal
from ase.calculators.vasp.create_input import GenerateVaspInput

mp_grid_txt = """Any Comment On This Line
              0
              Monkhorst-Pack
              4 4 4
              0 0 0
              """

mp_gamma_txt = """Also ANY Comment On This Line
               0
               g
               2 2 2
               0 0 0
               """

mp_auto_txt = """Any Comment On This Line
              0
              Auto
              20
              """

list_recip_txt = """Any Comment On This Line
               5
               Reciprocal
               0.  0. 0.
               0.1 0. 0.
               0.2 0. 0.
               0.5 0. 0.5
               0.  0. 0.5
               """

list_cart_txt = """Any Comment On This Line
              4
              k
              0.1 0. 0.
              0.2 0. 0.
              0.5 0. 0.5
              0.  0. 0.5

              """

line_cart_txt = """Any Comment On This Line
                3
                line
                CART
                0. 0. 0.
                1. 0. 0.
                1. 0. 0.
                1.  0. 1.
                0. 1. 0.
                0. 0. 0.

                """

line_recip_txt = """Any Comment On This Line
                3
                line
                rec
                0. 0. 0.
                1. 0. 0.
                1. 0. 0.
                1.  0. 1.
                0. 1. 0.
                0. 0. 0.

                """

interp_points = [[0., 0., 0.], [0.5, 0., 0.], [1., 0., 0.],
                 [1., 0., 0.], [1., 0., 0.5], [1., 0., 1.],
                 [0., 1., 0.], [0., 0.5, 0.], [0., 0., 0.]]


class TestReadKpoints:
    @classmethod
    def setup_class(self):
        self.outfile = 'KPOINTS'

    @classmethod
    def teardown_class(self):
        if os.path.isfile(self.outfile):
            os.remove(self.outfile)

    def test_read_mp_grid(self):
        with open(self.outfile, 'w') as fd:
            fd.write(mp_grid_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert_array_almost_equal(vaspinput.input_params['kpts'], [4, 4, 4])
        assert not vaspinput.input_params['gamma']

    def test_read_gamma_grid(self):
        with open(self.outfile, 'w') as fd:
            fd.write(mp_gamma_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert_array_almost_equal(vaspinput.input_params['kpts'], [2, 2, 2])
        assert vaspinput.input_params['gamma']

    def test_read_auto_grid(self):
        with open(self.outfile, 'w') as fd:
            fd.write(mp_auto_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert_array_almost_equal(vaspinput.input_params['kpts'], [20])
        assert not vaspinput.input_params['gamma']

    def test_read_recip_list(self):
        with open(self.outfile, 'w') as fd:
            fd.write(list_recip_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert len(vaspinput.input_params['kpts']) == 5
        assert_array_almost_equal(vaspinput.input_params['kpts'],
                                  [[0., 0., 0.],
                                   [0.1, 0., 0.],
                                   [0.2, 0., 0.],
                                   [0.5, 0., 0.5],
                                   [0., 0., 0.5]])
        assert vaspinput.input_params['reciprocal']

    def test_read_cart_list(self):
        with open(self.outfile, 'w') as fd:
            fd.write(list_cart_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert_array_almost_equal(vaspinput.input_params['kpts'],
                                  [[0.1, 0., 0.],
                                   [0.2, 0., 0.],
                                   [0.5, 0., 0.5],
                                   [0., 0., 0.5]])
        assert not vaspinput.input_params['reciprocal']

    def test_read_cart_lines(self):
        with open(self.outfile, 'w') as fd:
            fd.write(line_cart_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert_array_almost_equal(vaspinput.input_params['kpts'],
                                  interp_points)

        assert not vaspinput.input_params['reciprocal']

    def test_read_recip_lines(self):
        with open(self.outfile, 'w') as fd:
            fd.write(line_recip_txt)

        vaspinput = GenerateVaspInput()
        vaspinput.read_kpoints(self.outfile)

        assert_array_almost_equal(vaspinput.input_params['kpts'],
                                  interp_points)

        assert vaspinput.input_params['reciprocal']
