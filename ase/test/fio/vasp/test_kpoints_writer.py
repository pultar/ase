from ase.build import bulk
from ase.io.vasp_parsers.kpoints_writer import write_kpoints
from ase.dft.kpoints import RegularGridKPoints, WeightedKPoints
from unittest.mock import mock_open, patch
from collections import OrderedDict


def check_write_kpoints_file(parameters, expected_output):
    mock = mock_open()
    with patch("ase.io.vasp_parsers.kpoints_writer.open", mock):
        write_kpoints("directory", parameters)
        mock.assert_called_once_with("directory/KPOINTS", "w")
        kpoints = mock()
        kpoints.write.assert_called_once_with(expected_output)


def test_kpoints_Auto_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Auto
10"""
    parameters = {"Auto": 10}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"AUTO": 10}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"auto": 10}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Gamma_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Gamma
4 4 4"""
    parameters = {"Gamma": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"gamma": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"GAMMA": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)

def test_kpoints_Gamma_mode_with_shift():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Gamma
4 4 4
0.5 0.5 0.5"""
    parameters = {"Gamma": {'size': [4, 4, 4], "shift": [0.5, 0.5, 0.5]}}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"gamma": {'size': [4, 4, 4], "shift": [0.5, 0.5, 0.5]}}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"GAMMA": {'size': [4, 4, 4], "shift": [0.5, 0.5, 0.5]}}
    check_write_kpoints_file(parameters, expected_output)

def test_kpoints_Monkhorst_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Monkhorst
4 4 4"""
    parameters = {"Monkhorst": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"monkhorst": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"MONKHORST": [4, 4, 4]}
    check_write_kpoints_file(parameters, expected_output)

def test_kpoints_Monkhorst_mode_with_shift():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Monkhorst
4 4 4
0.5 0.5 0.5"""
    parameters = {"Monkhorst": {'size': [4, 4, 4], 'shift': [0.5, 0.5, 0.5]}}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"monkhorst": {'size': [4, 4, 4], 'shift': [0.5, 0.5, 0.5]}}
    check_write_kpoints_file(parameters, expected_output)
    parameters = {"MONKHORST": {'size': [4, 4, 4], 'shift': [0.5, 0.5, 0.5]}}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Line_reciprocal_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
40
Line
Reciprocal
0 0 0
0.5 0.5 0
0.5 0.5 0
0.5 0.75 0.25
0.5 0.75 0.25
0 0 0"""
    parameters = {
        "Line": 40,
        "reciprocal": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.25],
            [0, 0, 0],
        ],
    }
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Line_cartesian_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
40
Line
Cartesian
0 0 0
0.5 0.5 0
0.5 0.5 0
0.5 0.75 0.25
0.5 0.75 0.25
0 0 0"""
    parameters = {
        "Line": 40,
        "CaRTESIAN": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.25],
            [0, 0, 0],
        ],
    }
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_Line_mode_incorrect_order():
    expected_output = """KPOINTS created by Atomic Simulation Environment
40
Line
Reciprocal
0 0 0
0.5 0.5 0
0.5 0.5 0
0.5 0.75 0.25
0.5 0.75 0.25
0 0 0"""
    parameters = {
        "reciprocal": [
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0.5, 0],
            [0.5, 0.75, 0.25],
            [0.5, 0.75, 0.25],
            [0, 0, 0],
        ],
        "Line": 40,
    }
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_explicit_reciprocal_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
4
Reciprocal
0 0 0
0 0 0.5
0 0.5 0.5
0.5 0.5 0.5"""
    coordinates = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
    parameters = {"Reciprocal": coordinates}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_explicit_cartesian_mode():
    expected_output = """KPOINTS created by Atomic Simulation Environment
4
Cartesian
0 0 0
0 0 0.5
0 0.5 0.5
0.5 0.5 0.5"""
    coordinates = [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0.5]]
    parameters = {"Cartesian": coordinates}
    check_write_kpoints_file(parameters, expected_output)


def test_kpoints_explicit_str_file():
    expected_output = """KPOINTS created by Atomic Simulation Environment
4
Cartesian
0 0 0
0 0 0.5
0 0.5 0.5
0.5 0.5 0.5"""
    check_write_kpoints_file(expected_output, expected_output)


def test_kpoints_empty_parameters():
    mock = mock_open()
    with patch("ase.io.vasp_parsers.kpoints_writer.open", mock):
        write_kpoints("directory", None)
        mock.assert_not_called()


def test_kpoints_RegularGridKPoints():
    expected_output = """KPOINTS created by Atomic Simulation Environment
0
Monkhorst
4 4 4
0.5 0.5 0.5"""
    check_write_kpoints_file(RegularGridKPoints([4, 4, 4], [0.5, 0.5, 0.5]), expected_output)

def test_weights_RegularGridKPoints():
    expected_output = """KPOINTS created by Atomic Simulation Environment
8
Reciprocal
0.125 0.125 0.125 8.0
0.375 0.125 0.125 16.0
0.375 0.375 0.125 8.0
0.125 0.375 0.125 16.0
0.125 0.125 0.375 8.0
0.375 0.125 0.375 16.0
0.375 0.375 0.375 8.0
0.125 0.375 0.375 16.0"""
    kpts_list = [[0.125, 0.125, 0.125], [0.375, 0.125, 0.125], [0.375, 0.375, 0.125], [0.125, 0.375, 0.125], [0.125, 0.125, 0.375], [0.375, 0.125, 0.375], [0.375, 0.375, 0.375], [0.125, 0.375, 0.375]]
    weights_list = [8, 16, 8, 16, 8, 16, 8, 16]
    check_write_kpoints_file(WeightedKPoints(kpts_list, weights_list), expected_output)


def test_kpoints_BandPathKPoints():
    expected_output = """KPOINTS created by Atomic Simulation Environment
12
Reciprocal
0.0 0.0 0.0 1.0
0.07142857142857142 0.0 0.07142857142857142 1.0
0.14285714285714285 0.0 0.14285714285714285 1.0
0.21428571428571427 0.0 0.21428571428571427 1.0
0.2857142857142857 0.0 0.2857142857142857 1.0
0.3571428571428571 0.0 0.3571428571428571 1.0
0.42857142857142855 0.0 0.42857142857142855 1.0
0.5 0.0 0.5 1.0
0.5 0.0625 0.5625 1.0
0.5 0.125 0.625 1.0
0.5 0.1875 0.6875 1.0
0.5 0.25 0.75 1.0"""
    atoms = bulk('Si', crystalstructure='fcc', a=3.9)
    bandpath = atoms.cell.bandpath(npoints=12, path='GXW')
    check_write_kpoints_file(bandpath, expected_output)

