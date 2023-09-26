import pytest
from ase.io import read, write
from ase.build import bulk
from ase.calculators.calculator import compare_atoms
from ase.io.vasp_parsers.vasp_structure_io import (read_vasp_structure,
                                                   write_vasp_structure)
import itertools
@pytest.fixture
def atoms():
    _atoms = bulk('NaCl', crystalstructure='rocksalt', a=4.1, cubic=True)
    _atoms.wrap()
    return _atoms


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('vasp5', [True, False])
def test_read_write_roundtrip(atoms, vasp5, filename):
    write(filename, atoms, vasp5=vasp5)
    atoms_loaded = read(filename)

    assert len(compare_atoms(atoms, atoms_loaded)) == 0


@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{}, {'vasp5': True}])
def test_write_vasp5(atoms, filename, kwargs):
    """Test that we write the symbols to the POSCAR/CONTCAR
    with vasp5=True (which should also be the default)"""
    write(filename, atoms, format='vasp', **kwargs)
    with open(filename) as file:
        lines = file.readlines()
    # Test the 5th line, which should be the symbols
    assert lines[5].strip().split() == list(atoms.symbols)

@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{}, {'vasp5': True}])
def test_write_poscar(atoms, filename, kwargs):
    write_vasp_structure(filename, atoms=atoms, **kwargs)
    res = ['Cl Na\n',
           '  1.0000000000000000\n',
           '    4.0999999999999996    0.0000000000000000    0.0000000000000000\n',
           '    0.0000000000000000    4.0999999999999996    0.0000000000000000\n',
           '    0.0000000000000000    0.0000000000000000    4.0999999999999996\n',
           '  Cl Na\n',
           '  4  4\n',
           'Cartesian\n',
           '  2.0499999999999994  0.0000000000000000  0.0000000000000000\n',
           '  2.0499999999999994  2.0499999999999994  2.0499999999999994\n',
           '  0.0000000000000000  0.0000000000000000  2.0499999999999994\n',
           '  0.0000000000000000  2.0499999999999994  0.0000000000000000\n',
           '  0.0000000000000000  0.0000000000000000  0.0000000000000000\n',
           '  0.0000000000000000  2.0499999999999994  2.0499999999999994\n',
           '  2.0499999999999994  0.0000000000000000  2.0499999999999994\n',
           '  2.0499999999999994  2.0499999999999994  0.0000000000000000\n']
    with open(filename) as fil:
        for i, line in enumerate(fil.readlines()):
            for j, elem in enumerate(line.split()):
                assert elem == res[i].split()[j]
    read_vasp_structure(filename)

@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{'vasp6': True, 'potential_mapping': {'Na': 'Na_sv_GW/7aca660ab3bfaee19d1e', 'Cl': 'Cl_GW'}},
                                    {'vasp6': True,'potential_mapping': {'Na': 'Na_sv_GW/7aca66', 'Cl': 'Cl_GW/3ef'}}])
def test_write_vasp6(atoms, filename, kwargs):
    """Test that we write the symbols to the POSCAR/CONTCAR
    with vasp6=True"""
    write_vasp_structure(filename, atoms, **kwargs)
    with open(filename) as file:
        lines = file.readlines()
    # Test the 5th line, which should be the symbols in vasp 6 format
    assert lines[5].strip().split() == [kwargs['potential_mapping'][el][:14] for el in sorted(list(set(atoms.symbols)))]

@pytest.mark.parametrize('filename', ['POSCAR', 'CONTCAR'])
@pytest.mark.parametrize('kwargs', [{'vasp6': True, 'potential_mapping': {'Na': 'Na_sv_GW', 'Cl': 'Cl_GW/3ef'}}])
def test_write_poscar_vasp6(atoms, filename, kwargs):
    write_vasp_structure(filename, atoms=atoms, **kwargs)
    res = ['Cl Na\n',
           '  1.0000000000000000\n',
           '    4.0999999999999996    0.0000000000000000    0.0000000000000000\n',
           '    0.0000000000000000    4.0999999999999996    0.0000000000000000\n',
           '    0.0000000000000000    0.0000000000000000    4.0999999999999996\n',
           '  Cl_GW/3ef Na_sv_GW\n',
           '  4  4\n',
           'Cartesian\n',
           '  2.0499999999999994  0.0000000000000000  0.0000000000000000\n',
           '  2.0499999999999994  2.0499999999999994  2.0499999999999994\n',
           '  0.0000000000000000  0.0000000000000000  2.0499999999999994\n',
           '  0.0000000000000000  2.0499999999999994  0.0000000000000000\n',
           '  0.0000000000000000  0.0000000000000000  0.0000000000000000\n',
           '  0.0000000000000000  2.0499999999999994  2.0499999999999994\n',
           '  2.0499999999999994  0.0000000000000000  2.0499999999999994\n',
           '  2.0499999999999994  2.0499999999999994  0.0000000000000000\n']
    with open(filename) as fil:
        for i, line in enumerate(fil.readlines()):
            for j, elem in enumerate(line.split()):
                assert elem == res[i].split()[j]
    read_vasp_structure(filename)