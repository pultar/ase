"""Test methods of OctopusTemplate
TODO(AlexB) See issue 22 w.r.t. migrating this as an ASE merge
"""
import pytest
from pathlib import Path
import numpy as np
import re
import os

from ase.calculators.octopus import OctopusTemplate, OctopusProfile
from ase.atoms import Atoms


@pytest.fixture()
def calculator() -> OctopusTemplate:
    """Octopus calculator"""
    return OctopusTemplate()


@pytest.fixture()
def profile():
    """Octopus profile"""
    # TODO(Alex). How to set this. Machine-specific
    binary = '/usr/local/bin/octopus'
    if not os.path.exists(binary):
        raise FileNotFoundError(f"Octopus binary does not exist {binary}")
    return OctopusProfile(binary)


@pytest.fixture()
def silicon() -> Atoms:
    """ASE atoms object"""

    # Angstrom
    al = 5.397608

    lattice = al * np.array([[0.0,  0.5,  0.5],
                             [0.5,  0.0,  0.5],
                             [0.5,  0.5,  0.0]])

    atoms = Atoms(['Si', 'Si'],
                  scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
                  cell=lattice,
                  pbc=[1, 1, 1])

    return atoms


def test_octopus_template_write_input(tmp_path: Path,
                                      set_test_dir,
                                      calculator,
                                      silicon):

    # Location to write input to
    set_test_dir(tmp_path)

    # See if defining parameters in this manner works
    parameters = {"FromScratch": "yes",
                  "PeriodicDimensions": 3,
                  "BoxShape": "parallelepiped",
                  "Spacing": 0.5,
                  "KPointsGrid": '4 4 4',
                  "Preconditioner": 'pre_multigrid',
                  "ConvEigenError": "yes",
                  "ConvRelDens": 1.e-9,
                  "EigensolverTolerance": 1.e-8
                  }

    calculator.write_input('.', silicon, parameters, ['energy'])

    oct_input = tmp_path / 'inp'
    assert oct_input.exists(), "Octopus input file not written"

    # Parse result and check for "some" keys
    with open(oct_input) as fid:
        file_string = fid.read()

    matches = re.findall(r"(?i)\bPeriodicdimensions\b\s=.*", file_string)
    assert matches, "Expect key in file"
    assert matches[0] == 'periodicdimensions = 3', 'Octopus input is case insensitive'


def test_octopus_template_read_results(tmp_path: Path,
                                       set_test_dir,
                                       calculator,
                                       profile):

    # Location to write input to
    set_test_dir(tmp_path)

    input = """boxshape = parallelepiped
%LatticeParameters
 1.0 | 1.0 | 1.0
%

%LatticeVectors
 0.0 | 5.100000427313206 | 5.100000427313206
 5.100000427313206 | 0.0 | 5.100000427313206
 5.100000427313206 | 5.100000427313206 | 0.0
%

periodicdimensions = 3
fromscratch = yes
periodicdimensions = 3
spacing = 0.5
kpointsgrid = 4 4 4
preconditioner = pre_multigrid
conveigenerror = yes
convreldens = 1e-09
eigensolvertolerance = 1e-08

%ReducedCoordinates
 'Si' | 0.0 | 0.0 | -0.0
 'Si' | 0.25 | 0.25 | 0.25"""

    # Write file
    with open(tmp_path / 'inp', 'w') as fid:
        fid.write(input)

    # Execute Octopus
    # TODO(Alex) Would be nice to be able to parse what's returned from stdout/stderr
    calculator.execute(tmp_path, profile)

    # Parse results
    results = calculator.read_results(tmp_path)

    expected_keys = {'ibz_k_points', 'k_point_weights', 'nspins', 'nkpts', 'nbands', 'eigenvalues',
                     'occupations', 'energy', 'free_energy', 'forces'}

    assert set(results.keys()) == expected_keys
