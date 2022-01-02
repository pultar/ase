import pytest
from ase.atoms import Atoms

calc = pytest.mark.calculator


@pytest.fixture
def ch4():
    positions = [[0.000000, 0.000000, 0.100000],
                [0.682793, 0.682793, 0.682793],
                [-0.682793, -0.682793, 0.68279],
                [-0.682793, 0.682793, -0.682793],
                [0.682793, -0.682793, -0.682793]]
    return Atoms('CH4', positions=positions)


@pytest.fixture
def ch4_parameters():
    params = {
        "directory": "/home/schinavro/test/ASE_OMX/new_io/",
        "system_currentdirectory": "/home/schinavro/test/ASE_OMX/new_io/test/",
        "system_name": "ch4",
        "data_path": "/appl/openMX/openmx3.9/DFT_DATA19",
        "scf_energycutoff": 300,
        "scf_criterion": 0.0001,
        "scf_xctype": "gga-pbe",
        "scf_mixing_type": "rmm-diis",
        "definition_of_atomic_species": [['C', 'C5.0-s1p3', 'C_PBE19'],
                                         ['H', 'H5.0-s1', 'H_PBE19']]}
    return params


@pytest.fixture
def md_parameters():
    md_parameters = {
        "md_type": 'diis',                # Nomd|Opt|NVE|NVT_VS|NVT_NH
                                          # Constraint_Opt|DIIS
        "md_maxiter": 5,                  # default=1
        "md_timestep": 1.0,               # default=0.5 (fs)
        "md_opt_criterion": 1.0e-4        # default=1.0e-4 (Hartree/Bohr)
    }
    return md_parameters


@pytest.mark.calculator_lite
@calc('openmx')
def test_molecule_static(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        "scf_eigenvaluesolver": "cluster"
    }
    ch4.calc = factory.calc(**parameters)
    ch4.get_potential_energy()


@calc('openmx')
def test_molecule_md(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        **md_parameters,
        "scf_eigenvaluesolver": "cluster"
    }
    ch4.calc = factory.calc(**parameters)
    ch4.get_potential_energy()


@pytest.mark.calculator_lite
@calc('openmx')
def test_crystal_static(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        **md_parameters,
        "scf_eigenvaluesolver": "band"
    }
    ch4.cell = [10, 10, 10]
    ch4.calc = factory.calc(**parameters)
    ch4.get_potential_energy()


@pytest.mark.calculator_lite
@calc('openmx')
def test_crystal_static_stress(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        **md_parameters,
        "scf_eigenvaluesolver": "band"
    }
    ch4.cell = [10, 10, 10]
    ch4.calc = factory.calc(**parameters)
    ch4.get_stress()


@calc('openmx')
def test_crystal_md(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        **md_parameters,
        "scf_eigenvaluesolver": "band"
    }
    ch4.cell = [10, 10, 10]
    ch4.calc = factory.calc(**parameters)
    ch4.get_potential_energy()


@calc('openmx')
def test_crystal_md_stress(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        **md_parameters,
        "scf_eigenvaluesolver": "band"
    }
    ch4.cell = [10, 10, 10]
    ch4.calc = factory.calc(**parameters)
    ch4.get_stress()


@calc('openmx')
def test_band(factory, ch4, common_parameters, md_parameters):
    parameters = {
        **common_parameters,
        "scf_eigenvaluesolver": "band",
        "scf_kgrid": (3, 3, 3),
        "band_dispersion": True,
        "band_kpath_unitcell": [[10., 0., 0.],
                                [0.00, 10., 0.00],
                                [0.00, 0.00, 10.]],
        "band_nkpath": 3,
        "band_kpath": [[5, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 'g', 'X'],
                       [5, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 'X', 'W'],
                       [5, 1.0, 0.5, 0.0, 0.5, 0.5, 0.5, 'W', 'L']]
    }
    ch4.cell = [10, 10, 10]
    ch4.calc = factory.calc(**parameters)
    ch4.get_potential_energy()
