import pytest

from ase.build import bulk
from ase.calculators.dftb import Dftb

def test_kpoint_params():
    atoms = bulk('Si')

    params, _ = Dftb._get_kpts_parameters(None, atoms)
    assert params == {}

    params, _ = Dftb._get_kpts_parameters({'path': 'GXG', 'npoints': 5}, atoms)
    assert params == {
        'Hamiltonian_KPointsAndWeights_': 'Klines ',
        'Hamiltonian_KPointsAndWeights_empty000000000': '1 0.0 0.0 0.0',
        'Hamiltonian_KPointsAndWeights_empty000000001': '1 0.5 0.0 0.5',
        'Hamiltonian_KPointsAndWeights_empty000000002': '1 0.33333333333333337 0.0 0.33333333333333337',
        'Hamiltonian_KPointsAndWeights_empty000000003': '1 0.16666666666666669 0.0 0.16666666666666669',
        'Hamiltonian_KPointsAndWeights_empty000000004': '1 0.0 0.0 0.0'}
    
    params, _ = Dftb._get_kpts_parameters({'size': [2, 3, 4], 'gamma': True}, atoms)
    assert params == {
        'Hamiltonian_KPointsAndWeights_': 'SupercellFolding ',
        'Hamiltonian_KPointsAndWeights_empty000': '2 0 0',
        'Hamiltonian_KPointsAndWeights_empty001': '0 3 0',
        'Hamiltonian_KPointsAndWeights_empty002': '0 0 4',
        'Hamiltonian_KPointsAndWeights_empty003': '1.0 0 1.0'}

    params, _ = Dftb._get_kpts_parameters([2, 3, 4], atoms)
    assert params == {
        'Hamiltonian_KPointsAndWeights_': 'SupercellFolding ',
        'Hamiltonian_KPointsAndWeights_empty000': '2 0 0',
        'Hamiltonian_KPointsAndWeights_empty001': '0 3 0',
        'Hamiltonian_KPointsAndWeights_empty002': '0 0 4',
        'Hamiltonian_KPointsAndWeights_empty003': '0.5 0.0 0.5'}

    params, _ = Dftb._get_kpts_parameters([[0.1, 0.2, 0.3],
                                           [0.2, 0.3, 0.4],
                                           [0.3, 0.4, 0.5]],
                                          atoms)
    assert params == {
        'Hamiltonian_KPointsAndWeights_': '',
        'Hamiltonian_KPointsAndWeights_empty000000000': '0.1 0.2 0.3 1.0',
        'Hamiltonian_KPointsAndWeights_empty000000001': '0.2 0.3 0.4 1.0',
        'Hamiltonian_KPointsAndWeights_empty000000002': '0.3 0.4 0.5 1.0'}

    with pytest.raises(ValueError):
        Dftb._get_kpts_parameters('a string', atoms)

