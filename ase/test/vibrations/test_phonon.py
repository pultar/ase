import pytest

import numpy as np

import ase.build
from ase.calculators.emt import EMT
from ase.phonons import Phonons


def test_emt_phonons(testdir):
    """Test phonons with EMT Au-Pt alloy"""
    atoms = ase.build.bulk('Au') * [2, 1, 1]
    atoms[1].symbol = 'Pt'

    phonons = Phonons(atoms, EMT(), supercell=[2, 4, 4])
    phonons.run()
    phonons.read()

    pdos = phonons.get_pdos()

    assert pdos[0].info == {'symbol': 'Au', 'index': '0'}
    assert pdos[1].info == {'symbol': 'Pt', 'index': '1'}

    for spectrum in pdos:
        assert np.sum(spectrum.get_weights()) == pytest.approx(3.)

    # Pt should be biased towards low-freq and Au high-freq
    midpoint = np.median(pdos[0].get_energies())
    low_energies = pdos[0].get_energies() < midpoint
    assert np.sum(pdos[0].get_weights()[low_energies]) < 1.5
    assert np.sum(pdos[1].get_weights()[low_energies]) > 1.5    
