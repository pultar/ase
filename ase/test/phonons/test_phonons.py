import numpy as np
import pytest
from pathlib import Path

import ase.io
from ase.atoms import Atoms
from ase.build import bulk, molecule
from ase.phonons import Phonons, PhononsData
from ase.calculators.emt import EMT
from ase import units


@pytest.mark.usefixtures("testdir")
class TestPhonons:
    def check_set_atoms(self, atoms, set_atoms, expected_atoms):
        """ Perform a test that .set_atoms() only displaces the expected atoms. """
        phonons = Phonons(atoms, EMT())
        phonons.set_atoms(set_atoms)

        # TODO: For now, because there is no public API to iterate over/inspect
        #       displacements, we run and check the number of files in the cache.
        #       Later when the requisite API exists, we should use it both to
        #       check the actual atom indices and to avoid computation.
        phonons.run()
        assert len(phonons.cache) == 6 * len(expected_atoms) + 1

    def test_set_atoms_indices(self):
        self.check_set_atoms(molecule('CO2'), set_atoms=[0, 1], expected_atoms=[0, 1])

    def test_set_atoms_symbol(self):
        self.check_set_atoms(molecule('CO2'), set_atoms=['O'], expected_atoms=[1, 2])

    def test_check_eq_forces(self):
        phonons = self.diamond_sc_phonons()
        fmin, fmax, _i_min, _i_max = phonons.check_eq_forces()
        assert fmin < fmax

    def h2_phonons(self):
        atoms = molecule('H2')
        phonons = Phonons(atoms, EMT())
        phonons.run()
        return phonons

    def al_phonons(self):
        """The cheapest computation possible, with only one atom.  Can only handle gamma point."""
        atoms = bulk('Al')
        phonons = Phonons(atoms, EMT())
        phonons.run()
        return phonons

    def al_sc_phonons(self):
        """The cheapest supercell computation possible, with only one atom.  Can only handle GX."""
        atoms = bulk('Al')
        phonons = Phonons(atoms, EMT(), supercell=(2, 1, 1))
        phonons.run()
        return phonons

    def diamond_sc_phonons(self):
        """A very small supercell of diamond only suitable for phonons along the GX direction."""
        atoms = bulk('C')
        phonons = Phonons(atoms, EMT(), supercell=(2, 1, 1))
        phonons.run()
        return phonons

    # For testing that methods work even if calc=None.
    # (which should be true for most methods, once the computation has been run)
    def reread_without_calc(self, phonons):
        return Phonons(atoms=phonons.atoms, name=phonons.name, delta=phonons.delta,
                       center_refcell=phonons.center_refcell, supercell=phonons.supercell,
                       calc=None)

    # Regression test for #953;  data stored for eq should resemble data for displacements
    def test_check_consistent_format(self):
        phonons = self.h2_phonons()
        # Check that the data stored for `eq` is shaped like the data stored for displacements.
        eq_data = phonons.cache['eq']
        disp_data = phonons.cache['0x-']
        assert isinstance(eq_data, dict) and isinstance(disp_data, dict)
        assert set(eq_data) == set(disp_data), "dict keys mismatch"
        for array_key in eq_data:
            assert eq_data[array_key].shape == disp_data[array_key].shape, array_key

    def test_calc_none_in_get_phonons(self):
        phonons = self.reread_without_calc(self.h2_phonons())
        # Test that this does not throw
        phonons.get_phonons()

    def test_write_modes_basic_properties(self):
        phonons = self.reread_without_calc(self.al_sc_phonons())
        X = phonons.atoms.cell.bandpath().special_points['X']

        phonons.write_modes(X, branches=1, nimages=5)
        for i in range(3):
            assert Path(f'phonon.mode.{i}.traj').is_file() == (i == 1)

        traj = ase.io.read('phonon.mode.1.traj', index=':')
        assert len(traj) == 5

        # negative branch
        phonons.write_modes(X, branches=-1, nimages=5)
        for i in range(3):
            assert Path(f'phonon.mode.{i}.traj').is_file() == (i in [1, 2])

        # write all
        phonons.write_modes(X, branches=range(3), nimages=5)
        for i in range(3):
            assert Path(f'phonon.mode.{i}.traj').is_file()

    def test_write_modes_positions(self):
        phonons = self.reread_without_calc(self.diamond_sc_phonons())

        branch = 5  # this mode is non-degenerate along GX, leading to more reliable tests
        X = phonons.atoms.cell.bandpath().special_points['X']
        X2 = X - [1, 0, 0]
        phonons.write_modes([0, 0, 0], branches=branch, nimages=5, name='phonon.G')
        phonons.write_modes(X, branches=branch, nimages=5, name='phonon.X')
        phonons.write_modes(X2, branches=branch, nimages=5, name='phonon.X2')
        for i in range(6):
            assert Path(f'phonon.G.mode.{i}.traj').is_file() == (i == 5)

        G_traj = ase.io.read(f'phonon.G.mode.{branch}.traj', index=':')
        X_traj = ase.io.read(f'phonon.X.mode.{branch}.traj', index=':')
        X2_traj = ase.io.read(f'phonon.X2.mode.{branch}.traj', index=':')

        # check for common bug where first frame == last frame in a looping animation
        assert G_traj[0].get_positions() != pytest.approx(G_traj[-1].get_positions())

        # different Q should give different modes
        assert G_traj[1].get_positions() != pytest.approx(X_traj[1].get_positions())

        # Ensure that the above != tests did not succeed solely due to poor tolerances.
        #
        # For this, we look at a Q point that is equivalent to X, which should have the
        # same bands (up to degeneracy, which was already accounted for).
        assert X_traj[1].get_positions() == pytest.approx(X2_traj[1].get_positions())


class TestPhononsData:
    @pytest.fixture
    def cumulene_data(self):
        return {
            # Cumulene
            'atoms': Atoms(
                symbols='C',
                positions=np.array([[0.63496908, 5, 5]]),
                cell=np.diag([1.2699381536373493, 10, 10]),
            ),
            'center_refcell': False,
            # A supercell of even dimension is necessary to see the instability.
            'supercell': (4, 1, 1),
            'force_constants': np.array(
                [[[1.61519168e+01, 5.66053683e-05, -7.02876989e-06],
                  [8.76248824e-05, 2.46606042e+00, 8.40593921e-05],
                  [-2.79993898e-06, 8.61532247e-05, 2.46600294e+00]],
                 [[6.15251715e+01, -1.12466350e-05, 1.23304509e-05],
                  [-4.10113196e-05, -1.30729155e+00, -4.20297104e-05],
                  [-9.53052141e-06, -4.20296833e-05, -1.30732920e+00]],
                 [[-1.39202260e+02, -4.34741367e-06, 4.22884040e-06],
                  [-3.53669278e-05, 1.48522678e-01, 1.62077469e-12],
                  [9.48831265e-12, -2.09383097e-06, 1.48655450e-01]],
                 [[6.15251715e+01, -4.10113196e-05, -9.53052141e-06],
                  [-1.12466350e-05, -1.30729155e+00, -4.20296833e-05],
                  [1.23304509e-05, -4.20297104e-05, -1.30732920e+00]]],
            ).reshape(4, 1, 3, 1, 3),
            'ref_frequencies': {
                # The G modes are all acoustic.  Their actual numerical magnitude is the arbitrary
                # result of catastrophic cancelation and should be ignored.
                'G': np.array([0.0, 0.0, 0.0]),  # 0 to force tests to use an absolute tolerance
                # At Z there is an imaginary mode (a manifestation of Peierls distortion)
                'Z': np.array([-2360.46142, 344.074268, 344.086389]),
            },
        }

    def phdata_from_data(self, data):
        return PhononsData(atoms=data['atoms'], supercell=data['supercell'], force_constants=data['force_constants'],
                           center_refcell=data['center_refcell'])

    @pytest.fixture
    def cumulene_phdata(self, cumulene_data):
        return self.phdata_from_data(cumulene_data)

    def test_init(self, cumulene_data):
        # Check that init runs without error; properties are checked in other
        # methods using the (identical) cumulene_phdata fixture
        PhononsData(cumulene_data['atoms'], cumulene_data['force_constants'],
                    supercell=cumulene_data['supercell'], center_refcell=cumulene_data['center_refcell'])

    def test_energies_and_modes(self, cumulene_data, cumulene_phdata):
        data, phdata = cumulene_data, cumulene_phdata

        atoms = data['atoms']
        path = atoms.cell.bandpath('GZ', npoints=2)
        bs = phdata.get_band_structure(path)
        computed_frequencies = {
            'G': bs.energies[0][0] / units.invcm,
            'Z': bs.energies[0][1] / units.invcm,
        }
        assert computed_frequencies['G'] == pytest.approx(data['ref_frequencies']['G'], abs=1e-1)
        assert computed_frequencies['Z'] == pytest.approx(data['ref_frequencies']['Z'], rel=1e-5)
