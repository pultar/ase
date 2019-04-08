#!coding=utf-8
"""
Some tests of `ase.calculators.lammpsrun.LAMMPS`.
"""
from __future__ import print_function, absolute_import

import unittest
import shutil
import nose
import numpy as np

from ase.calculators.lammpsrun import LAMMPS
from ase.io import read, Trajectory
from ase.units import GPa
from os.path import dirname, join, exists
from nose.tools import assert_equal, assert_almost_equal


__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


class Ni4MoBoxRelaxTest(unittest.TestCase):

    def setUp(self) -> None:
        """
        The setup function.
        """
        self.local_dir = dirname(__file__)
        self.tmp_dir = join(self.local_dir, 'lammps')
        self.pot_zjw04_file = join(self.local_dir, 'MoNi_Zhou04.eam.alloy')
        self.traj_file = join(self.local_dir, 'out.traj')
        self.crystal = read(join(self.local_dir, 'Ni4Mo.poscar'))

    def test_box_relax(self):
        """
        The main test.
        """
        calc = LAMMPS(files=[self.pot_zjw04_file], tmp_dir=self.tmp_dir,
                      keep_alive=True, keep_tmp_files=True, no_data_file=False)

        params = {
            'pair_style': 'eam/alloy',
            'pair_coeff': ['* * MoNi_Zhou04.eam.alloy Mo Ni'],
            'minimize': '1e-8 1e-8 10000 100000',
            'fix': ['3 all box/relax aniso 0.0 vmax 0.0001']
        }
        calc.parameters = params
        calc.trajectory_out = Trajectory(self.traj_file, 'w')

        cryst = self.crystal.copy()

        calc.atoms = cryst
        calc.calculate(cryst)

        traj = Trajectory(self.traj_file)
        assert_equal(len(traj), 154)

        for index, atoms in enumerate(traj):
            assert_equal(atoms.get_potential_energy(),
                         calc.thermo_content[index]['etotal'])

            assert_almost_equal(atoms.get_volume(),
                                calc.thermo_content[index]['vol'],
                                delta=1e-6)

        rotation_ase2lammps = calc.prism.R

        #  5.33993e-10  1.06569e-09  5.2698e-10
        # -0.0335347    0.0172581   -0.409472
        #  0.375029    -0.163624    -0.0378692
        # -0.375029     0.163624     0.0378692
        #  0.0335347   -0.0172581    0.409472

        forces = traj[10].get_forces()
        lmp_forces = np.dot(forces, rotation_ase2lammps)

        assert_almost_equal(lmp_forces[0, 0], 0.0, delta=1e-8)
        assert_almost_equal(lmp_forces[0, 1], 0.0, delta=1e-8)
        assert_almost_equal(lmp_forces[0, 2], 0.0, delta=1e-8)
        assert_almost_equal(lmp_forces[1, 0], -0.0335347, delta=1e-6)
        assert_almost_equal(lmp_forces[2, 0], 0.375029, delta=1e-6)
        assert_almost_equal(lmp_forces[3, 0], -0.375029, delta=1e-8)
        assert_almost_equal(lmp_forces[1, 1], 0.0172581, delta=1e-8)
        assert_almost_equal(lmp_forces[3, 2], 0.0378692, delta=1e-8)
        assert_almost_equal(lmp_forces[4, 2], 0.409472, delta=1e-8)

        stress = traj[100].get_stress(voigt=False)
        lmp_stress = np.dot(np.linalg.inv(rotation_ase2lammps), stress)
        lmp_stress = -np.dot(lmp_stress, rotation_ase2lammps) / GPa * 1e4

        # -153.11793 516.55652 -425.39416 -2961.0069 -1893.5721 -2612.0746

        pxx, pyy, pzz, pyz, pxz, pxy = lmp_stress[[0, 1, 2, 1, 0, 0],
                                                  [0, 1, 2, 2, 2, 1]]

        assert_almost_equal(pxx, -153.11793, delta=1e-5)
        assert_almost_equal(pyy, 516.55652, delta=1e-5)
        assert_almost_equal(pzz, -425.39416, delta=1e-5)
        assert_almost_equal(pxy, -2961.0069, delta=1e-5)
        assert_almost_equal(pxz, -1893.5721, delta=1e-5)
        assert_almost_equal(pyz, -2612.0746, delta=1e-5)

        assert_almost_equal(traj[-1].get_volume(), 58.167359, delta=1e-5)

    def tearDown(self) -> None:
        """
        The cleanup function.
        """
        # if exists(self.tmp_dir):
        #     shutil.rmtree(self.tmp_dir, ignore_errors=True)
        pass


if __name__ == "__main__":
    nose.run()
