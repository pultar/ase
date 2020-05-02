
import unittest

from ase.gui.gui import GUI
from ase import Atoms
from ase.calculators.emt import EMT

class MinimizeTestCase(unittest.TestCase):

    def setUp(self):
        self.gui = GUI()
        self.atoms = Atoms('PtCu', positions=[[0, 0, 0], [2.5, 0, 0]],
                           cell=[5, 5, 5])
        self.gui.new_atoms(self.atoms)
        self.gui.simulation = {'calc': EMT}
        self.mmz = self.gui.energy_minimize_window()

    def tearDown(self):
        self.mmz.close()
        self.gui.exit()

    def test_MDMin(self):
        self.mmz.algo.value = 'MDMin'
        self.mmz.min_algo_specific()
        self.mmz.run()
        self.mmz.last_click()
        self.assertTrue(self.mmz.scale.value > 1)
        self.mmz.current_click()
        self.assertTrue(self.mmz.scale.value == self.gui.frame + 1)
        self.mmz.first_click()
        self.assertTrue(self.mmz.scale.value == 1)

    def test_BFGS(self):
        self.mmz.run()
        self.mmz.last_click()
        self.assertTrue(self.mmz.scale.value == len(self.gui.images))
        self.mmz.close()

if __name__ == '__main__':
    unittest.main()

