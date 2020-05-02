
import unittest

from ase.gui.gui import GUI
from ase import Atoms
from ase.calculators.emt import EMT

class EnergyForcesTestCase(unittest.TestCase):

    def setUp(self):
        self.gui = GUI()
        self.atoms = Atoms('PtCu', positions=[[0, 0, 0], [2.5, 0, 0]],
                           cell=[5, 5, 5])
        self.gui.new_atoms(self.atoms)
        self.gui.simulation = {'calc': EMT}

    def tearDown(self):
        self.gui.exit()

    def test_EMT(self):
        win = self.gui.energy_window()
        win.run()
        self.assertTrue('4.39' in win.output.text[0].split('\n')[1])
        win.close()


if __name__ == '__main__':
    unittest.main()

