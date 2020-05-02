
import unittest

from ase.gui.gui import GUI
from ase import Atoms

class CalculatorTestCase(unittest.TestCase):

    def setUp(self):
        self.gui = GUI()
        self.atoms = Atoms('PtCu', positions=[[0, 0, 0], [2.5, 0, 0]],
                           cell=[5, 5, 5])
        self.gui.new_atoms(self.atoms)
        self.clc = self.gui.calculator_window()

    def tearDown(self):
        self.clc.close()
        self.gui.exit()

    def test_LJ(self):
        self.clc.radiobuttons.value = 1  # LJ
        self.clc.radio_button_selected()
        self.assertTrue(self.clc.button_setup.active)
        win = self.clc.button_setup_clicked()
        win.ok()
        try:
            import asap3
        except ImportError:
            raise unittest.SkipTest('Asap3 not available')
        self.assertTrue(self.clc.apply())

    def test_EMT(self):
        self.clc.radiobuttons.value = 2  # EMT
        self.clc.radio_button_selected()
        self.assertTrue(self.clc.button_setup.active)
        win = self.clc.button_setup_clicked()
        win.ok()
        try:
            import asap3
        except ImportError:
            raise unittest.SkipTest('Asap3 not available')
        self.assertTrue(self.clc.apply())

    def test_ASEEMT(self):
        self.clc.radiobuttons.value = 3  # ASEEMT
        self.clc.radio_button_selected()
        self.assertFalse(self.clc.button_setup.active)
        self.assertTrue(self.clc.apply())

    def test_EAM(self):
        self.clc.radiobuttons.value = 4  # EAM
        self.clc.radio_button_selected()
        self.assertTrue(self.clc.button_setup.active)
        win = self.clc.button_setup_clicked()
        win.close()   # check if no exception happen

    def test_Brenner(self):
        self.atoms.set_chemical_symbols(2*['C'])
        self.clc.gui.new_atoms(self.atoms)
        self.clc.radiobuttons.value = 5  # Brenner
        self.clc.radio_button_selected()
        self.assertFalse(self.clc.button_setup.active)
        try:
            import asap3
        except ImportError:
            raise unittest.SkipTest('Asap3 not available')
        self.assertTrue(self.clc.apply())

    def test_GPAW(self):
        self.clc.radiobuttons.value = 6  # GPAW
        self.clc.radio_button_selected()
        self.assertTrue(self.clc.button_setup.active)
        win = self.clc.button_setup_clicked()
        win.ok()
        try:
            import gpaw
        except ImportError:
            raise unittest.SkipTest('GPAW not available')
        self.assertTrue(self.clc.apply())
        win = self.clc.button_setup_clicked()
        win.xc.value = 'LDA'
        win.mode.widget.current(1)  # LCAO
        win.ok()
        self.assertTrue(self.clc.apply())
        win = self.clc.button_setup_clicked()
        win.mode.widget.current(2)  # PW
        win.ok()
        self.assertTrue(self.clc.apply())

if __name__ == '__main__':
    unittest.main()

