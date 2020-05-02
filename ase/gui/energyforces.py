"""Module for calculating energies and forces.
"""

from ase.gui.i18n import _
import ase.gui.ui as ui
from ase.gui.simulation import Simulation

class EnergyForces(Simulation):
    def __init__(self, gui):
        super().__init__(gui)
        self.title = _('Potential energy and forces')
        self.add(ui.Label(_('Calculate potential energy and forces on atoms')))
        self.add(ui.Label(''))
        self.packimageselection()
        self.add(ui.Label(''))
        self.forces = ui.CheckButton(_('Calculate forces'))
        self.add(self.forces)
        self.add(ui.Label(''))
        self.output = ui.Text(9*'\n')
        self.savebutton = ui.Button('Save', self.saveoutput)
        self.add(ui.Label(_('Output:')))
        self.add([self.output, self.savebutton])
        self.savebutton.active = False
        self.makebutbox()

    def saveoutput(self):
        chooser = ui.SaveFileDialog(self.win)
        filename = chooser.go()
        if filename:
            with open(filename, 'w') as f:
                for line in self.output.text:
                    f.write(line)

    def run(self, *args):
        if not self.setup_atoms():
            return
        self.begin()
        e = self.atoms.get_potential_energy()
        txt = _('Potential Energy:\n')
        txt += _('  %8.3f eV\n') % (e,)
        txt += _('  %8.4f eV/atom\n\n') % (e / len(self.atoms),)
        if self.forces.value:
            txt += _('Forces:\n')
            forces = self.atoms.get_forces()
            for f in forces:
                txt += '  %8.3f, %8.3f, %8.3f eV/Å\n' % tuple(f)
        self.output.text = txt
        self.savebutton.active = True
        self.end()
