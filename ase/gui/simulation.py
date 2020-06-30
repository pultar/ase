"""Base class for simulation windows
"""
import ase.gui.ui as ui
from ase.gui.i18n import _


class Simulation:
    """  Base calss for EnergyForces and Minimize windows """

    def __init__(self, gui):
        self.win = ui.Window(title='Simulation')
        self.gui = gui
        self.gui.register_vulnerable(self)

    def close(self):
        self.gui.unregister_vulnerable(self)
        self.win.close()

    def packimageselection(self):
        """ Elements for selection of starting configuration """
        self.win.add(ui.Label(_('Select starting configuration')))
        self.numconfig_format = _('of %i frames:')
        self.numconfig_label = ui.Label('')
        self.win.add(self.numconfig_label)

        self.scale = ui.Scale(value=self.gui.frame, start=1,
                              end=len(self.gui.images), callback=None)
        self.win.add(self.scale)

        self.firstcurrlast = [ui.Button(_('First'), self.first_click),
                              ui.Button(_('Current'), self.current_click),
                              ui.Button(_('Last'), self.last_click)]
        self.win.add(self.firstcurrlast)
        self.setupimageselection()

    def setupimageselection(self):
        n = len(self.gui.images)
        if self.scale.value > n:
            self.scale.value = n
        self.scale.set_end(n)
        self.numconfig_label.text = self.numconfig_format % n

    def notify_atoms_changed(self):
        self.setupimageselection()  # update number of frames/images

    def first_click(self):
        self.scale.value = 1

    def current_click(self):
        self.scale.value = self.gui.frame + 1

    def last_click(self):
        self.scale.value = len(self.gui.images)

    def getimagenumber(self):
        """Get the image number selected in the start image frame."""
        return self.scale.value - 1

    def makebutbox(self, helptext=None):
        """ Add buttons Run/Close/Help"""
        self.runbut = ui.Button(_('Run'), self.run)
        self.closebut = ui.Button('Close', self.win.close)
        if helptext:
            self.helpbut = ui.helpbutton(helptext)
            self.win.add([self.runbut, self.closebut, self.helpbut])
        else:
            self.win.add([self.runbut, self.closebut])

    def setup_atoms(self):
        self.atoms = self.get_atoms()
        if self.atoms is None:
            return False
        try:
            self.calculator = self.gui.simulation['calc']
        except (KeyError, TypeError):
            ui.error(_('No calculator: Use Set Calculator first.'))
            return False
        self.atoms.calc = self.calculator()
        return True

    def get_atoms(self):
        """Make an atoms object from the active image"""
        images = self.gui.images
        atoms = images[self.getimagenumber()]
        natoms = len(atoms) // images.repeat.prod()
        if natoms < 1:
            ui.error(_('No atoms present'))
            return None
        return atoms[:natoms]

    def begin(self, **kwargs):
        if 'progress' in self.gui.simulation:
            self.gui.simulation['progress'].begin(**kwargs)

    def end(self):
        if 'progress' in self.gui.simulation:
            self.gui.simulation['progress'].end()

    def prepare_store_atoms(self):
        """Informs the gui that the next configuration should be the first."""
        iframe = self.getimagenumber()
        if iframe + 1 < len(self.gui.images):
            self.gui.images.delete(slice(iframe + 1, len(self.gui.images)))
        self.count_steps = 0

    def store_atoms(self):
        """Observes the minimization and stores the atoms in the gui."""
        self.gui.images.append(self.atoms)
        self.count_steps += 1

    def run(self):
        pass  # implement in childs
