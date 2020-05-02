"""Module for performing energy minimization
"""

import ase.gui.ui as ui

from ase.gui.simulation import Simulation
import ase
import ase.optimize

from ase.gui.i18n import _


class Minimize(Simulation):
    """Window for performing energy minimization."""

    minimizers = ('BFGS', 'BFGSLineSearch', 'LBFGS', 'LBFGSLineSearch',
                  'MDMin', 'FIRE')

    def __init__(self, gui):
        super().__init__(gui)
        self.title = _('Energy minimization')
        self.win.add(ui.Label(
            _('Find configuration with minimal potential energy.')))
        self.win.add(ui.Label(''))

        self.packimageselection()
        self.win.add(ui.Label(''))

        self.make_minimize_gui()
        self.win.add(ui.Label(''))

        self.status_label = ui.Label('')
        self.win.add(self.status_label)

        self.makebutbox()

    def make_minimize_gui(self):
        self.algo = ui.ComboBox(labels=self.minimizers,
                                callback=self.min_algo_specific)
        self.win.add([ui.Label(_('Algorithm: ')), self.algo])
        lbl = ui.Label(_('Convergence criterion: Fmax < '))
        self.fmax_spin = ui.SpinBox(value=0.010, start=0.000,
                                    end=10.0, step=0.005)
        self.win.add([lbl, self.fmax_spin])
        self.steps_spin = ui.SpinBox(value=100, start=1, end=1e6, step=10)
        self.win.add([ui.Label(_('Max. number of steps: ')), self.steps_spin])

        # Special stuff for MDMin
        self.mdmin_widgets = [ui.Label(_('Pseudo time step: ')),
            ui.SpinBox(value=0.05, start=0.0, end=10.0, step=0.01)]
        self.win.add(self.mdmin_widgets)

        self.min_algo_specific()

    def min_algo_specific(self, *args):
        """ Show or hide MDMin-specific widgets """
        minimizer = self.minimizers[self.algo.value]
        for w in self.mdmin_widgets:
            w.active = minimizer == 'MDMin'

    def run(self, *args):
        """ run the minimization """
        if not self.setup_atoms():
            return
        fmax = self.fmax_spin.value
        steps = self.steps_spin.value
        mininame = self.minimizers[self.algo.value]
        self.begin(mode='min', algo=mininame, fmax=fmax, steps=steps)
        algo = getattr(ase.optimize, mininame)
        try:
            logger_func = self.gui.simulation['progress'].get_logger_stream
        except (KeyError, AttributeError):
            logger = '-'  # None
        else:
            logger = logger_func()  # Don't catch errors in the function.

        # Display status message
        self.status_label.text = _('Running while fmax > %.3f for %i max. steps') \
                                 % (fmax, steps)
        self.win.update()

        self.prepare_store_atoms()
        if mininame == 'MDMin':
            opt = algo(self.atoms, logfile=logger,
                       dt=self.mdmin_widgets[1].value)
        else:
            opt = algo(self.atoms, logfile=logger)
        opt.attach(self.step_performed)
        try:
           opt.run(fmax=fmax, steps=steps)
        # except AseGuiCancelException:
        # # Update display to reflect cancellation of simulation.
        #     self.status_label.set_text(_('Minimization CANCELLED after %i steps.')
        #                                % self.count_steps)
        except MemoryError:
            self.status_label.text = _('Out of memory! Consider using LBFGS method')
        else:
            self.status_label.text = _('Minimization completed in %i steps.') \
                                     % self.count_steps

        self.end()
        if self.count_steps:
            self.gui.notify_vulnerable()

        # Open movie window and energy graph
        # XXX disabled 2018-10-19.  --askhl
        #if self.gui.images.nimages > 1:
        #    self.gui.movie()
        #    assert not np.isnan(self.gui.images.E[0])
        #    if not self.gui.plot_graphs_newatoms():
        #        expr = 'i, e - E[-1]'
        #        self.gui.plot_graphs(expr=expr)

    def step_performed(self):
        ''' called after each optimization step'''
        forces = self.atoms.get_forces()
        fmax = (forces**2).sum(axis=1).max()
        self.status_label.text = _('Current maximal force: %.3f eV/Å') \
                                 % (fmax**0.5)
        self.win.update()

        self.store_atoms()

        self.gui.set_frame(len(self.gui.images) - 1)
