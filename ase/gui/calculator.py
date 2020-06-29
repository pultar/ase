"""calculator.py - Module for choosing a calculator
"""

from ase.gui.i18n import _

import ase.gui.ui as ui
import numpy as np
from copy import copy, deepcopy

import ase
from ase.data import chemical_symbols

# Asap and GPAW may be imported if selected.

introtext = _("""\
To make most calculations on the atoms, a Calculator object must first
be associated with it.  ASE supports a number of calculators, supporting
different elements, and implementing different physical models for the
interatomic interactions.\
""")

# Informational text about the calculators
lj_info_txt = _("""\
The Lennard-Jones pair potential is one of the simplest
possible models for interatomic interactions, mostly
suitable for noble gasses and model systems.

Interactions are described by an interaction length and an
interaction strength.\
""")

emt_info_txt = _("""\
The EMT potential is a many-body potential, giving a
good description of the late transition metals crystalling
in the FCC crystal structure.  The elements described by the
main set of EMT parameters are Al, Ni, Cu, Pd, Ag, Pt, and
Au, the Al potential is however not suitable for materials
science application, as the stacking fault energy is wrong.

A number of parameter sets are provided.

<b>Default parameters:</b>

The default EMT parameters, as published in K. W. Jacobsen,
P. Stoltze and J. K. Nørskov, <i>Surf. Sci.</i> <b>366</b>, 394 (1996).

<b>Alternative Cu, Ag and Au:</b>

An alternative set of parameters for Cu, Ag and Au,
reoptimized to experimental data including the stacking
fault energies by Torben Rasmussen (partly unpublished).

<b>Ruthenium:</b>

Parameters for Ruthenium, as published in J. Gavnholt and
J. Schiøtz, <i>Phys. Rev. B</i> <b>77</b>, 035404 (2008).

<b>Metallic glasses:</b>

Parameters for MgCu and CuZr metallic glasses. MgCu
parameters are in N. P. Bailey, J. Schiøtz and
K. W. Jacobsen, <i>Phys. Rev. B</i> <b>69</b>, 144205 (2004).
CuZr in A. Paduraru, A. Kenoufi, N. P. Bailey and
J. Schiøtz, <i>Adv. Eng. Mater.</i> <b>9</b>, 505 (2007).
""")

aseemt_info_txt = _("""\
The EMT potential is a many-body potential, giving a
good description of the late transition metals crystalling
in the FCC crystal structure.  The elements described by the
main set of EMT parameters are Al, Ni, Cu, Pd, Ag, Pt, and
Au.  In addition, this implementation allows for the use of
H, N, O and C adatoms, although the description of these is
most likely not very good.

<b>This is the ASE implementation of EMT.</b> For large
simulations the ASAP implementation is more suitable; this
implementation is mainly to make EMT available when ASAP is
not installed.
""")

eam_info_txt = _("""\
The EAM/ADP potential is a many-body potential
implementation of the Embedded Atom Method and
equipotential plus the Angular Dependent Potential,
which is an extension of the EAM to include
directional bonds. EAM is suited for FCC metallic
bonding while the ADP is suited for metallic bonds
with some degree of directionality.

For EAM see M.S. Daw and M.I. Baskes,
Phys. Rev. Letters 50 (1983) 1285.

For ADP see Y. Mishin, M.J. Mehl, and
D.A. Papaconstantopoulos, Acta Materialia 53 2005
4029--4041.

Data for the potential is contained in a file in either LAMMPS Alloy
or ADP format which need to be loaded before use. The Interatomic
Potentials Repository Project at http://www.ctcms.nist.gov/potentials/
contains many suitable potential files.

For large simulations the LAMMPS calculator is more
suitable; this implementation is mainly to make EAM
available when LAMMPS is not installed or to develop
new EAM/ADP poentials by matching results using ab
initio.
""")

brenner_info_txt = _("""\
The Brenner potential is a reactive bond-order potential for
carbon and hydrocarbons.  As a bond-order potential, it takes
into account that carbon orbitals can hybridize in different
ways, and that carbon can form single, double and triple
bonds.  That the potential is reactive means that it can
handle gradual changes in the bond order as chemical bonds
are formed or broken.

The Brenner potential is implemented in Asap, based on a
C implentation published at http://www.rahul.net/pcm/brenner/ .

The potential is documented here:
  D W Brenner, O A Shenderova, J A Harrison, S J Stuart, B Ni
  and S B Sinnott: "A second-generation reactive empirical bond
  order (REBO) potential energy expression for hydrocarbons",
  J. Phys.: Condens. Matter 14 (2002) 783-802.
  doi: 10.1088/0953-8984/14/4/312
""")

gpaw_info_txt = _("""\
GPAW implements Density Functional Theory using a
<b>G</b>rid-based real-space representation of the wave
functions, and the <b>P</b>rojector <b>A</b>ugmented <b>W</b>ave
method for handling the core regions.
""")


class SetCalculator:
    ''' Window for selecting a calculator. '''

    # List the names of the radio button attributes
    captions = ('None', 'LJ', 'EMT', 'ASEEMT', 'EAM', 'Brenner', 'GPAW')
    # 'AIMS' and 'VASP' calculator windows were deleted in
    # commit 5c9024693666aa7ba9ecacb3b2206b6b5832e3b5
    # Restore from that commit if willing to reimplement
    unsetappable = (0, 3, 5)  # no setup for ase-emt and brenner
    longcaptions = (_('None'),
                    _('Lennard-Jones (ASAP)'),
                    _('EMT - Effective Medium Theory (ASAP)'),
                    _('EMT - Effective Medium Theory (ASE)'),
                    _('EAM - Embedded Atom Method (ASE)'),
                    _('Brenner Potential (ASAP)'),
                    _('Density Functional Theory (GPAW)'))
    # List of names for the parameter dictionaries
    paramdicts = ('lj_parameters',
                  'emt_parameters'
                  'eam_parameters',
                  'gpaw_parameters')
    # The name used to store parameters in the gui object
    classname = 'SetCalculator'

    def __init__(self, gui):
        self.win = ui.Window(title='Simulation')
        self.gui = gui
        self.win.add(ui.Text(introtext))
        self.win.add(ui.Label(_('Calculator:')))

        self.radiobuttons = ui.RadioButtons(
            self.captions, vertical=False,
            callback=self.radio_button_selected)
        self.win.add(self.radiobuttons)

        self.longcaption = ui.Label(text='')
        self.win.add(self.longcaption)

        self.button_setup = ui.Button(_('Setup'),
            callback=self.button_setup_clicked)
        self.button_info = ui.Button(_('Info'),
            callback=self.button_info_clicked)
        self.win.add([self.button_setup, self.button_info])

        self.radio_button_selected()

        self.win.add([  # pybutton(_('Python'), _('TODO')),
                 ui.Button(_('Cancel'), self.win.close),
                 ui.Button(_('Apply'), self.apply),
                 ui.Button(_('Ok'), self.ok)])

    def radio_button_selected(self, event=None):
        icalc = self.radiobuttons.value
        self.longcaption.text = self.longcaptions[icalc]
        self.button_setup.active = icalc not in self.unsetappable
        self.button_info.active = icalc > 0

    def button_setup_clicked(self, event=None):
        if not self.get_atoms():
            ui.error(_('There is no atoms!'))
            return
        icalc = self.radiobuttons.value
        scalc = self.captions[icalc].lower()
        win = None
        if scalc == 'lj':
            param = getattr(self, 'lj_parameters', None)
            win = LJ_Window(self, param, 'lj_parameters')
        elif scalc == 'emt':
            param = getattr(self, 'emt_parameters', None)
            win = EMT_Window(self, param, 'emt_parameters')
        elif scalc == 'aseemt':
            pass
        elif scalc == 'eam':
            eam_param = getattr(self, 'eam_parameters', None)
            win = EAM_Window(self, eam_param, 'eam_parameters')
        elif scalc == 'brenner':
            pass
        elif scalc == 'gpaw':
            gpaw_param = getattr(self, 'gpaw_parameters', None)
            win = GPAW_Window(self, gpaw_param, 'gpaw_parameters')
        return win

    def button_info_clicked(self, event=None):
        icalc = self.radiobuttons.value
        scalc = self.captions[icalc].lower()
        if scalc == 'lj':
            ui.helpwindow(lj_info_txt)
        elif scalc == 'emt':
            ui.helpwindow(emt_info_txt)
        elif scalc == 'aseemt':
            ui.helpwindow(aseemt_info_txt)
        elif scalc == 'eam':
            ui.helpwindow(eam_info_txt)
        elif scalc == 'brenner':
            ui.helpwindow(brenner_info_txt)
        elif scalc == 'gpaw':
            ui.helpwindow(gpaw_info_txt)

    def get_atoms(self):
        ''' Make an atoms object from the active frame '''
        images = self.gui.images
        frame = self.gui.frame
        try:
            atoms = images[frame]
        except IndexError:
            ui.error(_('No atoms present'))
            return False
        if len(atoms) < 1:
            return False
        self.atoms = deepcopy(atoms)
        if not images.get_dynamic(self.atoms).all():
            from ase.constraints import FixAtoms
            self.atoms.set_constraint(
                FixAtoms(mask=1 - images.get_dynamic(self.atoms)))
        return True

    def apply(self, *widget):
        icalc = self.radiobuttons.value
        scalc = self.captions[icalc].lower()
        if scalc == 'lj':
            if self.lj_check():
                self.choose_lj()
            else:
                return False
        elif scalc == 'emt':
            if self.emt_check():
                self.choose_emt()
            else:
                return False
        elif scalc == 'aseemt':
            if self.aseemt_check():
                self.choose_aseemt()
            else:
                return False
        elif scalc == 'eam':
            if self.eam_check():
                self.choose_eam()
                return True
            else:
                return False
        elif scalc == 'brenner':
            if self.brenner_check():
                self.choose_brenner()
                return True
            else:
                return False
        elif scalc == 'gpaw':
            if self.gpaw_check():
                self.choose_gpaw()
                return True
            else:
                return False
        else:
            print('Unknown calculator name')
            return False
        self.save_state()
        return True

    def ok(self, *widget):
        if self.apply():
            self.win.close()

    def save_state(self):
        state = {}
        state['icalc'] = self.radiobuttons.value
        for p in self.paramdicts:
            if hasattr(self, p):
                state[p] = getattr(self, p)
        self.gui.module_state[self.classname] = state

    def load_state(self):
        try:
            state = self.gui.module_state[self.classname]
        except KeyError:
            return
        self.radiobuttons.value = state['icalc']
        self.radio_button_selected()
        for p in self.paramdicts:
            if p in state:
                setattr(self, p, state[p])

    def lj_check(self):
        try:
            import asap3
        except ImportError:
            ui.error(_('ASAP is not installed. (Failed to import asap3)'))
            return False
        if not hasattr(self, 'lj_parameters'):
            ui.error(_('You must set up the Lennard-Jones parameters'))
            return False
        try:
            self.atoms.calc = asap3.LennardJones(**self.lj_parameters)
        except (asap3.AsapError, TypeError, ValueError) as e:
            ui.error(
                _('Could not attach Lennard-Jones calculator.\n\n%s') % str(e))
            return False
        return True

    def choose_lj(self):
        import asap3

        def lj_factory(p=self.lj_parameters, lj=asap3.LennardJones):
            return lj(**p)

        self.gui.simulation['calc'] = lj_factory

    def emt_get(self):
        import asap3
        provider = EMT_Window.emt_parameters[self.emt_parameters['index']][1]
        if provider is not None:
            provider = getattr(asap3, provider)
        return (asap3.EMT, provider, asap3)

    def emt_check(self):
        if not self.get_atoms():
            return False
        try:
            emt, provider, asap3 = self.emt_get()
        except ImportError:
            ui.error(
                _('ASAP is not installed. (Failed to import asap3)'))
            return False
        try:
            if provider is not None:
                self.atoms.calc = emt(provider())
            else:
                self.atoms.calc = emt()
        except (asap3.AsapError, TypeError, ValueError) as e:
            ui.error(
                _('Could not attach EMT calculator to the atoms.\n\n%s')
                % str(e))
            return False
        if provider is None:
            return self.element_check('EMT', ['Al', 'Cu', 'Ag', 'Au',
                                              'Ni', 'Pd', 'Pt'])
        elif provider == 'EMTRasmussenParameters':
            return self.element_check('EMT CuAg', ['Cu', 'Ag', 'Au'])
        elif provider == 'EMThcpParameters':
            return self.element_check('EMT Ru', ['Ru'])
        elif provider == 'EMTMetalGlassParameters':
            return (self.element_check('EMT CuMg', ['Cu', 'Mg']) or
                    self.element_check('EMT CuZr', ['Cu', 'Zr']))
        else:
            return False

    def choose_emt(self):
        emt, provider, asap3 = self.emt_get()
        if provider is None:
            emt_factory = emt
        else:

            def emt_factory(emt=emt, prov=provider):
                return emt(prov())

        self.gui.simulation['calc'] = emt_factory

    def aseemt_check(self):
        return self.element_check('ASE EMT', [
            'H', 'Al', 'Cu', 'Ag', 'Au', 'Ni', 'Pd', 'Pt', 'C', 'N', 'O'
        ])

    def choose_aseemt(self):
        import ase.calculators.emt
        self.gui.simulation['calc'] = ase.calculators.emt.EMT
        # In case Asap has been imported
        ase.calculators.emt.EMT.disabled = False

    def eam_check(self):
        from ase.calculators.eam import EAM
        if not hasattr(self, 'eam_parameters'):
            ui.error(_('EAM parameters not ready'))
            return False

        self.atoms.calc = EAM(**self.eam_parameters)
        return self.element_check('EAM', self.atoms.get_calculator().elements)

    def choose_eam(self):
        from ase.calculators.eam import EAM

        def eam_factory(p=self.eam_parameters):
            calc = EAM(**p)
            return calc

        self.gui.simulation['calc'] = eam_factory

    def brenner_check(self):
        try:
            import asap3
            asap3  # silence pyflakes
        except ImportError:
            ui.error(_('ASAP is not installed. (Failed to import asap3)'))
            return False
        return self.element_check('Brenner potential', ['H', 'C', 'Si'])

    def choose_brenner(self):
        import asap3
        self.gui.simulation['calc'] = asap3.BrennerPotential

    def gpaw_check(self):
        try:
            import gpaw
            gpaw  # silence pyflakes
        except ImportError:
            ui.error(_('GPAW is not installed. (Failed to import gpaw)'))
            return False
        if not hasattr(self, 'gpaw_parameters'):
            ui.error(_('GPAW parameters not specified'))
            return False
        return True

    def choose_gpaw(self):
        # This reuses the same GPAW object.
        import gpaw
        p = self.gpaw_parameters
        use = ['xc', 'kpts', 'mode']
        if p['mode'] == 'fd':
            if p['use_h'] == 0:
                use.append('h')
            else:
                use.append('gpts')
        elif p['mode'] == 'lcao':
            use.append('basis')
        elif p['mode'] == 'pw':
            use.append('pwcutoff')
        gpaw_param = {}
        for s in use:
            gpaw_param[s] = p[s]
        if p['use_mixer']:
            mx = getattr(gpaw, p['mixer'])
            mx_args = {}
            mx_arg_n = ['beta', 'nmaxold', 'weight']
            if p['mixer'] == 'MixerDif':
                mx_arg_n.extend(['beta_m', 'nmaxold_m', 'weight_m'])
            for s in mx_arg_n:
                mx_args[s] = p[s]
            gpaw_param['mixer'] = mx(**mx_args)
        # ~ progress = GpawProgressIndicator()
        # ~ self.gui.simulation['progress'] = progress
        gpaw_param['txt'] = '-'  # was: progress.get_gpaw_stream()
        if 'pwcutoff' in use:
            from gpaw import PW
            gpaw_param.pop('mode')
            gpaw_param.pop('pwcutoff')
            gpaw_calc = gpaw.GPAW(mode=PW(p['pwcutoff']), **gpaw_param)
        else:
            gpaw_calc = gpaw.GPAW(**gpaw_param)

        def gpaw_factory(calc=gpaw_calc):
            return calc

        self.gui.simulation['calc'] = gpaw_factory

    def element_check(self, name, elements):
        '''
        Check that all atoms are allowed

        name: calcultor label used for error report
        elements: list of allowed elements
        '''
        elements = [ase.data.atomic_numbers[s] for s in elements]
        elements_dict = {}
        for e in elements:
            elements_dict[e] = True
        if not self.get_atoms():
            ui.error(_('Can\'t extract information about atoms.'))
            return False
        try:
            for e in self.atoms.get_atomic_numbers():
                elements_dict[e]
        except KeyError:
            ui.error(
                _('Element %(sym)s not allowed by the "%(name)s" calculator') %
                dict(sym=ase.data.chemical_symbols[e], name=name))
            return False
        return True


class LJ_Window:
    ''' Window for configuring Lennard-Jones parameters '''

    def __init__(self, owner, param, attrname):
        self.win = ui.Window(title=_('Lennard-Jones parameters'))
        self.owner = owner
        self.attrname = attrname
        atoms = owner.atoms
        atnos = atoms.get_atomic_numbers()
        found = {}
        for z in atnos:
            found[z] = True
        self.present = list(found.keys())
        self.present.sort()  # Sorted list of atomic numbers
        nelem = len(self.present)
        self.win.add(ui.Label(_('Specify the Lennard-Jones parameters here')))
        self.win.add(ui.Label(''))

        self.win.add(ui.Label(_('Epsilon (eV):')))
        self.epsilon_adj = self.makematrix(self.present)
        self.win.add(ui.Label(''))

        self.win.add(ui.Label(_(u"Sigma (Å):")))
        self.sigma_adj = self.makematrix(self.present)
        self.win.add(ui.Label(''))

        # TRANSLATORS: Shift roughly means adjust (about a potential)
        self.modif = ui.CheckButton(_('Shift potential to zero at cutoff'))
        self.win.add(self.modif)
        self.modif.value = True
        self.win.add(ui.Label(''))

        self.win.add([ui.Button(_('Cancel'), self.win.close),
                  ui.Button(_('Ok'), self.ok)])

        # restore from param
        if param and param['elements'] == self.present:
            self.set_param(self.epsilon_adj, param['epsilon'], nelem)
            self.set_param(self.sigma_adj, param['sigma'], nelem)
            self.modif.value = param['modified']

    def makematrix(self, present):
        nelem = len(present)
        adjdict = {}

        s = 8 * ' '
        for i in range(nelem):  # header
            s += '  %s    ' % chemical_symbols[present[i]]
        self.win.add(ui.Label(s))
        for i in range(nelem):  # entries
            row = [ui.Label(' %s ' % chemical_symbols[present[i]])]
            for j in range(i + 1):
                adjdict[(i, j)] = ui.Entry(value='1.0', width=5)
                row.append(adjdict[(i, j)])
            self.win.add(row)
        return adjdict

    def set_param(self, adj, params, n):
        for i in range(n):
            for j in range(n):
                if j <= i:
                    adj[(i, j)].value = params[i, j]

    def get_param(self, adj, params, n):
        for i in range(n):
            for j in range(n):
                if j <= i:
                    params[i, j] = params[j, i] = adj[(i, j)].value

    def ok(self, *args):
        params = {}
        params['elements'] = copy(self.present)
        n = len(self.present)
        eps = np.zeros((n, n))
        self.get_param(self.epsilon_adj, eps, n)
        sigma = np.zeros((n, n))
        self.get_param(self.sigma_adj, sigma, n)
        params['epsilon'] = eps
        params['sigma'] = sigma
        params['modified'] = self.modif.value
        print(params)
        setattr(self.owner, self.attrname, params)
        self.win.close()


class EMT_Window:
    ''' Window for selection EMT(ASAP) parameters '''

    emt_parameters = (
        (_('Default (Al, Ni, Cu, Pd, Ag, Pt, Au)'), None),
        (_('Alternative Cu, Ag and Au'), 'EMTRasmussenParameters'),
        (_('Ruthenium'), 'EMThcpParameters'),
        (_('CuMg and CuZr metallic glass'), 'EMTMetalGlassParameters'))

    def __init__(self, owner, param, attrname):
        self.win = ui.Window(title=_('asap3.EMT parameters'))
        self.owner = owner
        self.attrname = attrname

        self.win.add(ui.Label(_('Select appropriate EMT paramseters set')))
        items = [item[0] for item in self.emt_parameters]
        self.emt_setup = ui.ComboBox(labels=items)
        self.win.add(self.emt_setup)

        if param:
            self.emt_setup.value = self.emt_parameters[param['index']][0]

        self.win.add([ui.Button(_('Cancel'), self.win.close),
                  ui.Button(_('Ok'), self.ok)])

    def ok(self):
        params = {}
        params['index'] = self.emt_setup.value
        print(params)
        setattr(self.owner, self.attrname, params)
        self.win.close()


class EAM_Window:
    ''' Window for selection EAM(ASAP) potential file '''

    def __init__(self, owner, param, attrname):
        self.win = ui.Window(title='EAM parameters')
        self.owner = owner
        self.attrname = attrname
        atoms = owner.atoms
        self.natoms = len(atoms)

        self.win.add(ui.Label(_('Select *.alloy or *.adp file')))
        self.win.add(ui.Label(''))
        self.filename_entry = ui.Entry()
        self.import_potential_but = ui.Button(_('Import Potential'),
                                              self.import_potential)
        self.win.add([self.filename_entry, self.import_potential_but])

        self.win.add([ui.Button(_('Cancel'), self.win.close),
                  ui.Button(_('Ok'), self.ok)])

        if param:  # sync with provided data
            self.eam_file = param['potential']
            self.filename_entry.value = self.eam_file

    def ok(self, *args):
        if not hasattr(self.owner, 'eam_parameters'):
            ui.error(_('You need to import the potential file'))

        self.win.close()

    def import_potential(self, *args):
        filename = 'Al99.eam.alloy'
        chooser = ui.LoadFileDialog(self.win)
        filename = chooser.go()
        if filename:
            param = {}
            param['potential'] = filename
            self.filename_entry.value = filename
            setattr(self.owner, self.attrname, param)


class GPAW_Window:
    ''' Window for configuring GPAW parameters '''

    gpaw_xc_list = ('LDA', 'PBE', 'RPBE', 'revPBE', 'GLLBSC')

    gpaw_modes = (_('FD - Finite Difference (grid) mode'),
                  _('LCAO - Linear Combination of Atomic Orbitals'),
                  _('PW - Plane Waves'))

    gpaw_basises = (_('sz - Single Zeta'),
                    _('szp - Single Zeta polarized'),
                    _('dzp - Double Zeta polarized'))

    gpaw_mixers = ('Mixer', 'MixerSum', 'MixerDif')

    def __init__(self, owner, param, attrname):
        self.win = ui.Window(title=_('GPAW parameters'))
        self.owner = owner
        self.attrname = attrname
        atoms = owner.atoms
        self.ucell = atoms.get_cell()
        self.size = tuple([self.ucell[i, i] for i in range(3)])
        self.pbc = atoms.get_pbc()
        self.orthogonal = self.ucell.orthorhombic
        self.natoms = len(atoms)

        # self.win.add(ui.Label(_('Configure GPAW parameters')))

        # Print some info
        txt = _('System of %i atoms with\n') % self.natoms
        if self.orthogonal:
            txt += _(
                'orthogonal unit cell: %.2f x %.2f x %.2f Å') % self.size
        else:
            txt += _('non-orthogonal unit cell:') + '\n'
            txt += str(self.ucell)
        self.win.add(ui.Label(txt))

        # XC potential
        self.xc = ui.ComboBox(labels=self.gpaw_xc_list, width=10)
        self.win.add(ui.Label(_('')))
        self.win.add([ui.Label(_('Exchange-correlation functional: ')),
                     self.xc])
        self.xc.value = self.gpaw_xc_list[1]  # PBE

        # Mode
        self.mode = ui.ComboBox(labels=self.gpaw_modes, width=40,
                                callback=self.mode_changed)
        self.win.add(ui.Label(_('')))
        self.win.add([ui.Label(_('Mode: ')), self.mode])

        # Grid spacing
        self.radio_h = ui.RadioButtons(
            labels=[_('Grid spacing'), _('Grid points')],
            vertical=False, callback=self.radio_h_toggled)
        self.h = ui.SpinBox(value=0.18, start=0.0, end=1.0, step=0.01,
                            callback=self.h_changed)
        self.win.add(self.radio_h)
        self.win.add([ui.Label(' h = '), self.h, ui.Label(' Å')])
        self.gpts = []
        for i in range(3):
            g = ui.SpinBox(value=4, start=4, end=1000, step=4,
                           callback=self.gpts_changed)
            self.gpts.append(g)
        self.gpts_hlabel = ui.Label('')
        self.gpts_hlabel_format = _('h_eff = (%.3f, %.3f, %.3f) Å')
        self.win.add([ui.Label(' gpts = ('), self.gpts[0],
                  ui.Label(', '), self.gpts[1], ui.Label(', '),
                  self.gpts[2], ui.Label(')  '), self.gpts_hlabel])

        # LCAO basis functions
        self.basis = ui.ComboBox(labels=self.gpaw_basises, width=30)
        self.win.add([ui.Label(_('Basis functions: ')), self.basis])
        self.basis.value = self.gpaw_basises[2]  # dzp

        # PW cutoff
        self.pwcutoff = ui.SpinBox(value=350, start=50, end=3500, step=50)
        self.win.add([ui.Label(_('Plane wave cutoff energy: ')),
                  self.pwcutoff, ui.Label(_('eV'))])

        # K-points
        self.kpts = []
        for i in range(3):
            if self.pbc[i] and self.orthogonal:
                default = np.ceil(20.0 / self.size[i])
            else:
                default = 1
            g = ui.SpinBox(value=default, start=1, end=100, step=1,
                           callback=self.k_changed)
            self.kpts.append(g)
        self.win.add(ui.Label(_('')))
        self.win.add([ui.Label(_('k-points grid k = (')), self.kpts[0],
                  ui.Label(', '), self.kpts[1], ui.Label(', '),
                  self.kpts[2], ui.Label(')')])
        for i in range(3):
            self.kpts[i].active = self.pbc[i]
        self.kpts_label = ui.Label('')
        self.kpts_label_format = _('k-points x size:  (%.1f, %.1f, %.1f) Å')
        self.win.add(self.kpts_label)

        # Spin polarization
        self.win.add(ui.Label(_('')))
        self.spinpol = ui.CheckButton(_('Spin polarized'),
                                      callback=self.mixer_changed)
        self.win.add(self.spinpol)

        # Mixer
        self.win.add(ui.Label(''))
        self.use_mixer = ui.CheckButton(_('Customize mixer parameters'),
                                        callback=self.mixer_changed)
        self.win.add(self.use_mixer)
        self.radio_mixer = ui.RadioButtons(labels=self.gpaw_mixers,
                                           vertical=False,
                                           callback=self.mixer_changed)
        self.win.add(self.radio_mixer)
        self.beta = ui.SpinBox(value=0.25, start=0.0, end=1.0, step=0.05)
        self.nmaxold = ui.SpinBox(value=3, start=1, end=10, step=1)
        self.weight = ui.SpinBox(value=50, start=1, end=500, step=1)
        self.win.add([ui.Label('  beta      = '), self.beta,
                  ui.Label('  nmaxold      = '), self.nmaxold,
                  ui.Label('  weight      = '), self.weight])
        self.beta_m = ui.SpinBox(value=0.70, start=0.0, end=1.0, step=0.05)
        self.nmaxold_m = ui.SpinBox(value=2, start=1, end=10, step=1)
        self.weight_m = ui.SpinBox(value=10, start=1, end=500, step=1)
        self.win.add([ui.Label('  beta_m = '), self.beta_m,
                  ui.Label('  nmaxold_m = '), self.nmaxold_m,
                  ui.Label('  weight_m = '), self.weight_m])

        # Eigensolver
        # Poisson-solver

        # Buttons at the bottom
        self.win.add(ui.Label(''))
        self.win.add([ui.Button(_('Cancel'), self.win.close),
                  ui.Button(_('Ok'), self.ok)])

        # restore from parameters
        if param:
            self.xc.value = param['xc']
            self.radio_h.value = int(not param['use_h'])
            for i in range(3):
                self.gpts[i].value = param['gpts'][i]
                self.kpts[i].value = param['kpts'][i]
            self.spinpol.value = param['spinpol']
            if param['mode'] == 'fd':
                self.mode.value = self.gpaw_modes[0]
            elif param['mode'] == 'lcao':
                self.mode.value = self.gpaw_modes[1]
            elif param['mode'] == 'pw':
                self.mode.value = self.gpaw_modes[2]
            else:
                print('ERROR: Incorrect mode "%s"!' % param['mode'])
            self.basis.value = param['basis']
            self.pwcutoff.value = param['pwcutoff']
            self.use_mixer.value = param['use_mixer']
            self.radio_mixer.value = param['mixer#']
            for t in ('beta', 'nmaxold', 'weight',
                      'beta_m', 'nmaxold_m', 'weight_m'):
                getattr(self, t).value = param[t]

        # finalize initialization
        self.mode_changed()
        self.k_changed()
        self.mixer_changed()

    def mode_changed(self, widget=None):
        if self.mode.value == 0:  # FD
            self.radio_h.active = True
            self.radio_h_toggled()
            self.basis.active = False
            self.pwcutoff.active = False
        elif self.mode.value == 1:  # LCAO
            self.radio_h.active = False
            self.h.active = False
            for g in self.gpts:
                g.active = False
            self.basis.active = True
            self.pwcutoff.active = False
        elif self.mode.value == 2:  # PW
            self.radio_h.active = False
            self.h.active = False
            for g in self.gpts:
                g.active = False
            self.basis.active = False
            self.pwcutoff.active = True

    def radio_h_toggled(self, widget=None):
        if self.radio_h.value == 0:  # h
            self.h.active = True
            for g in self.gpts:
                g.active = False
            self.h_changed()
        else:  # grid
            self.h.active = False
            for g in self.gpts:
                g.active = True
            self.gpts_changed()

    def gpts_changed(self, *args):
        if self.radio_h.value == 1:  # gpts
            g = np.array([int(g.value) for g in self.gpts])
            size = np.array([self.ucell[i, i] for i in range(3)])
            txt = self.gpts_hlabel_format % tuple(size / g)
            self.gpts_hlabel.text = txt
        else:  # h
            self.gpts_hlabel.text = ''

    def h_changed(self, *args):
        h = self.h.value
        for i in range(3):
            g = 4 * round(self.ucell[i, i] / (4 * h))
            self.gpts[i].active = True
            self.gpts[i].value = int(g)
            self.gpts[i].active = False

    def k_changed(self, *args):
        size = []
        for i in range(3):
            size.append(self.kpts[i].value *
                        np.sqrt(np.vdot(self.ucell[i], self.ucell[i])))
        self.kpts_label.text = self.kpts_label_format % tuple(size)

    def mixer_changed(self, *args):
        self.radio_mixer.active = self.use_mixer.value
        if self.use_mixer.value:
            self.beta.active = True
            self.nmaxold.active = True
            self.weight.active = True
            if self.spinpol.value:
                if self.radio_mixer.value > 1:
                    self.beta_m.active = True
                    self.nmaxold_m.active = True
                    self.weight_m.active = True
                else:
                    self.beta_m.active = False
                    self.nmaxold_m.active = False
                    self.weight_m.active = False
        else:
            self.beta.active = False
            self.nmaxold.active = False
            self.weight.active = False
            self.beta_m.active = False
            self.nmaxold_m.active = False
            self.weight_m.active = False

    def ok(self, *args):
        param = {}
        param['xc'] = self.gpaw_xc_list[self.xc.value]
        param['mode'] = self.gpaw_modes[self.mode.value].split()[0].lower()
        param['use_h'] = not bool(self.radio_h.value)
        param['h'] = self.h.value
        param['gpts'] = [int(g.value) for g in self.gpts]
        param['basis'] = self.gpaw_basises[self.basis.value].split()[0].lower()
        param['pwcutoff'] = self.pwcutoff.value
        param['kpts'] = [int(k.value) for k in self.kpts]
        param['spinpol'] = self.spinpol.value
        param['use_mixer'] = self.use_mixer.value
        param['mixer'] = self.gpaw_mixers[self.radio_mixer.value]
        param['mixer#'] = self.radio_mixer.value
        for t in ('beta', 'nmaxold', 'weight',
                  'beta_m', 'nmaxold_m', 'weight_m'):
            param[t] = getattr(self, t).value
        setattr(self.owner, self.attrname, param)
        print(param)
        self.win.close()
