"""
Gaussian calculator for ASE written by:

    Glen R. Jenness
    University of Wisconsin - Madison

Based off of code written by:

    Glen R. Jenness
    Kuang Yu
    Torsten Kerber, Ecole normale superieure de Lyon (*)
    Paul Fleurat-Lessard, Ecole normale superieure de Lyon (*)
    Martin Krupicka

(*) This work is supported by Award No. UK-C0017, made by King Abdullah
University of Science and Technology (KAUST), Saudi Arabia.

See accompanying license files for details.
"""
import os
from shutil import which

from ase.calculators.calculator import EnvironmentError, FileIOCalculator, Parameters, ReadError

"""
Gaussian has two generic classes of keywords:  link0 and route.
Since both types of keywords have different input styles, we will
distinguish between both types, dividing each type into str's, int's
etc.

For more information on the Link0 commands see:
    http://www.gaussian.com/g_tech/g_ur/k_link0.htm
For more information on the route section keywords, see:
    http://www.gaussian.com/g_tech/g_ur/l_keywords09.htm
"""
link0_keys = ['chk',
              'mem',
              'rwf',
              'int',
              'd2e',
              'lindaworkers',
              'kjob',
              'subst',
              'save',
              'nosave',
              'nprocshared',
              'nproc']

# This one is a little strange.  Gaussian has several keywords where you just
# specify the keyword, but the keyword itself has several options.
# Ex:  Opt, Opt=QST2, Opt=Conical, etc.
# These keywords are given here.
route_self_keys = ['opt',
                   'irc',
                   'force',
                   'freq',
                   'complex',
                   'fmm',
                   'genchk',
                   'polar',
                   'prop',
                   'pseudo',
                   'restart',
                   'scan',
                   'scrf',
                   'sp',
                   'sparse',
                   'stable',
                   'population',
                   'volume',
                   'densityfit',
                   'nodensityfit']

route_keys = [# int keys
              # Multiplicity and charge are not really route keywords,
              # but we will put them here anyways
              'cachesize',
              'cbsextrapolate',
              'constants',
              # str keys
              'functional',
              'maxdisk',
              'cphf',
              'density',
              'ept',
              'field',
              'geom',
              'guess',
              'gvb',
              'integral',
              'ircmax',
              'name',
              'nmr',
              'oniom',
              'output',
              'punch',
              'scf',
              'symmetry',
              'td',
              'units',
              # Float keys
              'pressure',
              'scale',
              'temperature']


class GaussianOptimizer:
    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc

    def todict(self):
        return {'type': 'optimization',
                'optimizer': 'GaussianOptimizer'}

    def run(self, fmax=None, steps=None, **gaussian_kwargs):
        if fmax is not None:
            if not isinstance(fmax, str):
                raise ValueError('fmax has to be a string if the internal optimizer of Gaussian is called via ASE.')

        opt = gaussian_kwargs.pop('opt', '')

        if fmax is not None:
            opt = '{}, {}'.format(opt, fmax)
        if steps is not None:
            opt = '{}, maxcycles={}'.format(opt, steps)

        force = self.calc.parameters.pop('force', None)
        irc = self.calc.parameters.pop('irc', None)
        self.calc.parameters['opt'] = opt
        self.atoms.calc = self.calc
        self.atoms.get_potential_energy()
        self.atoms.cell = self.calc.atoms.cell
        self.atoms.positions = self.calc.atoms.positions.copy()
        if force is not None:
            self.atoms.calc['force'] = force
        if irc is not None:
            self.atoms.calc['irc'] = irc


class GaussianIRC:
    def __init__(self, atoms, calc):
        self.atoms = atoms
        self.calc = calc

    def todict(self):
        return {'type': 'irc',
                'optimizer': 'GaussianIRC'}

    def run(self, direction=None, steps=None, **gaussian_kwargs):

        irc = gaussian_kwargs.pop('irc', '')

        if direction is not None:
            irc = '{}, {}'.format(irc, direction)
        if steps is not None:
            irc = '{}, maxpoints={}'.format(irc, steps)

        opt = self.calc.parameters.pop('opt', None)
        force = self.calc.parameters.pop('force', None)
        freq = self.calc.parameters.pop('freq', None)
        self.calc.parameters['irc'] = irc
        self.atoms.calc = self.calc
        self.atoms.get_potential_energy()
        self.atoms.cell = self.calc.get_atoms().cell
        self.atoms.positions = self.calc.get_atoms().copy()
        if force is not None:
            self.atoms.calc['force'] = force
        if opt is not None:
            self.atoms.calc['opt'] = opt
        if freq is not None:
            self.atoms.calc['freq'] = freq


class Gaussian(FileIOCalculator):
    """
    Gaussian calculator
    """
    name = 'Gaussian'

    implemented_properties = ['energy', 'forces', 'dipole', 'freq']

    command = 'GAUSSIAN < PREFIX.com > PREFIX.log'

    default_parameters = {'charge': 0,
                          'method': 'hf',
                          'basis': '6-31g*'}


    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, optimized=None, scratch=None, ioplist=list(),
                 basisfile=None, extra=None, addsec=None, **kwargs):

        """Constructs a Gaussian-calculator object.

        extra: any extra text to be included in the input card
        addsec: a list of strings to be included as "additional sections"

        """

        gaussians = ('g16', 'g09', 'g03')
        for gau in gaussians:
            if which(gau):
                self.command = self.command.replace('GAUSSIAN', gau)
                label = label or gau
                break
        else:
            raise EnvironmentError('missing Gaussian executable {}'.format(gaussians))

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)

        if restart is not None:
            try:
                self.read(restart)
            except ReadError:
                if ignore_bad_restart_file:
                    self.reset()
                else:
                    raise

        self.optimized  = optimized
        self.ioplist = ioplist
        self.scratch = scratch
        self.basisfile = basisfile

        # store extra parameters
        self.extra = extra
        self.addsec = addsec

    def set(self, **kwargs):
        changed_parameters = FileIOCalculator.set(self, **kwargs)
        if changed_parameters:
            self.reset()
        return changed_parameters

    def check_state(self, atoms):
        system_changes = FileIOCalculator.check_state(self, atoms)

        ignore = ['cell', 'pbc']
        for change in system_changes:
            if change in ignore:
                system_changes.remove(change)

        return system_changes

    def write_input(self, atoms, properties=None, system_changes=None):
        """Writes the input file"""
        FileIOCalculator.write_input(self, atoms, properties, system_changes)

        magmoms = atoms.get_initial_magnetic_moments().tolist()
        self.parameters.initial_magmoms = magmoms
        self.parameters.write(self.label + '.ase')

        # Set default behavior
        if ('multiplicity' not in self.parameters):
            tot_magmom = atoms.get_initial_magnetic_moments().sum()
            mult = tot_magmom + 1
        else:
            mult = self.parameters['multiplicity']

        filename = self.label + '.com'
        inputfile = open(filename, 'w')

        link0 = str()
        if self.parameters['basis'] != '':
            route = '#p %s/%s' % (self.parameters['method'],
                                  self.parameters['basis'])
        else:
            route = '#p %s' % (self.parameters['method'])

        for key, val in self.parameters.items():
            if key.lower() in link0_keys:
                link0 += ('%%%s=%s\n' % (key, val))
            elif key.lower() in route_self_keys:
                if (val.lower() == key.lower()):
                    route += (' ' + val)
                else:
                    if ',' in val:
                        route += ' %s(%s)' % (key, val)
                    else:
                        route += ' %s=%s' % (key, val)

            elif key.lower() in route_keys:
                route += ' %s=%s' % (key, val)

        if 'forces' in properties and 'force' not in self.parameters:
            route += ' force'
        # include any other keyword(s)
        if self.extra is not None:
            route += ' ' + self.extra

        if self.ioplist:
            route += ' IOp('
            route += ', '.join(self.ioplist)
            route += ')'

        inputfile.write(link0)
        inputfile.write(route)
        inputfile.write(' \n\n')
        inputfile.write('Gaussian input prepared by ASE\n\n')
        inputfile.write('%i %i\n' % (self.parameters['charge'],
                                     mult))

        symbols = atoms.get_chemical_symbols()
        coordinates = atoms.get_positions()
        for i in range(len(atoms)):
            inputfile.write('%-10s' % symbols[i])
            for j in range(3):
                inputfile.write('%20.10f' % coordinates[i, j])
            inputfile.write('\n')

        inputfile.write('\n')

        if 'opt' in self.parameters:
            if 'modredun' in [par.lower() for par in self.parameters['opt'].split(',')]:
                if 'release' in self.parameters: #coordinates that first need to be unfrozen
                    for r in self.parameters['release']:
                        inputfile.write('%s A\n'%' '.join(map(str,r)))
                if 'fix' in self.parameters: #coordinates that need to be frozen
                    for fi in self.parameters['fix']:
                        inputfile.write('%s F\n'%' '.join(map(str,fi)))
                if 'change' in self.parameters: #coordinates that first need to be updated and then frozen
                    for c in self.parameters['change']:
                        inputfile.write('%s F\n'%' '.join(map(str,c)))
                if 'relaxed_scan' in self.parameters: #coordinates that first need to be scanned
                    for s in self.parameters['relaxed_scan']:
                        inputfile.write('%s S %i %.2f\n'%(' '.join(map(str,s[:-2])),s[-2],s[-1]))

        if 'gen' in self.parameters['basis'].lower():
            if self.basisfile is None:
                raise RuntimeError('Please set basisfile.')
            elif not os.path.isfile(self.basisfile.rstrip('/N').lstrip('@')):
                error = 'Basis file %s does not exist.' % self.basisfile
                raise RuntimeError(error)
            elif self.basisfile[0] == '@':
                inputfile.write(self.basisfile + '\n\n')
            else:
                f2 = open(self.basisfile, 'r')
                inputfile.write(f2.read())
                f2.close()

        if atoms.get_pbc().any():
            cell = atoms.get_cell()
            line = str()
            for v in cell:
                line += 'TV %20.10f%20.10f%20.10f\n' % (v[0], v[1], v[2])
            inputfile.write(line)

        # include optional additional sections
        if self.addsec is not None:
            inputfile.write('\n\n'.join(self.addsec))

        inputfile.write('\n\n')

        inputfile.close()

    def read(self, label):
        """Used to read the results of a previous calculation if restarting"""
        FileIOCalculator.read(self, label)

        from ase.io.gaussian import read_gaussian_out
        filename = self.label + '.log'

        if not os.path.isfile(filename):
            raise ReadError

        self.atoms = read_gaussian_out(filename, quantity='atoms')
        self.parameters = Parameters.read(self.label + '.ase')
        initial_magmoms = self.parameters.pop('initial_magmoms')
        self.atoms.set_initial_magnetic_moments(initial_magmoms)
        self.read_results()

    def read_results(self):
        """Reads the output file using GaussianReader"""
        from ase.io.gaussian import read_gaussian_out

        filename = self.label + '.log'

        quantities = ['energy', 'forces', 'dipole']
        with open(filename, 'r') as fileobj:
            for quant in quantities:
                self.results[quant] = read_gaussian_out(fileobj,
                                                        quantity=quant)

            self.results['magmom'] = read_gaussian_out(fileobj,
                                                       quantity='multiplicity')
            self.results['magmom'] -= 1

    def clean(self):
        """Cleans up from a previous run"""
        extensions = ['.chk', '.com', '.log']

        for ext in extensions:
            f = self.label + ext
            try:
                if (self.directory is not None):
                    os.remove(os.path.join(self.directory, f))
                else:
                    os.remove(f)
            except OSError:
                pass

    def get_version(self):
        return self.read_output(self.label + '.log', 'version')

    def get_opt_state(self):
        return self.optimized
