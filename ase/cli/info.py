# Note:
# Try to avoid module level import statements here to reduce
# import time during CLI execution


class CLICommand:
    """Print information about files or system.

    Without arguments, show information about ASE installation
    and library versions of dependencies.
    """

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--files', nargs='*', metavar='PATH',
                            help='Print information about specified files.')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Show more information about files.')
        parser.add_argument('--config', action='store_true',
                            help='List configured calculators')
        parser.add_argument('--formats', nargs='*', metavar='NAME', dest='io_formats',
                            help='List file formats: all known to ASE, or the ones which names contain given string(s)')
        parser.add_argument('--calculators', nargs='*', metavar='NAME',
                            help='List calculators and their configuration:'
                                 ' all known to ASE, or the ones which names contain given string(s)')
        parser.add_argument('--viewers', nargs='*', metavar='NAME',
                            help='List viewers: all the known to ASE, or the ones which names contain given string(s) ')
        parser.add_argument('--plugins', nargs='*', metavar='NAME',
                            help='List plugins: all installed, or the ones which names contain given string(s)')

    @staticmethod
    def run(args):
        print_info()
        if args.files:
             print_file_info(args)

        if args.config:
            print()
            from ase.config import cfg
            cfg.print_everything()

        for i in ('io_formats', 'calculators', 'viewers', 'plugins'):
            val = getattr(args, i)
            if val is None:
                continue
            print()
            print_plugables(i, val or None)


def print_file_info(args):
    from ase.io.formats import UnknownFileTypeError, filetype, ioformats
    from ase.io.bundletrajectory import print_bundletrajectory_info
    from ase.io.ulm import print_ulm_info
    n = max(len(filename) for filename in args.files) + 2
    nfiles_not_found = 0
    for filename in args.files:
        try:
            format = filetype(filename)
        except FileNotFoundError:
            format = '?'
            description = 'No such file'
            nfiles_not_found += 1
        except UnknownFileTypeError:
            format = '?'
            description = '?'
        else:
            if format in ioformats:
                description = ioformats[format].description
            else:
                description = '?'

        print('{:{}}{} ({})'.format(filename + ':', n,
                                    description, format))
        if args.verbose:
            if format == 'traj':
                print_ulm_info(filename)
            elif format == 'bundletrajectory':
                print_bundletrajectory_info(filename)

    raise SystemExit(nfiles_not_found)


def print_info():
    import platform
    import sys

    from ase.dependencies import all_dependencies

    versions = [('platform', platform.platform()),
                ('python-' + sys.version.split()[0], sys.executable)]

    for name, path in versions + all_dependencies():
        print(f'{name:24} {path}')


def print_plugables(i, only_given=None):
    import ase.plugins as plugins
    to_print = getattr(plugins, i)
    if only_given:
        only_given = [ i.lower() for i in only_given ]

        def filter(plugable):
            for i in plugable.lowercase_names:
                for j in only_given:
                    if j in i:
                        return True
            return False
    else:
        filter = None

    print(to_print.info(filter = filter))
