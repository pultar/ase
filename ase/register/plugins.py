""" This module contains the classes for listing plugins and
the instances (Calculators, IOs, etc...) provided by the plugins.


The structure is as follows


-----------                     --------------------------------
|         |                     |                              |
| Plugins |--------1:n----------|   Plugin (can provide some   |
|         |  (for each plugin   |   calculator, formats...)    |
|         |      package)       |                              |
^^^^^^^^^^                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                           |
   1:n  for each type                         1:n - plugin can return lists
    |   (i.e viewer, calculator)                | of instances
    |                                           | of each instance type
    |                                           | (calculator, viewer)
    |                                           | that it provides
    |                                           |
    |                                           |
    |                                           |
-------------------------                ------------------------
|                       |                |                      |
|    Instances:         |                |    Instance:         |
|      list all the     |------1:n-------|    holds inform.     |
|   available 'items'   |                |    about just one    |
|   (calcs, viewers)    |                |    calculator        |
|   of the given type   |                |    or viewer         |
|                       |                |                      |
^^^^^^^^^^^^^^^^^^^^^^^^^                ^^^^^^^^^^^^^^^^^^^^^^^^
                                                  |
                                                 1:1
                                                  |
                                         ------------------------
                                         |                      |
                                         |   Implementation     |
                                         |   (e.g. a Calculator |
                                         |    subclass)         |
                                         |                      |
                                         """"""""""""""""""""""""
"""

from contextlib import contextmanager
import importlib
import pkgutil
import warnings
from ase.utils import lazyproperty
from .listing import Listing
_current_plugin = None


@contextmanager
def within_the_plugin(plugin):
    global _current_plugin
    ocp = _current_plugin
    _current_plugin = plugin
    yield
    _current_plugin = ocp


def get_currently_registered_plugin():
    plugin = _current_plugin
    if not plugin:
        import ase.plugins as ase_plugins
        plugin = ase_plugins.plugins['external']
    return plugin


def import_module(name, path):
    try:
        module = importlib.import_module(name)
        return module
    except ImportError:
        raise
        warnings.warn(f"Can not import {name} in {path}."
                      " Probably broken ASE plugin.")


class Plugins(Listing):
    """ A class, that holds all the installed plugins in
        the given namespace package."""

    """ This information is just for initial creating of the instances """
    def __init__(self, namespace_package, instance_types):
        self.namespace_package = namespace_package
        self._instances = {
            k: cls(self, k)
            for k, cls in instance_types.items()
        }

    def packages(self):
        """ Return all the plugin packages, that are in the
        given namespace package (so that are in 'ase.plugins') """
        modules = []
        package = importlib.import_module(self.namespace_package)

        def mod_name(mod):
            return self.namespace_package + '.' + mod.name

        modules = (import_module(mod_name(mod), mod.module_finder.path)
                   for mod in pkgutil.iter_modules(package.__path__))

        modules = (i for i in modules if i)
        return modules

    @lazyproperty
    def plugins(self):
        return {p.__name__.rsplit('.', 1)[-1]: Plugin(self, p)
                for p in self.packages()}

    @property
    def items(self):
        return self.plugins.values()

    def instances_of(self, class_type):
        return self._instances[class_type]

    def all_instances(self):
        return self._instances.values()

    @lazyproperty
    def calculators(self):
        return self.instances_of('calculators')

    @lazyproperty
    def io_formats(self):
        return self.instances_of('io_formats')

    def __repr__(self):
        return f"<ASE plugins from: {self.namespace_package}>"

    def info(self, prefix='', opts={}):
        return "Plugins:\n"\
               "--------\n" + super().info(prefix, opts)

    def register(self):
        for i in self:
            i.register()


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, plugins, package):
        self.plugins = plugins
        self.package = package
        self.instances = {
            i.class_type: {} for i in plugins.all_instances()
        }
        self.modules = {}
        self._instances = {}
        self.registered = False

    @property
    def name(self):
        return self.package.__name__[12:]   # get rig 'ase.plugins'

    def register(self):
        if not self.registered:
            if hasattr(self.package, 'ase_register'):
                with within_the_plugin(self):
                    self.package.ase_register()
            self.registered = True

    def info(self, prefix='', opts={}):
        info = f'{prefix}{self.name}'

        prefix += '  '
        opts = opts.copy()
        opts['plugin'] = False

        for instances in self.plugins.all_instances():
            itype = instances.class_type
            inst = self.instances[itype]
            if inst:
                p = f'\n{prefix}{instances.singular_name}: '
                for i in inst.values():
                    info += i.info(p, opts)
        return info

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"
