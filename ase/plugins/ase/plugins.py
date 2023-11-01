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

import importlib
import pkgutil
import warnings
from ase.utils import lazyproperty
from collections.abc import Mapping
from typing import Dict


def import_module(name, path):
    try:
        module = importlib.import_module(name)
        return module
    except ImportError:
        warnings.warn(f"Can not import {name} in {path}."
                      " Probably broken ASE plugin.")


class Listing(Mapping):

    def info(self, prefix: str = '', opts: Dict = {}) -> str:
        """
        Parameters
        ----------
        prefix
            Prefix, which should be prepended before each line.
            E.g. indentation.
        opts
            Dictionary, that can holds options,
            what info to print and which not.

        Returns
        -------
        info
          Information about the object and (if applicable) contained items.
        """

        out = [i.info(prefix) for i in self.sorted()]
        return '  \n'.join(out)

    @staticmethod
    def sorting_key(i):
        return i.name.lower()

    def sorted(self):
        ins = self.items
        ins = ins.copy() if isinstance(self.items, list) else list(ins)
        ins.sort(key=self.sorting_key)
        return ins

    def __len__(self):
        return len(self.items)

    def __getitem__(self, name):
        out = self.find_by_name(name)
        if not out:
            raise KeyError(f"There is no {name} in {self}")
        return out

    def __iter__(self):
        return iter(self.items)


class Instance:
    """
    A class, that holds information about an implementation of a calculator,
    viewer, IO, whatsoever. """

    def __init__(self, plugin, class_type, name, cls):
        self.plugin = plugin
        self.class_type = class_type
        self.name = name
        self.cls = cls

    @property
    def implementation(self):
        return self()

    def __call__(self):
        return self.cls

    def __repr__(self):
        return f"<ASE {self.class_type}: {self.name} provided by {self.cls}>"

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError()
        return getattr(self.cls, name)

    @lazyproperty
    def names(self):
        name = getattr(self.cls, 'name', None)
        if isinstance(name, str):
            name = [name]
        if isinstance(name, (list, set, tuple)):
            if self.name not in name:
                name.append(self.name)
        else:
            name = [self.name]

        if self.cls.__name__ not in name:
            name.append(self.cls.__name__)
        return name

    @lazyproperty
    def lowercase_names(self):
        return {i.lower() for i in self.names}

    def info(self, prefix='', opts={}):
        out = f"{prefix}{self.name}"
        if opts.get('plugin', True):
            out += f"    (from plugin {self.plugin.name})"
        return out


class Instances(Listing):
    """ This class holds all the Instances (Calculators, Viewers, ...)
    of one type, that can be used by user """

    child_class = Instance

    def __init__(self, plugin_list: 'Plugins', class_type: str):
        self.plugins = plugin_list
        self.class_type = class_type

    @property
    def singular_name(self):
        return self.class_type[:-1]

    @lazyproperty
    def instances(self):
        """ List the instances (e.g. calculators) from all plugins. """
        return list(self._instances())

    def _instances(self):
        for p in self.plugins:
            yield from p.instances_of(self.class_type)

    @property
    def items(self):
        return self.instances

    @staticmethod
    def instance_has_attribute(obj, attribute, value):
        v = getattr(obj, attribute, None)
        if value == v:
            return True
        if isinstance(v, (list, set)):
            for i in v:
                if i == value:
                    return True
        return False

    def find_by(self, attribute, value):
        """ Find plugin according the given attribute.
        The attribute can be given by list of alternative values,
        or not at all - in this case, the default value for the attribute
        will be used """
        for i in self:
            if Instances.instance_has_attribute(i, attribute, value):
                return i

    def find_all_by(self, attribute, value):
        """ Find plugin according the given attribute.
        The attribute can be given by list of alternative values,
        or not at all - in this case, the default value for the attribute
        will be used """
        return (i for i in self.plugins if
                Instances.instance_has_attribute(i, attribute, value))

    def __repr__(self):
        return f"<ASE list of {self.class_type}>"

    def find_by_name(self, name):
        out = self.find_by('names', name.lower())
        if not out:
            out = self.find_by('lowercase_names', name.lower())
        return out

    def __getitem__(self, name):
        return super.__getitem__(name)()


class CalculatorInstance(Instance):

    def __init__(self, plugin, class_type, name, cls):
        super().__init__(plugin, class_type, name, cls)
        if not hasattr('cls', 'ase_calculator_name'):
            cls.ase_calculator_name = name

    @lazyproperty
    def availability(self):
        return self.cls.availability_information()


class CalculatorInstances(Instances):
    """ Just a few specialities for instances of calculators """

    child_class = CalculatorInstance

    def info(self, prefix='', opts={}):
        return f"{prefix}Calculators:\n" \
               f"{prefix}------------\n" + super().info(prefix + '  ', opts)


class Plugins(Listing):
    """ A class, that holds all the installed plugins in
        the given namespace package."""

    """ This information is just for initial creating of the instances """
    _instance_types = {
        'calculators': CalculatorInstances
    }

    def __init__(self, namespace_package):
        self.namespace_package = namespace_package
        self._instances = {}

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
        if class_type not in self._instances:
            it = self._instance_types.get(class_type, Instances)
            self._instances[class_type] = it(self, class_type)
        return self._instances[class_type]

    def instances(self):
        return [self.instances_of(i) for i in self._instance_types]

    @lazyproperty
    def calculators(self):
        return self.instances_of('calculators')

    def __repr__(self):
        return f"<ASE plugins from: {self.namespace_package}>"

    def info(self, prefix='', opts={}):
        return "Plugins:\n"\
               "--------\n" + super().info(prefix, opts)


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, plugins, package):
        self.plugins = plugins
        self.package = package
        self.modules = {}
        self._instances = {}

    @property
    def name(self):
        return self.package.__name__[12:]   # get rig 'ase.plugins'

    def get_module(self, class_type):
        """ Return a module for a given type, e.g. calculator, io, viewer... """
        if class_type not in self.modules:
            name = self.package.__name__
            mod = importlib.util.find_spec(name + '.' + class_type,
                                           self.package.__path__)
            if not mod:
                self.modules[class_type] = None
            else:
                module = importlib.util.module_from_spec(mod)
                mod.loader.exec_module(module)
                self.modules[class_type] = module

        return self.modules[class_type]

    def _instances_of(self, class_type):
        module = self.get_module(class_type)
        instance_type = self.plugins.instances_of(class_type).child_class
        if not module:
            return []
        if hasattr(module, "__all__"):
            names = module.__all__
        else:
            name = module.__name__.rsplit(',', 1)[-1]
            # camelize
            name = ''.join([i.title() for i in name.split('_')])
            names = [name]
        for i in names:
            try:
                out = instance_type(self, class_type, i, getattr(module, i))
                yield out
            except AttributeError:
                warnings.warn(f"Can not import {i} from {module.__name__}. "
                              "Probably broken ASE plugin.")

    def instances_of(self, class_type):
        if class_type not in self._instances:
            self._instances[class_type] = self._instances_of(class_type)
        return self._instances[class_type]

    def info(self, prefix='', opts={}):
        info = f'{prefix}{self.name}'

        prefix += '  '
        opts = opts.copy()
        opts['plugin'] = False

        for instances in self.plugins.instances():
            itype = instances.class_type
            inst = self.instances_of(itype)
            if inst:
                p = f'\n{prefix}{instances.singular_name}: '
                for i in inst:
                    info += i.info(p, opts)
        return info

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"
