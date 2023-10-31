""" This module contains the classes for listing plugins and
the instances (Calculators, IOs, etc...) provided by the plugins """

import importlib
import pkgutil
import warnings
from ase.utils import lazyproperty


def import_module(name, path):
    try:
      module=importlib.import_module(name)
      return module
    except ImportError:
      warnings.warn(f"Can not import {name} in {path}. Probably broken ASE plugin.")


class Plugins:
    """ A class, that holds all the installed plugins in the given namespace package."""
    def __init__(self, namespace_package):
        self.namespace_package = namespace_package
        self._instances = {}

    def packages(self):
        """ Return all plugin packages, that are in the given namespace package """
        modules = []
        package=importlib.import_module(self.namespace_package)
        modules = (import_module(self.namespace_package + '.' + mod.name, mod.module_finder.path) for mod in pkgutil.iter_modules(package.__path__))

        modules = (i for i in modules if i)
        return modules

    @lazyproperty
    def plugins(self):
        return {p.__name__.rsplit('.',1)[-1]:Plugin(p) for p in self.packages()}

    def __iter__(self):
        return iter(self.plugins.values())

    def instances(self, class_type):
        if class_type not in self._instances:
            self._instances[class_type] = Instances(self, class_type)
        return self._instances[class_type]

    @lazyproperty
    def calculators(self):
        return self.instances('calculators')

    def __getitem__(self, name):
        return self.plugins[name]

    def __repr__(self):
        return f"<ASE plugins from: {self.namespace_package}>"


class Instance:
    """ An implementation of an calculator, viewer, IO, whatsoever """

    def __init__(self, class_type, name, _class):
        self._class = _class
        self.class_type = class_type
        self._name=name

    def __call__(self):
        return self._class

    def __repr__(self):
        return f"<ASE {self.class_type}: {self._name} provided by {self._class}>"

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError()
        return getattr(self._class, name)

    @lazyproperty
    def names(self):
        name=getattr(self._class, 'name', None)
        if isinstance(name, str):
            name=[name]
        if isinstance(name,(list,set,tuple)):
            if self._name not in name:
               name.append(self._name)
        else:
            name = [self._name]

        if self._class.__name__ not in name:
            name.append(self._class.__name__)
        return name

    @lazyproperty
    def lowercase_names(self):
        return {i.lower() for i in self.names}


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, package):
        self.package = package
        self.modules = {}

    def get_module(self, class_type):
        """ Return a module for a given type, e.g. calculator, io, viewer.... """
        if class_type not in self.modules:
            mod = importlib.util.find_spec(self.package.__name__ + '.' + class_type, self.package.__path__)
            if not mod:
                self.modules[class_type] = None
            else:
                module = importlib.util.module_from_spec(mod)
                mod.loader.exec_module(module)
                self.modules[class_type] = module

        return self.modules[class_type]

    def get_instances(self, class_type):
        module = self.get_module(class_type)
        if not module:
            return []
        if hasattr(module, "__all__"):
            names = module.__all__
        else:
            name = module.__name__.rsplit(',',1)[-1]
            # camelize
            name = ''.join([i.title() for i in name.split('_')])
            names = [name]
        for i in names:
            try:
              out = Instance(class_type, i, getattr(module, i))
              yield out
            except AttributeError:
              breakpoint()
              warnings.warn(f"Can not import {i} from {module.__name__}. Probably broken ASE plugin.")

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"


class Instances:

    def __init__(self, plugin_list:Plugins, class_type: str):
        self.plugins = plugin_list
        self.class_type = class_type

    def __iter__(self):
        return iter(self.instances)

    @lazyproperty
    def instances(self):
        """ List the instances (e.g. calculators) from all plugins. """
        return list(self._instances())

    def _instances(self):
        for p in self.plugins:
            yield from p.get_instances(self.class_type)

    @staticmethod
    def instance_has_attribute(obj, attribute, value):
        v=getattr(obj, attribute, None)
        if value == v:
            return True
        if isinstance(v, (list,set)):
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

    def __getitem__(self, name):
        out=self.find_by_name(name)
        if not out:
            raise KeyError(f"There is no implementation of {self.class_type} with a name {name}")
        return out()

    def find_by_name(self, name):
        out=self.find_by('names', name.lower())
        if not out:
            out=self.find_by('lowercase_names', name.lower())
        return out
