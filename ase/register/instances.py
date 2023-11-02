""" Instances are the "week" (in terms that they not need to import
the real implementator, if they are not actually used) references
to the classes/modules/function...whatever implements a given
functionality, e.g. Calculators, IOFormats etc."""

from ase.utils import lazyproperty
import importlib
from . listing import Listing


class BaseInstance:
    """
    A class, that holds information about an implementation of a calculator,
    viewer, IO, whatsoever.
    """

    def register(self):
        self.plugin.instances[self.class_type][self.name] = self

    def info(self, prefix: str = '', opts={}):
        raise NotImplementedError()


class Instance(BaseInstance):
    """ An instance, that is implemented by a class """

    def __init__(self, plugin, class_type, name, cls):
        self.plugin = plugin
        self.class_type = class_type
        self.name = name
        self.cls = cls

    @lazyproperty
    def implementation(self):
        module, cls = self.cls.rsplit(',', 1)
        module = importlib.import_module(module)
        return module.cls

    def __call__(self):
        self.implementation

    def __repr__(self):
        return f"<ASE {self.class_type}: {self.name} provided by {self.cls}>"

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

    item_class = Instance

    def __init__(self, plugin_list, class_type: str):
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
            yield from p.instances[self.class_type].values()

    @property
    def items(self):
        return self.instances

    def find_by_name(self, name):
        out = self.find_by('names', name.lower())
        if not out:
            out = self.find_by('lowercase_names', name.lower())
        return out

    def __repr__(self):
        return f"<ASE list of {self.class_type}>"

    def __getitem__(self, name):
        return super.__getitem__(name).implementation


class CalculatorInstance(Instance):

    def __init__(self, plugin, class_type, name, cls):
        super().__init__(plugin, class_type, name, cls)

    @lazyproperty
    def availability(self):
        return self.cls.availability_information()


class CalculatorInstances(Instances):
    """ Just a few specialities for instances of calculators """

    item_type = CalculatorInstance

    def info(self, prefix='', opts={}):
        return f"{prefix}Calculators:\n" \
               f"{prefix}------------\n" + super().info(prefix + '  ', opts)
