""" Plugables are the "week" (in terms that they not need to import
the real implementator, if they are not actually used) references
to the classes/modules/function...whatever implements a given
functionality, e.g. Calculators, IOFormats etc."""

from ase.utils import lazyproperty
import importlib
from . listing import Listing


class BasePlugable:
    """
    A class, that holds information about an implementation of a calculator,
    viewer, IO, whatsoever.
    """
    def register(self):
        self.plugin.add_plugable(self)
        self.plugin.plugins.plugables_of(self.class_type).add(self)

    def info(self, prefix: str = '', opts={}):
        raise NotImplementedError()

    @lazyproperty
    def lowercase_names(self):
        """ Some plugables can be found not only by the name of the plugable,
        but e.g. by the name of the implementing class """
        return (self.name.lower(),)


class Plugable(BasePlugable):
    """ An plugable, that is implemented by a class """

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


class Plugables(Listing):
    """ This class holds all the Plugables (Calculators, Viewers, ...)
    of one type, that can be used by user """

    item_type = Plugable

    def __init__(self, plugin_list, class_type: str):
        super().__init__()
        self.plugins = plugin_list
        self.class_type = class_type

    @lazyproperty
    def singular_name(self):
        """ Return the human readable name of plugable it contains,
        for the purpose of the output. E.g.
        > CalculatorPlugables(None, 'io_formats').singular_name
        'io format'
        """
        return self.class_type[:-1].replace('_', ' ')

    def find_by_name(self, name):
        """ Plugables can be found by their lowercased name (and ), too """
        out = self.find_by('names', name.lower())
        if not out:
            out = self.find_by('lowercase_names', name.lower())
        return out

    def __repr__(self):
        return f"<ASE list of {self.class_type}>"

    def __getitem__(self, name):
        return super.__getitem__(name).implementation


class CalculatorPlugable(Plugable):
    """ Calculators plugable has now no special implementation,
    but with an introdution of Configuration, there should be
    a better info() method (that shows whether the calculator
    is configured or not)
    """
    pass


class CalculatorPlugables(Plugables):
    """ Just a few specialities for plugables calculators """

    item_type = CalculatorPlugable

    def info(self, prefix='', opts={}):
        return f"{prefix}Calculators:\n" \
               f"{prefix}------------\n" + super().info(prefix + '  ', opts)
