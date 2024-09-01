""" This module contains the classes for listing plugins and
the pluggables (Calculators, IOs, etc...) provided by the plugins.


The structure is as follows

```
-----------                     --------------------------------
|         |                     |                              |
| Plugins |--------1:n----------|   Plugin (can provide some   |
|         |  (for each plugin   |   calculator, formats...)    |
|         |      package)       |                              |
^^^^^^^^^^                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |                                           |
   1:n  for each type                         1:n - plugin can return lists
    |   (i.e viewer, calculator)                | of pluggables
    |                                           | of each pluggable type
    |                                           | (calculator, viewer)
    |                                           | that it provides
    |                                           |
    |                                           |
    |                                           |
-------------------------                ------------------------
|                       |                |                      |
|    Pluggables:        |                |    Pluggable:        |
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
```

To register a plugin, see a docstring of
:module:`ase.plugins.builtin.plugins`
"""

import warnings
from contextlib import contextmanager

from ase.utils import lazyproperty

from .listing import Listing

_current_plugin = None


@contextmanager
def within_the_plugin(plugin):
    """ When a plugin register() function is called,
    the registered plugin would have to say which plugin
    am I.
    This can lead to errors, so importing of the plugin
    is enclosed by this helper, due to the
    :func:get_currently_registered_plugin
    can say, which plugin is currently imported.
    """

    global _current_plugin
    ocp = _current_plugin
    _current_plugin = plugin
    yield
    _current_plugin = ocp


def get_currently_registered_plugin():
    """
    Which plugin is imported and so to which plugin
    belongs currently imported Pluggables

    See the :func:within_the_plugin
    """
    plugin = _current_plugin
    if not plugin:
        import ase.plugins as ase_plugins
        plugin = ase_plugins.plugins['external']
    return plugin


class Plugins(Listing):
    """ A class, that holds all the installed plugins in
        the given namespace package.
        Plugins are registered by importing register subpackage.
    """

    def __init__(self, pluggable_types):
        self._pluggables = {
            k: cls(k)
            for k, cls in pluggable_types.items()
        }
        self._items = {}

    def register(self, plugin):
        self._items[plugin.name] = plugin

    def pluggables_of(self, class_type):
        return self._pluggables[class_type]

    def all_pluggables(self):
        return self._pluggables.values()

    @lazyproperty
    def calculators(self):
        return self.pluggables_of('calculators')

    @lazyproperty
    def viewers(self):
        return self.pluggables_of('viewers')

    @lazyproperty
    def io_formats(self):
        return self.pluggables_of('io_formats')

    def __repr__(self):
        return "<ASE plugins>"

    def info(self, prefix='', opts={}, filter=None):
        return "Plugins:\n"\
               "--------\n" + super().info(prefix, opts, filter)

    def create_plugin(self, module, name=None):
        """ A factory method to create a plugin. """
        return Plugin(self, module, name)


class Plugin:
    """ A class, that encapsulates a plugin package """

    def __init__(self, plugins, package, name=None):
        self.plugins = plugins
        self.package = package
        self.pluggables = {
            i.class_type: {} for i in plugins.all_pluggables()
        }
        self.modules = {}
        self._pluggables = {}
        self.registered = False
        if name is None:
            name = self.package.__name__[12:]   # get rig 'ase.plugins'
        self.name = name

    def add_pluggable(self, pluggable):
        """ Called by Pluggable.register() """
        self.pluggables[pluggable.class_type][pluggable.name] = pluggable
        self.plugins.pluggables_of(pluggable.class_type).add(pluggable)

    @property
    def lowercase_names(self):
        return (self.name.lower(), )

    def register(self):
        """ Register the pluggables in the plugin:
        Call the ase_register function from the
        __init__.py of the plugin package
        """
        if not self.registered:
            try:
                if hasattr(self.package, 'ase_register'):
                    with within_the_plugin(self):
                        self.package.ase_register()
            except Exception as e:
                warnings.warn(f"Can not register plugin {self} because of {e}")
            self.registered = True
            self.plugins.register(self)

    def info(self, prefix='', opts={}):
        info = f'{prefix}{self.name}'

        prefix += '  '
        opts = opts.copy()
        opts['plugin'] = False

        for pluggables in self.plugins.all_pluggables():
            itype = pluggables.class_type
            inst = self.pluggables[itype]
            if inst:
                p = f'\n{prefix}{pluggables.singular_name}: '
                for i in inst.values():
                    info += i.info(p, opts)
        return info

    def __repr__(self):
        return f"<ASE plugin: {self.package}>"
