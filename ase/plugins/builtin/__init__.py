"""
Modules of this package are automatically imported.
These modules takes care of registering plugins.
"""

import importlib
import pkgutil
import warnings
from ..plugin import Plugin


def modules():
    """ Return all the plugin packages, that are in the
    given namespace package (so that are in 'ase.plugins') """
    modules = []

    def mod_name(mod):
        return __package__ + '.' + mod.name

    def import_plugin_module(name, path):
        try:
            module = importlib.import_module(name)
            return module
        except ImportError:
            warnings.warn(f"Can not import {name} in {path}."
                          " This ASE plugin is probably broken.")

    modules = (import_plugin_module(mod_name(mod), mod.module_finder.path)
               for mod in pkgutil.iter_modules(__path__))

    modules = (i for i in modules if i)
    return modules


def register_plugins():
    from .. import plugins

    for module in modules():
        if hasattr(module, 'ase_register'):
            name = getattr(module, 'plugin_name', None)
            Plugin(plugins, module, name).register()
        elif hasattr(module, 'ase_register_ex'):
            module.ase_register_ex()
