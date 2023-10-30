import importlib
import pkgutil
import warnings
cache = {}


def list_plugins(plugin_class:str):
    importlib
    if plugin_class not in cache:
        plugins = []
        clsname = 'ase.plugins.' + plugin_class
        package=importlib.import_module(clsname)

        def add(module, i):
          try:
            plugins.append(getattr(module, i))
          except AttributeError:
            warnings.warn(f"Can not import {i} from {module.__name__}. Probably broken ASE plugin.")

        for mod in pkgutil.iter_modules(package.__path__):
            try:
              name = clsname + '.' + mod.name
              module=importlib.import_module(name)
              if hasattr(module, "__all__"):
                for i in module.__all__:
                    add(module, i)
              else:
                name = mod.name
                # camelize
                name = ''.join([i.title() for i in mod.name.split('_')])
                add(module, name)

            except ImportError:
              warnings.warn(f"Can not import {name} in {i.finder.path}. Probably broken ASE plugin.")
        cache[clsname] = plugins
    return cache[clsname]
