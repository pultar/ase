import ase.io.formats as ioformats
from ase.register.plugins import get_currently_registered_plugin


def register(where, cls: str, name=None):
    """ Register a calculator exposed by a plugin.
    The name can be derived from the cls name
    """
    if not name:
        name = cls.rsplit(".", 1)[-1]
    plugin = get_currently_registered_plugin()
    icls = plugin.plugins.instances_of(where).item_class
    instance = icls(plugin, where, name, cls)
    instance.register()


def register_calculator(cls: str, name=None):
    """ Register a calculator exposed by a plugin.
    The name can be derived from the cls name
    """
    register('calculators', cls, name)


def register_io_format(module, desc, code, *, name=None, ext=None,
                       glob=None, magic=None, encoding=None,
                       magic_regex=None):
    if not name:
        name = module.rsplit(".", 1)[-1]
    fmt = ioformats.define_io_format(name, desc, code,
                                     module=module,
                                     ext=ext,
                                     glob=glob,
                                     magic=magic,
                                     encoding=encoding,
                                     magic_regex=magic_regex,
                                     external=True)
    fmt.plugin = get_currently_registered_plugin()
    return fmt
