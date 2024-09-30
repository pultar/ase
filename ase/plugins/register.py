""" This modules contains functions for creating and registering
the Pluggables (calculators, viewers etc....) """

import functools
from typing import List, Optional, Tuple, Union

import numpy as np

from .plugin import get_currently_registered_plugin


def _register(pluggable_type: str, cls: str, name=None):
    """ Register a calculator or other pluggable exposed by a plugin.
    The name can be derived from the cls name

    Parameters
    ----------
    pluggable_type
      Which to register. E.g. 'calculators', 'io_formats' and so on.
      However, io_formats have its old routine to register, so they
      do not use this mechanism.

    cls: str
      Which class implements the pluggable (e.g.
      ``ase.plugins.emt.EMTCalculator``)
      The class goes by its name only to avoid importing too myuch stuff.
    """
    if not name:
        name = cls.rsplit(".", 1)[-1]
    plugin = get_currently_registered_plugin()
    p_cls = plugin.plugins.pluggables_of(pluggable_type).item_type
    pluggable = p_cls(pluggable_type, name, cls)
    pluggable.register(plugin)


def register_calculator(cls: str, name=None):
    """ Register a calculator exposed by a plugin.
    The name can be derived from the cls name
    """
    _register('calculators', cls, name)


def register_io_format(module, desc, code, *, name=None, ext=None,
                       glob=None, magic=None, encoding=None,
                       magic_regex=None, external=True,
                       allowed_pbc: Optional[List[
                           Union[str, bytes, np.ndarray, List, Tuple]
                       ]] = None):
    """ Just a wrapper for :func:`ioformats.define_io_format`.
    The order of parameters is however slightly different here,
    to be as much as possible similiar to the :func:`register_calculator`

    If not external is set, define_io_format add ase.io to the module.
    """
    if not name:
        name = module.rsplit(".", 1)[-1]
    fmt = formats.define_io_format(name, desc, code,
                                   module=module,
                                   ext=ext,
                                   glob=glob,
                                   magic=magic,
                                   encoding=encoding,
                                   magic_regex=magic_regex,
                                   external=external,
                                   allowed_pbc=allowed_pbc
                                   )
    return fmt


def register_viewer(name, desc, *, module=None, cli=False, fmt=None, argv=None,
                    external=True):
    return viewers.define_viewer(name, desc, module=module, cli=cli,
                                 fmt=fmt, argv=argv, external=external)


def register_function(fn):
    """ Apply on function that register something """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        out.register(get_currently_registered_plugin())
        return out

    return wrapper


def define_to_register(fce):
    """ Some old registering (define_...) function have its own order
    of parameters.
    Not to rewrite the whole list, this function maps from the old
    to the new order of parameters. """

    def register(name, *args, module=None, external=False, **kwargs):
        return fce(module, *args, name=name, external=external, **kwargs)

    return register



import ase.io.formats as formats  # NOQA
import ase.visualize.viewers as viewers        # NOQA

__all__ = ['register_viewer', 'register_io_format', 'register_calculator']
