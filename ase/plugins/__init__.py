"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .pluggables import CalculatorPluggables
from .register import register_viewer, register_io_format, register_calculator # NOQA


from .plugin import Plugins
from ase.io import formats as _formats
from ase.visualize import viewers as _viewers
from .builtin import register_plugins # NOQA

plugins = Plugins({
    'calculators': CalculatorPluggables,
    'io_formats': _formats.IOFormatPluggables,
    'viewers': _viewers.ViewerPluggables
})

calculators: CalculatorPluggables = plugins.calculators
io_formats: _formats.IOFormatPluggables = plugins.io_formats
viewers: _viewers.ViewerPluggables = plugins.viewers

# set up the legacy ways how to get the pluggables
# it has to be here to avoid circular import
_viewers.VIEWERS = viewers
_viewers.CLI_VIEWERS = viewers.cli_viewers
_viewers.PY_VIEWERS = viewers.python_viewers
_formats.ioformats = io_formats
# Aliased for compatibility only. Please do not use.
_formats.all_formats = io_formats
_formats.extension2format = io_formats.by_extension


# install the plugins
register_plugins()


__all__ = [
    'plugins',
    'calculators',
    'io_formats',
    'viewers',
    'register_viewer',
    'register_io_format',
    'register_calculator'
]
