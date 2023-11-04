"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ase.register.plugins import Plugins
from ase.register.plugables import CalculatorPlugables
from ase.io import formats as _formats
from ase.visualize import viewers as _viewers

plugins = Plugins('ase.plugins', {
    'calculators': CalculatorPlugables,
    'io_formats': _formats.IOFormatPlugables,
    'viewers': _viewers.ViewerPlugables
})

calculators: CalculatorPlugables = plugins.calculators
io_formats: _formats.IOFormatPlugables = plugins.io_formats
viewers: _viewers.ViewerPlugables = plugins.viewers

plugins.register()

__all__ = [
    'plugins',
    'calculators',
    'io_formats',
    'viewers'
]

# set up the legacy ways how to get the pluggables
# it has to be here to avoid circular import
_viewers.VIEWERS = viewers
_viewers = viewers.filter(lambda item: isinstance(item, _viewers.CLIViewer))
_viewers = viewers.filter(lambda item: isinstance(item, _viewers.PyViewer))

_formats.ioformats = io_formats
# Aliased for compatibility only. Please do not use.
_formats.all_formats = io_formats
_formats.extension2format = io_formats.view_by('extensions')
