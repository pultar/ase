"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ase.register.plugins import Plugins
from ase.register.plugables import CalculatorPlugables
from ase.io import formats as _formats
from ase.visualize import viewers as _viewers
from ase.register.listing import ListingView

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
_viewers.VIEWERS: _viewers.ViewerPlugables = viewers
_viewers.CLI_VIEWERS = \
    viewers.filter(lambda item: isinstance(item, _viewers.CLIViewer))
_viewers.PY_VIEWERS = \
    viewers.filter(lambda item: isinstance(item, _viewers.PyViewer))

_formats.ioformats: _formats.IOFormatPlugables = io_formats
# Aliased for compatibility only. Please do not use.
_formats.all_formats = io_formats
_formats.extension2format: ListingView = io_formats.view_by('extensions')
