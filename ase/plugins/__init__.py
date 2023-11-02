"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ase.register.plugins import Plugins
from ase.register.instances import CalculatorInstances
from ase.io.formats import IOFormatInstances

plugins = Plugins('ase.plugins', {
    'calculators': CalculatorInstances,
    'io_formats': IOFormatInstances
})

plugins.register()
calculators = plugins.calculators
io_formats = plugins.io_formats

__all__ = [
    'plugins',
    'calculators',
    'io_formats'
]
