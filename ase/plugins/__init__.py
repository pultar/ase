"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ase.register.plugins import Plugins
from ase.register.plugables import CalculatorPlugables
from ase.io.formats import IOFormatPlugables

plugins = Plugins('ase.plugins', {
    'calculators': CalculatorPlugables,
    'io_formats': IOFormatPlugables
})

plugins.register()
calculators: CalculatorPlugables = plugins.calculators
io_formats: IOFormatPlugables = plugins.io_formats

__all__ = [
    'plugins',
    'calculators',
    'io_formats',
    'viewers'
]
