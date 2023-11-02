"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from ase.register.plugins import Plugins
from ase.register.instances import CalculatorInstances

plugins = Plugins('ase.plugins', {
    'calculators': CalculatorInstances,
})

plugins.register()
calculators = plugins.calculators

__all__ = [
    'plugins',
    'calculators'
]
