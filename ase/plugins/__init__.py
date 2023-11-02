"""Atomic Simulation Environment plugin package."""
__path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .builtins.plugins import Plugins

plugins = Plugins('ase.plugins')
calculators = plugins.calculators

__all__ = [
    'plugins',
    'calculators'
]
