from ase.calculators.calculator import Calculator
import importlib


def mock_if_not_exists(module, _name):
    try:
        module = importlib.import_module(module, package=None)
        return getattr(module, _name)
    except Exception:

        class NonexistentCalculator(Calculator):
            """ Just a placeholder for an non-existing calculator """

            is_available_if = False

            name = _name

            def __init__(self, *args, **kwargs):
                raise NotImplementedError(f"Calculator {name} in module {module} can not be imported and so it is not available")
        return NonexistentCalculator
