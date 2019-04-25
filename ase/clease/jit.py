try:
    from numba import jit as numba_jit
    jit = numba_jit
except ImportError:
    # Numba is not installed
    def dummy_jit(**options):
        def decorate_func(func):
            def wrapper(*args):
                return func(*args)
            return wrapper
        return decorate_func
    jit = dummy_jit
