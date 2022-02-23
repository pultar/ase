import warnings

_unused_paceholder = object()


def deprecated():
    return _unused_paceholder


def warn_if_used(value, message, tp=FutureWarning):
    if value is not unused_paceholder:
        warnings.warn(message, tp)
