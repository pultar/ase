"""
In this plugins, external calculators (either the old long known by
ASE, or the ones registered using entry-points) are present
"""

import sys
import warnings
from ase.register import register_calculator
from ase.io.formats import define_io_format
from ase.utils.plugins import ExternalIOFormat
from ase.register.plugins import get_currently_registered_plugin

if sys.version_info >= (3, 8):
    from importlib.metadata import entry_points
else:
    from importlib_metadata import entry_points


def define_external_io_format(entry_point):

    fmt = entry_point.load()
    # if entry_point.name in ioformats:
    #    raise ValueError(f'Format {entry_point.name} already defined')
    if not isinstance(fmt, ExternalIOFormat):
        raise TypeError('Wrong type for registering external IO formats '
                        f'in format {entry_point.name}, expected '
                        'ExternalIOFormat')
    fmt = define_io_format(entry_point.name, **fmt._asdict(), external=True)
    fmt.plugin = get_currently_registered_plugin()
    fmt.register()


def register_external_io_formats(group):
    if hasattr(entry_points(), 'select'):
        fmt_entry_points = entry_points().select(group=group)
    else:
        fmt_entry_points = entry_points().get(group, ())

    for entry_point in fmt_entry_points:
        try:
            define_external_io_format(entry_point)
        except Exception as exc:
            warnings.warn(
                'Failed to register external '
                f'IO format {entry_point.name}: {exc}'
            )


def ase_register():
    register_calculator("asap3.EMT")
    register_calculator("gpaw.GPAW")
    register_calculator("hotbit.Calculator", name='hotbit')

    # Register IO formats exposed through the ase.ioformats entry point
    register_external_io_formats('ase.ioformats')
