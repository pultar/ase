"""
Module for managing viewers

For docs on importing viewers using the old entry-point mechanism,
please see the ase.plugins.external.
Now the preffered way how to define a viewer is to create plugin:
package ase.plugins.<yourplugin> and in its __init__.py do

::
  import register_viewer from ase.register
  def ase_register():
      register_viewer(...)
"""

import pickle
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from importlib import import_module

from ase.io import write
from ase.io.formats import ioformats
from ase.register.plugables import BasePlugable, Plugables


class UnknownViewerError(Exception):
    """The view tyep is unknown"""


class AbstractPlugableViewer(BasePlugable):

    def __init__(self, name):
        self.name = name

    def view(self, *args, **kwargss):
        raise NotImplementedError()


class PyViewer(AbstractPlugableViewer):
    def __init__(self, name: str, desc: str, module_name: str):
        """
        Instantiate an viewer
        """
        self.name = name
        self.desc = desc
        self.module_name = module_name

    def _viewfunc(self):
        """Return the function used for viewing the atoms"""
        return getattr(self.module, "view_" + self.name, None)

    @property
    def module(self):
        try:
            return import_module(self.module_name)
        except ImportError as err:
            raise UnknownViewerError(
                f"Viewer not recognized: {self.name}.  Error: {err}"
            ) from err

    def view(self, atoms, *args, **kwargs):
        return self._viewfunc()(atoms, *args, **kwargs)


class CLIViewer(AbstractPlugableViewer):
    """Generic viewer for"""

    def __init__(self, name, fmt, argv):
        self.name = name
        self.fmt = fmt
        self.argv = argv

    @property
    def ioformat(self):
        return ioformats[self.fmt]

    @contextmanager
    def mktemp(self, atoms, data=None):
        ioformat = self.ioformat
        suffix = "." + ioformat.extensions[0]

        if ioformat.isbinary:
            mode = "wb"
        else:
            mode = "w"

        with tempfile.TemporaryDirectory(prefix="ase-view-") as dirname:
            # We use a tempdir rather than a tempfile because it's
            # less hassle to handle the cleanup on Windows (files
            # cannot be open on multiple processes).
            path = Path(dirname) / f"atoms{suffix}"
            with path.open(mode) as fd:
                if data is None:
                    write(fd, atoms, format=self.fmt)
                else:
                    write(fd, atoms, format=self.fmt, data=data)
            d path

    def view_blocking(self, atoms, data=None):
        with self.mktemp(atoms, data) as path:
            subprocess.check_call(self.argv + [str(path)])

    def view(
        self,
        atoms,
        data=None,
        repeat=None,
        **kwargs,
    ):
        """Spawn a new process in which to open the viewer."""
        if repeat is not None:
            atoms = atoms.repeat(repeat)

        proc = subprocess.Popen(
            [sys.executable, "-m", "ase.visualize.viewers"],
            stdin=subprocess.PIPE
        )

        pickle.dump((self, atoms, data), proc.stdin)
        proc.stdin.close()
        return proc


class ViewerPlugables(Plugables):

    item_type = AbstractPlugableViewer

    def info(self, prefix='', opts={}):
        return f"{prefix}IO Formats:\n" \
               f"{prefix}-----------\n" + super().info(prefix + '  ', opts)


def _pipe_to_ase_gui(atoms, repeat, **kwargs):
    buf = BytesIO()
    write(buf, atoms, format="traj")

    args = [sys.executable, "-m", "ase", "gui", "-"]
    if repeat:
        args.append("--repeat={},{},{}".format(*repeat))

    proc = subprocess.Popen(args, stdin=subprocess.PIPE)
    proc.stdin.write(buf.getvalue())
    proc.stdin.close()
    return proc


def define_viewer(
    name, desc, *, module=None, cli=False, fmt=None, argv=None, external=False
):
    if not external:
        if module is None:
            module = name
        module = "ase.visualize." + module
    if cli:
        fmt = CLIViewer(name, fmt, argv)
    else:
        if name == "ase":
            # Special case if the viewer is named `ase` then we use
            # the _pipe_to_ase_gui as the viewer method
            fmt = PyViewer(name, desc, module_name=None)
            fmt.view = _pipe_to_ase_gui
        else:
            fmt = PyViewer(name, desc, module_name=module)
    VIEWERS[name] = fmt
    return fmt


def cli_main():
    """
    This is mainly to facilitate launching CLI viewer in a separate python
    process
    """
    cli_viewer, atoms, data = pickle.load(sys.stdin.buffer)
    cli_viewer.view_blocking(atoms, data)


if __name__ == "__main__":
    cli_main()


# Just here, to avoid circular imports
import ase.plugins as ase_plugins  # NOQA: F401,E402

VIEWERS: ViewerPlugables = ase_plugins.io_formats
CLI_VIEWERS = VIEWERS.filter(lambda item: isinstance(item, CLIViewer))
PY_VIEWERS = VIEWERS.filter(lambda item: isinstance(item, PyViewer))
