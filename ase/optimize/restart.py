"""Saving the state of Dynamics objects to a file."""

import builtins
import pickle

import numpy
import ase
from ase.utils import IOContext


class RestartWriter(IOContext):

    def __init__(
        self,
        dyn,
        pickle_file
    ):
        self.dyn = dyn
        self.pickle_file = pickle_file

    def __del__(self):
        self.close()

    def __call__(self):

        data = self.dyn.save_state()

        name = f"{self.pickle_file}.restart.{self.dyn.nsteps}.pkl"

        with open(name, "wb") as f:
            pickle.dump(data, f)


class RestartReader(pickle.Unpickler):

    ALLOWED_MODULES = {
        "builtins": builtins,
        "numpy.random._pickle": numpy.random._pickle,
        "numpy.core.multiarray": numpy.core.multiarray,
        "numpy": numpy,
    }

    def __init__(self, file_path):
        super().__init__(open(file_path, 'rb'))

    def find_class(self, module, name):

        if module not in self.ALLOWED_MODULES:
            raise pickle.UnpicklingError(
                f"attempting to unpickle from unauthorized module: {module}"
            )

        module_obj = self.ALLOWED_MODULES[module]
        return getattr(module_obj, name)
