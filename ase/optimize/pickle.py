"""Saving the state of Dynamics objects to a file."""

from typing import IO, Any, Union

from ase.parallel import world
from ase.utils import IOContext

import pickle


class PickleLogger(IOContext):

    def __init__(
        self,
        dyn: Any,
        pickle_file: Union[IO, str],
        unique=False,
        mode: str = "a",
    ):
        self.dyn = dyn
        self.pickle_file = pickle_file
        self.mode = mode
        self.unique = unique

    def __del__(self):
        self.close()

    def __call__(self):

        data = self.dyn.todict()

        if self.unique:
            name = f"{self.pickle_file}.{self.dyn.nsteps}"
        else:
            name = self.pickle_file

        with open(name, "wb") as f:
            pickle.dump(data, f)
