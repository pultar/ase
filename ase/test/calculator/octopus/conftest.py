import glob
import os
from pathlib import Path
import pytest
from typing import Optional


def clean_dir(path: Path):
    """Clean directory"""
    files = glob.glob(path.as_posix() + '/*')
    print('Removing:')
    for f in files:
        print(f)
        os.remove(f)


# To call fixtures as functions, the fixture should return a function
@pytest.fixture()
def set_test_dir(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Return function that changes to a working test directory"""

    def wrapper(path: Optional[Path] = tmp_path, clean=False) -> None:
        """ Set test directory.

        :param path: Optional directory path. If nothing is
         provided, this defaults to tmp_path.
        :param clean: If True, remove all files from the
         current directory.
        """
        monkeypatch.chdir(path)
        print(f'Test run directory: {os.getcwd()}')
        if clean:
            clean_dir(path)
        return

    return wrapper
