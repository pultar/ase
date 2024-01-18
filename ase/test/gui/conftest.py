import os
import pytest

from ase.gui.gui import GUI


@pytest.fixture
def display():
    pytest.importorskip('tkinter')
    if not os.environ.get('DISPLAY'):
        raise pytest.skip('no display')


@pytest.fixture
def guifactory(display):
    guis = []

    def factory(images):
        gui = GUI(images)
        guis.append(gui)
        return gui
    yield factory

    for gui in guis:
        gui.exit()


@pytest.fixture
def gui(guifactory):
    return guifactory(None)
