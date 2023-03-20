"""Test matching magic for input/output files of different DFT codes."""
from ase.io.formats import ioformats


GPAW_TEXT = b"""

  ___ ___ ___ _ _ _  
 |   |   |_  | | | | 
 | | | | | . | | | | 
 |__ |  _|___|_____|  19.8.2b1
 |___|_|             

"""  # noqa: W291

EXCITING_OUT_TEXT = b"""
==================================================================
| EXCITING CARBON started                                        =
"""

EXCITING_IN_TEXT = b"""<?xml version="1.0"?>
<?xml-stylesheet href="http://xml.exciting-code.org/info.xsl
"""


def test_gpaw_match_magic():
    gpaw = ioformats['gpaw-out']
    assert gpaw.match_magic(GPAW_TEXT)


def test_exciting_match_magic():
    exciting_input = ioformats['exciting-in']
    assert exciting_input.match_magic(EXCITING_IN_TEXT)
    exciting_output = ioformats['exciting-out']
    assert exciting_output.match_magic(EXCITING_OUT_TEXT)
