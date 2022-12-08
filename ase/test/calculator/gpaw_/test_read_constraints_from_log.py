"""A simple test for reading in the constraints given the gpaw logfile
I do not have access to the ase-datafiles repo, so I am adding the logfile
into the same directory
"""
from ase.io import read
from pathlib import Path
import numpy as np


def test_gpaw_constraints_from_log():
    parent = str(Path(__file__).parent)
    gpaw_logfile = parent + '/log_for_constraint_reading'

    # try:
    atoms = read(gpaw_logfile)
    # except IOError:
    # raise IOError('gpaw logfile could not be read. This could have multiple '
    #               'reasons')

    assert len(atoms.constraints) == 16

    constraints = {i: '' for i in range(len(atoms))}
    # Read the labels that should be in the positions table from
    # atoms.constraints
    for const in atoms.constraints:
        const_label = [i for i in const.todict()['name']
                       if i.lstrip('Fix').isupper()][0]

        indices = []
        for key, value in const.__dict__.items():
            # Since the indices in the varying constraints are labeled
            # differently we have to search for all the labels
            if key in ['a', 'index', 'pairs', 'indices']:
                indices = np.unique(np.array(value).reshape(-1))

        for index in indices:
            constraints[index] += const_label

    # Read the positions table to compare whether the labels have been
    # set correctly.

    infile = open(gpaw_logfile)
    lines = infile.readlines()
    infile.close()

    i1 = 0
    for i2, line in enumerate(lines):
        if i1 > 0:
            if not len(line.split()):
                i3 = i2
                break
        if len(line.split()):
            if line.split()[0].rstrip(':') == 'Positions':
                i1 = i2 + 1

    # Check if labels in the table correspond to the indices of the contraints
    for n, line in enumerate(lines[i1:i3]):
        assert constraints[n] == line.split()[5]
