from ase.topoatoms import TopoAtoms
from ase.quaternions import Quaternions


class TopoQuaternions(TopoAtoms, Quaternions):
    ''' Inherits methods made in TopoAtoms followed by Quaternions
     followed by Atoms'''
    pass