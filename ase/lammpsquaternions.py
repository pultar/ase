from ase.lammpsatoms import LammpsAtoms
from ase.quaternions import Quaternions


class LammpsQuaternions(LammpsAtoms, Quaternions):
    ''' Inherits methods made in LammpsAtoms followed by Quaternions
     followed by Atoms'''
    pass