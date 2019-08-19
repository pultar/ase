from ase.topoatoms import TopoAtoms
from ase.quaternions import Quaternions
import numpy as np


class TopoQuaternions(TopoAtoms, Quaternions):
    ''' Inherits methods made in TopoAtoms followed by Quaternions
     followed by Atoms'''
    def __init__(self, *args, **kwargs):
        quaternions = None
        if 'quaternions' in kwargs:
            quaternions = np.array(kwargs['quaternions'])
            del kwargs['quaternions']
        TopoAtoms.__init__(self, *args, **kwargs)
        if quaternions is not None:
            self.set_array('quaternions', quaternions, shape=(4,))
            # set default shapes
            self.set_shapes(np.array([[3, 2, 1]] * len(self)))
    pass
