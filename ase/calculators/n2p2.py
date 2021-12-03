import sys
import numpy as np
from ase.calculators.calculator import all_changes
from ase.calculators.calculator import FileIOCalculator
#from ase.calculators.genericfileio import GenericFileIOCalculator
import os
import shutil
from ase.io.n2p2 import read_n2p2, write_n2p2

class N2P2Calculator(FileIOCalculator):
#class N2P2Calculator(GenericFileIOCalculator):
    '''example: 
        N2P2Calculator( 
            directory = 'tmp',
            files = [
                'input.nn',
                'scaling.data',
                'weights.008.data',
                'weights.001.data'],
            )
    '''
    
    command = 'nnp-predict 0'
    'Command used to start calculation'

    name = 'n2p2'

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label=None, atoms=None, command=None, files=[], **kwargs):
        """File-IO calculator.

        command: str
            Command used to start calculation.
        """

        self.files = files

        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label,
                            atoms, **kwargs)
        #GenericFileIOCalculator.__init__(self, restart, ignore_bad_restart_file, label,
        #                    atoms, **kwargs)

        if command is not None:
            self.command = command
        else:
            name = 'ASE_' + self.name.upper() + '_COMMAND'
            self.command = os.environ.get(name, self.command)
            
        self.implemented_properties = {
            'energy' : self.calculate,
            'forces' : self.calculate}
        self.results = {}

        ## preparing
        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
        self.write_files()

    def write_input(self, atoms, properties=None, system_changes=None):
        """Write input file(s).

        Call this method first in subclasses so that directories are
        created automatically."""

        if self.directory != os.curdir and not os.path.isdir(self.directory):
            os.makedirs(self.directory)
            
        write_n2p2(
            os.path.join(self.directory, 'input.data'),
            atoms,
            with_energy_and_forces = False)

    def write_files(self): #should this be initialize?
        for filename in self.files:
            src = filename
            basename = os.path.basename(filename)
            dest = os.path.join(self.directory, basename)
            shutil.copyfile(src, dest)

    def read_results(self):

        res_atoms = read_n2p2(
                    filename= os.path.join(self.directory,'output.data'),
                    index=-1, 
                    with_energy_and_forces = True)

        self.results = res_atoms.calc.results
