import subprocess

from ase.calculators.phx.create_input import write

class EspressoPhonons:
    def __init__(self, directory, **kwargs):
        self.directory = directory
        write(directory=directory, **kwargs)
    
    def run(self, command):
        subprocess.run(command, shell=True, cwd=self.directory)