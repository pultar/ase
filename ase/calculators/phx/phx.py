import subprocess

from ase.calculators.phx.create_input import write

class EspressoPhonons:
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.kwargs_dict = kwargs
    
    def run(self, command):
        write(directory=self.directory, **self.kwargs_dict)
        subprocess.run(command, shell=True, cwd=self.directory)

    @classmethod
    def from_scf(cls, scf_dir, phonon_dir, **kwargs):
        subprocess.run(f"cp -r {scf_dir} {phonon_dir}", cwd="./", shell=True)
        # print(subprocess.run(f"ls", cwd="./", shell=True))
        return cls(target_dir, **kwargs)