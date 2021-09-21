import subprocess
from pathlib import Path


from ase.calculators.phx.create_input import write_ph_input, write_q2r_input, write_matdyn_input

class EspressoPhonons:
    def __init__(self, directory, **kwargs):
        self.directory = directory
        self.kwargs_dict = kwargs
    
    def run(self, command):
        write_ph_input(directory=self.directory, **self.kwargs_dict)
        subprocess.run(command, shell=True, cwd=self.directory)

    @classmethod
    def from_scf(cls, scf_dir, phonon_dir, **kwargs):
        subprocess.run(f"cp -r '{scf_dir}' '{phonon_dir}'", cwd="./", shell=True)
        # print(subprocess.run(f"ls", cwd="./", shell=True))
        return cls(Path(phonon_dir), **kwargs)
    
    
    def q2r(self, command="q2r.x -in iq2r.in > oq2r.out", directory="postprocess", **kwargs):
        q2r_dir = (self.directory / directory)
        q2r_dir.mkdir(exist_ok=True)
        print("q2r directory", q2r_dir)
        write_q2r_input(directory=q2r_dir, **kwargs)
        subprocess.run(command, shell=True, cwd=q2r_dir)
        
    def matdyn(self, command="matdyn.x -in imdyn.in > omdyn.out", directory="postprocess", **kwargs):
        matdyn_dir = (self.directory / directory)
        matdyn_dir.mkdir(exist_ok=True)
        print("matdyn directory", matdyn_dir)
        write_matdyn_input(directory=matdyn_dir, **kwargs)
        subprocess.run(command, shell=True, cwd=matdyn_dir)