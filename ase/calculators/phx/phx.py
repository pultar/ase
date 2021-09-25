import subprocess
from pathlib import Path


from ase.calculators.phx.create_input import write_ph_input, write_q2r_input, write_matdyn_input

class EspressoPhononsProfile:
    def __init__(self, argv):
        self.argv = tuple(argv)

    def run(self, directory, inputfile, outputfile):
        from subprocess import check_call
        argv = list(self.argv) + ['-in', str(inputfile)]
        with open(directory / outputfile, 'wb') as fd:
            check_call(argv, stdout=fd, cwd=directory)


# class EspressoPhononTemplate(CalculatorTemplate):
#     def __init__(self):
#         # super().__init__(
#         #     'espresso',
#         #     ['energy', 'free_energy', 'forces', 'stress', 'magmoms'])
#         self.inputname = 'ph.in'
#         self.outputname = 'ph.out'

#     def write_input(self, directory, atoms, parameters, properties):
#         directory.mkdir(exist_ok=True, parents=True)
#         dst = directory / self.inputname
#         write(dst, atoms, format='espresso-in', properties=properties,
#               **parameters)

#     def execute(self, directory, profile):
#         profile.run(directory,
#                     self.inputname,
#                     self.outputname)

#     def read_results(self, directory):
#         path = directory / self.outputname
#         atoms = read(path, format='espresso-out')
#         return dict(atoms.calc.properties())


class EspressoPhonons:
    def __init__(self, profile: EspressoPhononsProfile, directory, **kwargs):
        self.profile = profile
        self.directory = directory
        self.kwargs_dict = kwargs
    
    def run(self):
        write_ph_input(directory=self.directory, infilename='ph.in', **self.kwargs_dict)
        # subprocess.run(command, shell=True, cwd=self.directory)
        # check_call(f"{self.profile.argv} -in {self.profile.}' '{phonon_dir}'",cwd="./")
        self.profile.run(self.directory, 'ph.in', 'ph.out')

    def final_diagonalize(diagonalize_profile=EspressoPhononsProfile(argv=['ph.x'])):
        profile.run(self.directory, 'ph.in', 'phdiag.out')
        

    @classmethod
    def from_scf(cls, scf_dir, phonon_dir, profile, **kwargs):
        from subprocess import check_call
        check_call(f"cp -r '{scf_dir}' '{phonon_dir}'", shell=True, cwd="./")
        return cls(profile, Path(phonon_dir), **kwargs)

def q2r(directory, command="q2r.x -in iq2r.in > oq2r.out", **kwargs):
    q2r_dir = Path(directory)
    q2r_dir.mkdir(exist_ok=True)
    print("q2r directory", q2r_dir)
    write_q2r_input(directory=q2r_dir, **kwargs)
    subprocess.run(command, shell=True, cwd=q2r_dir)

def q2r2(directory, inputname, outputname, command, **kwargs):
    pass
    
def matdyn( directory, command="matdyn.x -in imdyn.in > omdyn.out", **kwargs):
    matdyn_dir = Path(directory)
    matdyn_dir.mkdir(exist_ok=True)
    print("matdyn directory", matdyn_dir)
    write_matdyn_input(directory=matdyn_dir, **kwargs)
    subprocess.run(command, shell=True, cwd=matdyn_dir)
