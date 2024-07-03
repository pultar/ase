import os
import re
from copy import deepcopy
import warnings
from pathlib import Path

from ase.calculators.genericfileio import (BaseProfile, CalculatorTemplate,
                                           GenericFileIOCalculator)
from ase.io import read, write
from ase.io.pwmat_namelist.namelist import Namelist_pwmat

class PWmatProfile(BaseProfile):
    configvars = {}
    nprocs = os.environ.get('SLURM_NPROCS')
    if nprocs is None:
        raise Exception("slurm platform is necessary") # 

    def __init__(self, pseudo_dir, 
                 command=f'mpirun -np {nprocs} -iface ib0 PWmat | tee output', **kwargs):
        super().__init__(command, **kwargs)
        self.pseudo_dir = Path(pseudo_dir)
        
    def version(self):
        module_list = os.popen('module list').read()
        pattern = r'pwmat/(\d+\.\d+\.\d+)'
        ret = re.search(pattern, module_list)
        if ret:
            return ret.group(1)
        else:
            return 'pwmat version information not found'
    
    def get_calculator_command(self, inputfile):
        return []

class PWmatTemplate(CalculatorTemplate):
    #_label = 'pwmat'

    def __init__(self):
        super().__init__(
            'pwmat',
            ['energy', 'free_energy', 'forces', 'stress', 'magmoms', 'dipole'],
        )
        self.inputname = 'etot.input'
        self.outputname = 'pwmat.out'
        self.errorname = 'ERRORS'

    def write_input(self,profile,directory,atoms,parameters, properties):
        dst = directory/self.inputname
        pp_dir = Path(profile.pseudo_dir)
        input_data = Namelist_pwmat(parameters.pop('input_data',None))
        input_data.to_nested()
        if 'IN.PSP' in list(input_data):
            if not isinstance(input_data['IN.PSP'],list):
                raise ValueError("The keyword IN.PSP must be List")
            in_psp_tmp = deepcopy(input_data['IN.PSP'])
            for i,psp in enumerate(in_psp_tmp):
                pp_path = pp_dir/psp
                if not os.path.exists(pp_path):
                    raise FileNotFoundError("Pseudopotential file does not exist.")
                else:
                    input_data['IN.PSP'][i] = pp_path
        parameters['input_data'] = input_data

        write(
            dst,
            atoms,
            format='pwmat-in',
            properties=properties,
            **parameters,
        )

        config_name = parameters['input_data'].get('IN.ATOM', None)
        if config_name is None:
            config_name = 'atom.config'
        dst_config = directory/config_name
        write(dst_config, atoms, format='pwmat')

    def execute(self, directory, profile):
        profile.run(directory, self.inputname, self.outputname, 
                    errorfile=self.errorname)

    def read_results(self, directory):
        path = directory / 'REPORT'
        path_movement = directory / 'MOVEMENT'
        path_qdiv = directory / 'OUT.QDIV'
        atoms = read(path, format='pwmat-report', 
                     MOVEMENT=path_movement, QDIV=path_qdiv, index=-1)
        return dict(atoms.calc.properties())

    def load_profile(self, cfg, **kwargs):
        return PWmatProfile.from_config(cfg, self.name, **kwargs)

    def socketio_parameters(self, unixsocket, port):
        return {}

    def socketio_argv(self, profile, unixsocket, port):
        if unixsocket:
            ipi_arg = f'{unixsocket}:UNIX'
        else:
            ipi_arg = f'localhost:{port:d}'  # XXX should take host, too
        return profile.get_calculator_command(self.inputname) + [
            '--ipi',
            ipi_arg,
        ]

class PWmat(GenericFileIOCalculator):
    def __init__(
        self,
        *,
        profile=None,
        directory='.',
        **kwargs,
    ):
    
        super().__init__(
            template=PWmatTemplate(),
            profile=profile,
            directory=directory,
            parameters=kwargs,
        )