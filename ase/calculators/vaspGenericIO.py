"""
ASE-interface to VASP using GenericFileIOCalculator.

Work in Progress by Michael Wolloch based on VASP input creation and output
parsing by Martin Schlipf using py4vasp
michael.wolloch@univie.ac.at
martin.schlipf@vasp.at

www.vasp.at
www.vasp.at/py4vasp/latest/
"""

import os
from ase import Atoms
from xml.etree import ElementTree
from ase.calculators import calculator
import copy

from ase.calculators.genericfileio import (CalculatorTemplate,
                                           GenericFileIOCalculator)
import ase.io.vasp_parsers.incar_writer as incar
from ase.io.vasp_parsers.kpoints_writer import write_kpoints
from ase.io.vasp_parsers.vasp_structure_io import write_vasp_structure



class VaspProfile:
    def __init__(self, vasp_command):
        self.vasp_command = vasp_command

    def version(self):
        from subprocess import check_output
        full_version_str = check_output(self.vasp_command + ['--version']).decode('ascii')
        numeric_version = full_version_str.split()[0][5:]
        return numeric_version

    def run(self, directory, outputfile):
        from subprocess import check_call
        with open(directory / outputfile, 'w') as fd:
            check_call(self.vasp_command, stdout=fd, cwd=directory)


class VaspTemplate(CalculatorTemplate):
    def __init__(self):
        super().__init__(
            name='vasp',
            implemented_properties=['energy',
                                    'free_energy',
                                    'forces',
                                    'dipole',
                                    'stress',
                                    'magmoms'],
        )
        
        
    def _read_xml(self, directory):
        """Read vasprun.xml, and return the last calculator object.
        Returns calculator from the xml file.
        Raises a ReadError if the reader is not able to construct a calculator.
        """
        from ase.io.formats import read
        file = directory / 'vasprun.xml'
        incomplete_msg = (
            f'The file "{file}" is incomplete, and no DFT data was available. '
            'This is likely due to an incomplete calculation.')
        try:
            _xml_atoms = read(file, index=-1, format='vasp-xml')
            # Silence mypy, we should only ever get a single atoms object
            assert isinstance(_xml_atoms, Atoms)
        except ElementTree.ParseError as exc:
            raise calculator.ReadError(incomplete_msg) from exc

        if _xml_atoms is None or _xml_atoms.calc is None:
            raise calculator.ReadError(incomplete_msg)

        return _xml_atoms.calc

    def read_results(self, directory):
        """
        Read vasp.h5 ouptut and extract data with py4vasp-core.
        
        For now, just use the existing ASE code and xml parser.
        However, we do not resort the forces since that requires an ase-sort.dat
        file and a lot of other code.
        """
        
        try:
            import py4vasp_core
            print('parsing with py4vasp')
            calc = py4vasp_core.Calculation.from_path(directory)
            return read_from_py4vasp(calc)
        except ModuleNotFoundError:
            results = {}
    
            # Read the data we can from vasprun.xml
            calc_xml = self._read_xml(directory)
            xml_results = calc_xml.results
    
            # Fix sorting
            #xml_results['forces'] = xml_results['forces'][self.resort]
    
            results.update(xml_results)
    
    
            # Stress is not always present.
            # Prevent calculation from going into a loop
            if 'stress' not in results:
                results.update(dict(stress=None))

        return results

    def execute(self, directory, profile):
        profile.run(directory, 'vasp.out')

    def write_input(self, directory, atoms, parameters, profile, properties=None):
        """
        Write INCAR, POSCAR and KPOINTS for VASP 6, using new input creation
        code written by Martin Schlipf and extended by Michael Wolloch to
        incorporate Adam Jackson's new KPOINTS classes.
        
        For VASP 5, the old ASE code is used to create the inputs.

        Also use the properties here eventually.
        """

        if os.environ.get('VASP_POTENTIALS') and profile.version() >= '6.0.0':
            params = copy.deepcopy(parameters)
            kpts = params.pop('kpts', None)
            potential_mapping = params.pop('potential_mapping', {})
            xc = params.pop('xc', None)
            if xc == 'LDA':
                params.update({'POTENTIALTYP': 'LDA'})
            else:
                params.update({'POTENTIALTYP': 'PBE'})
            incar.write_incar(directory, params)
            write_kpoints(directory, kpts)
            write_vasp_structure(file=f"{directory}/POSCAR",
                                atoms=atoms,
                                vasp6=True,
                                potential_mapping=potential_mapping)
        else:
            from ase.calculators.vasp.create_input import  GenerateVaspInput
            vasp_input_generator = GenerateVaspInput()
            vasp_input_generator.initialize(atoms)
            vasp_input_generator.set(**parameters)
            vasp_input_generator.write_input(atoms, directory)


def read_from_py4vasp(calc):
    properties = ["energy","force","stress","magnetism"]
    return {prop: read_prop(calc,prop) for prop in properties}

def read_prop(calc,prop):
    return getattr(calc,prop)[:].to_dict()

class Vasp(GenericFileIOCalculator):
    """
    Vasp calculator.
    """

    def __init__(self,
                 profile=None,
                 directory='.',
                 **kwargs):
        """
        Create Vasp calculator.
        """
        if (profile is None and 'ASE_VASP_COMMAND' in os.environ):
            profile = VaspProfile(os.environ['ASE_VASP_COMMAND'].split())

        super().__init__(profile=profile, template=VaspTemplate(),
                         directory=directory,
                         parameters=kwargs)
