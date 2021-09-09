import os
from ase.atoms import Atoms
from ase.calculators.calculator import FileIOCalculator
import numpy as np
from ase.units import Hartree, Bohr

class QUICK(FileIOCalculator):
    '''
    2021-09-08

    Kiyoto Aramis Tanemura

    ASE interface to QUICK (https://quick-docs.readthedocs.io/en/21.3.0/contents.html)
    Load necessary module and source approriate file for using QUICK. Source the quick.rc as instructed in QUICK user manual.
    Then use the QUICK as a calculator in ASE
    Example:
        >>> from ase.build import molecule
        >>> from ase.calculators.quick import QUICK
        >>> geom = molecule('CH3CH2OH')
        >>> geom.calc = QUICK()
        >>> geom.get_charges()
        array([-0.3072, -0.0196, -0.4032,  0.2331,  0.0867,  0.0867,  0.0979, 0.1128,  0.1128])
    '''
    implemented_properties = ['energy', 'forces', 'dipole', 'charges']
    # implement error message if $ASE_QUICK_COMMAND is not present. Likely missed sourcing quick.rc
    command = os.environ['ASE_QUICK'] + ' PREFIX.com'
    discard_results_on_any_change = True

    default_parameters = {'charge': 0,
                          'hamiltonian': 'hf',
                          'dft': 'B3LYP',
                          'basis': '6-31g*'}

    def __init__(self, restart=None, ignore_bad_restart_file=False,
                 label='QUICK', atoms=None, scratch=None, ioplist=list(),
                 basisfile=None, extra=None, addsec=None, **kwargs):
        FileIOCalculator.__init__(self, restart, ignore_bad_restart_file,
                                  label, atoms, **kwargs)


    def write_input(self, atoms, properties=None, system_changes=None):
        FileIOCalculator.write_input(self, atoms, properties, system_changes)
        atoms.write(self.label + '.com', format = 'xyz')
        with open(self.label + '.com', 'r') as f:
            lines = f.readlines()
        # write input parameters/keywords
        lines[0] = self.parameters.hamiltonian.upper()
        if self.parameters.hamiltonian.upper() == 'DFT':
            lines[0] += ' ' + self.parameters.dft.upper()
        lines[0] += ' ' + 'BASIS=' + self.parameters.basis + ' CUTOFF=1.0d-10 DENSERMS=1.0d-6 GRADIENT DIPOLE CHARGE=' + str(self.parameters.charge) + '\n'
        lines[1] = '\n'
        with open(self.label + '.com', 'w') as g:
            g.writelines(lines)
    
    def read_results(self):
        with open(self.label + '.out', 'r') as f:
            lines = f.readlines()
        geom_index = [x for x in range(len(lines)) if 'ANALYTICAL GRADIENT: ' in lines[x]][0] + 4
        charge_index = [x for x in range(len(lines)) if 'ATOMIC CHARGES' in lines[x]][0] + 2
        dipole_index = [x for x in range(len(lines)) if 'DIPOLE (DEBYE)' in lines[x]][0] + 2
        energy_index  = [x for x in range(len(lines)) if 'TOTAL ENERGY' in lines[x]][0]
        elem = []
        mulliken = []
        lowdin = []
        # record elements and atomic charges
        while 'TOTAL' not in lines[charge_index]:
            e, m, l = lines[charge_index].split()
            elem.append(e)
            mulliken.append(float(m))
            lowdin.append(float(l))
            charge_index += 1
        # record coordinates and gradients
        coords = np.zeros([len(elem), 3])
        grads = np.zeros([len(elem), 3])
        i = 0
        readindex = geom_index + i
        while '----------------------------------------' not in lines[readindex]:
            lab, c, g = lines[readindex].split()
            atom_index = i // 3
            axis_index = i % 3
            coords[atom_index, axis_index] = float(c)
            grads[atom_index, axis_index] = float(g)
            i += 1
            readindex = geom_index + i
        
        # record energy and forces
        self.results['energy'] = float(lines[energy_index].split()[-1]) * Hartree 
        self.results['forces'] = - grads * Hartree / Bohr
        self.results['dipole'] = np.array([float(x) for x in lines[dipole_index].split()[:3]]).reshape([1,3])
        self.results['charges'] = np.array(mulliken)
