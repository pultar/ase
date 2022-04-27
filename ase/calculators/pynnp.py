
import pynnp
import numpy as np
from ase.data import chemical_symbols, atomic_numbers
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)



def elementmap_from_element_list(elements):
    nnp_element_map = pynnp.ElementMap()
    #nnp_structure = pynnp.Structure()
    element_line = ''
    for el in elements:
        element_line += el+' '
    nnp_element_map.registerElements(element_line)
    return nnp_element_map


def make_structure_from_ase_atoms(ase_atoms, elementMap):
    nnp_structure = pynnp.Structure()
    nnp_structure.setElementMap(elementMap)

    for ase_atom in ase_atoms:
        nnp_atom = pynnp.Atom()
        nnp_atom.r[0] = ase_atom.x
        nnp_atom.r[1] = ase_atom.y
        nnp_atom.r[2] = ase_atom.z
        nnp_structure.addAtom(nnp_atom, ase_atom.symbol)
            
    nnp_structure.isPeriodic = ase_atoms.pbc[0] # I should check all of the directions eventually
    nnp_structure.isTriclinic = True # does it matter if this is always true?
    
    for i in range(3):
        for j in range(3):
            nnp_structure.box[i][j] = ase_atoms.cell[i][j]

    return nnp_structure


def setup_nnp_mode(input_file, scaling_file, weight_file_format, 
    use_unscaled_symmetry_functions = False):

    # Initialize NNP setup (symmetry functions only).
    nnp_mode = pynnp.Mode()
    nnp_mode.initialize()
    nnp_mode.loadSettingsFile(input_file)
    nnp_mode.setupNormalization()

    nnp_mode.setupElementMap()
    nnp_mode.setupElements()
    nnp_mode.setupCutoff()
    
    #print('nnp_mode.setupSymmetryFunctions()')
    nnp_mode.setupSymmetryFunctions()
    #print('nnp_mode.setupSymmetryFunctionMemory()')
    nnp_mode.setupSymmetryFunctionMemory()
    #print('nnp_mode.setupSymmetryFunctionCache()')
    nnp_mode.setupSymmetryFunctionCache()
    #print('nnp_mode.setupSymmetryFunctionGroups()')
    nnp_mode.setupSymmetryFunctionGroups()
    
    if use_unscaled_symmetry_functions:
        nnp_mode.setupSymmetryFunctionScalingNone()
    else:
        nnp_mode.setupSymmetryFunctionScaling(scaling_file)
        # these are flags for warnings, what should we do? False supresses them
        #nnp_mode.setupSymmetryFunctionStatistics(True, True, True, False)
        nnp_mode.setupSymmetryFunctionStatistics(False, False, False, False) 

    nnp_mode.setupNeuralNetwork()
    nnp_mode.setupNeuralNetworkWeights(weight_file_format)

    return nnp_mode




def calculate_on_nnp_structure(nnp_str, nnp_mode):

    use_normalization = nnp_mode.useNormalization()
    meanEnergy = nnp_mode.getMeanEnergy()
    convEnergy = nnp_mode.getConvEnergy()
    convLength = nnp_mode.getConvLength()

    nnp_mode.removeEnergyOffset(nnp_str);
    # If normalization is used, convert structure data.
    if use_normalization:
        s.toNormalizedUnits(meanEnergy, convEnergy, convLength)

    # Retrieve cutoff radius form NNP setup.
    cutoffRadius = nnp_mode.getMaxCutoffRadius()
    #print("Cutoff radius = ", cutoffRadius/convLength)

    # Calculate neighbor list.
    nnp_str.calculateNeighborList(cutoffRadius)

    # Calculate symmetry functions for all atoms (use groups).
    #nnp_mode.calculateSymmetryFunctions(nnp_str, True)
    nnp_mode.calculateSymmetryFunctionGroups(nnp_str, True)

    # Calculate atomic neural networks.
    nnp_mode.calculateAtomicNeuralNetworks(nnp_str, True)

    # Sum up potential energy.
    nnp_mode.calculateEnergy(nnp_str)

    # Collect force contributions.
    nnp_mode.calculateForces(nnp_str)


    if False:
        print("dir(nnp_str):")
        for meth in dir(nnp_str):
            print(meth)

        print("dir(nnp_mode):")
        for meth in dir(nnp_mode):
            print(meth)

        print("dir(nnp_str.atoms[0]):")
        for meth in dir(nnp_str.atoms[0]):
            print(meth)

        print("------------")
        print("Structure:")
        print("------------")
        print("Cell:") 
        print('a]',nnp_str.box[0])
        print('b]',nnp_str.box[1])
        print('c]',nnp_str.box[2])
        print("numAtoms           : ", nnp_str.numAtoms)
        print("numAtomsPerElement : ", nnp_str.numAtomsPerElement)
        print("------------")
        print("Energy (Ref) : ", nnp_str.energyRef)
        print("Energy (NNP) : ", nnp_str.energy)
        print("------------")
        for atom in nnp_str.atoms:
            print(atom.index, atom.element, nnp_mode.elementMap[atom.element], atom.energy, atom.f.r)
            

def parse_input_file(input_file): 
    
    fid=open(input_file,'r')
    lines_with_comments=fid.readlines()
    fid.close()
    
    parameters = {}
    symfunctions = []
    for line in lines_with_comments:
        if line[0] != '#':
            comment_free_line = line.split('#')[0]
            line_parts = comment_free_line.split()
            if len(line_parts)>0:# skip empty
                if line_parts[0] == 'symfunction_short': #the symfunction keywords are probably the only ones that repeat.
                    symfunctions.append(line_parts)
                else:
                    if len(line_parts)==1:
                        line_parts.append(None) # some parameters are on-off flags so we need a value to store, maybe the value should be '' to make writting easier
                    parameters[line_parts[0]]=line_parts[1:]
                    ### we could do this so that we don't create single element lists
                    #if len(line_parts)>2:
                    #    parameters[line_parts[0]]=line_parts[1:]
                    #else:   
                    #    parameters[line_parts[0]]=line_parts[1]
                    
    return parameters, symfunctions


class PyNNP(Calculator):
    '''uses to n2p2's python interface'''

    implemented_properties = ['energy', 'free_energy', 'energies', 'forces']
    #                          'stress', 'magmom', 'magmoms']

    nolabel = True

    #default_parameters = {'asap_cutoff': False}
    

    def __init__(self, 
                input_file         = 'input.nn',
                weight_file_format = 'weights.%03zu.data',
                scaling_file       = 'scaling.data',
                use_unscaled_symmetry_functions = False,
                atoms=None, 
                **kwargs
                ):
        
        self.elements = None
        self.input_file = input_file
        self.weight_file_format = weight_file_format
        self.scaling_file       = scaling_file
        self.use_unscaled_symmetry_functions = use_unscaled_symmetry_functions
        self.nnp_mode = None
        self.G = None
        self.dGdr = None
        self.dEdG = None
        self.input_parameters = {}
        self.input_symmetry_functions = []
        Calculator.__init__(self, **kwargs)

    def initialize(self):#, atoms):
        self.input_parameters, self.input_symmetry_functions = parse_input_file(self.input_file)
        self.elements = self.input_parameters['elements']
    
        self.elementmap = elementmap_from_element_list(self.elements)
        self.nnp_mode = setup_nnp_mode( self.input_file, self.scaling_file, self.weight_file_format,
                                        self.use_unscaled_symmetry_functions) 

    
    def calculate(self, atoms=None, properties=['energy'],
              system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        #if 'numbers' in system_changes or self.nnp_mode is None:
        #    self.initialize(self.atoms)
        if self.nnp_mode is None:
             self.initialize()#self.atoms)
            
        nnp_str = make_structure_from_ase_atoms(atoms, self.elementmap)
        calculate_on_nnp_structure(nnp_str,self.nnp_mode)
        
        self.results['energy'] = nnp_str.energy
        self.results['free_energy'] = nnp_str.energy
        energies=np.zeros(len(atoms))
        forces  =np.zeros((len(atoms),3))
        self.G = []
        self.dGdR = []
        self.dEdG = []
        for i in range(len(atoms)):
            energies[i] = nnp_str.atoms[i].energy 
            forces[i]    = nnp_str.atoms[i].f.r
            self.G.append(nnp_str.atoms[i].G)
            self.dGdR.append(nnp_str.atoms[i].dGdr)
            self.dEdG.append(nnp_str.atoms[i].dEdG)
        
        self.results['energies'] = energies
        self.results['forces'] = forces
        
        if 'stress' in properties:
            raise PropertyNotImplementedError
