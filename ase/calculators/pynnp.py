
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

    print("")

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
        print(atom.index, atom.element, nnp_mode.elementMap[atom.element], atom.f.r)
            



class PyNNP(Calculator):
    '''uses to n2p2's python interface'''

    implemented_properties = ['energy', 'free_energy', 'energies', 'forces']
    #                          'stress', 'magmom', 'magmoms']

    nolabel = True

    #default_parameters = {'asap_cutoff': False}
    

    def __init__(self, 
                elements, # looks ['Nb', 'Ti', 'Zr', 'O'], could be read from the input file in the future
                input_file         = 'input.nn',
                weight_file_format = 'weights.%03zu.data',
                scaling_file       = 'scaling.data',
                use_unscaled_symmetry_functions = False,
                **kwargs):
        Calculator.__init__(self, **kwargs)

        assert len(elements) > 0
        self.elements = elements
        self.input_file = input_file
        self.weight_file_format = weight_file_format
        self.scaling_file       = scaling_file
        self.use_unscaled_symmetry_functions = use_unscaled_symmetry_functions
        self.nnp_mode = None

    def initialize(self):#, atoms):
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
        #self.results['energies'] = self.energies
        #self.results['free_energy'] = self.energy
        #self.results['forces'] = self.forces
        
        
        
        if 'stress' in properties:
            raise PropertyNotImplementedError
