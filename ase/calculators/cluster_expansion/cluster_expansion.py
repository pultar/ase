#Cluster expansion calculator


from ase.calculators.calculator import Calculator
import numpy as np

    
class ClusterExpansion(Calculator):
    implemented_properties = ['energy']
    
    def __init__(self, atoms, ce_parameters,log='CE.log'):
        Calculator.__init__(self)
        
        self.atoms = atoms
        self.parameters = ce_parameters
        self.log = log
        self.old_atoms = None
        
    def calculate(self, atoms, properties, system_changes):
        Calculator.calculate(self, self.atoms)
        
        self.update_energy()
        self.write_output()
        self.old_atoms = self.atoms
        self.results['energy'] = self.energy
        
        #return self.energy
        
    def read_parameters(self):
        # read parameters from the ce parameter file
        # need to make some modifications to parameters/their handling?
        return True
        
    def update_energy(self):
        # for GA/MC only a small subset of the 
        # parameters needs to be updated -->
        # check from the previous geometry which places have changed
        
        if self.old_atoms is None
            self.old_atoms == self.atoms
            self.energy = self.atoms.energy
            
        else:
            #check which atoms have changed places
             indices = self.check_changes(self.new_atoms, self.old_atoms)
             self.energy = self.update_ce(indices)
        
        return self.energy
             
    
    def check_changes(self, new, old):
        #return the indices of atoms which are swapped places
        n_pos = new.positions
        o_pos = old.positions
        
        check = (n_pos == o_pos)
        changed = np.argwhere(check == False)[:,0]
        return np.unique(changed)
        
    def update_ce(self,indices):
         # based on the CE construction, update the local changes in energy
         
         return energy
        
    
    def write_output(self):
        self.log.flush()
        
    
    
    