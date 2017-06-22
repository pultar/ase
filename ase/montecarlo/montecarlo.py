# -*- coding: utf-8 -*-
"""Monte Carlo method for ase."""
from __future__ import division

import numpy as np
import ase.units as units
#from ase.io.trajectory import Trajectory



class Montecarlo:
    """ Class for performing MonteCarlo sampling for atoms

    """

    def __init__(self,atoms,temp,indeces = None):
        """ Initiliaze Monte Carlo simulations object

        Arguments:
        atoms : ASE atoms object 
        temp  : Temperature of Monte Carlo simulation in Kelvin
        indeves; List of atoms involved Monte Carlo swaps. default is all atoms.

        """
        self.atoms = atoms
        self.T = temp
        if indeces == None:
            self.indeces = range(len(self.atoms))
        else:
            self.indeces = indeces


        
    def runMC(self,steps = 10):
        """ Run Monte Carlo simulation

        Arguments:
        steps : Number of steps in the MC simulation

        """

        # Atoms object should have attached calculator
        # Add check that this is show
        self.current_energy = self.atoms.get_potential_energy() # Get starting energy

        totalenergies = []
        totalenergies.append(self.current_energy)
        for step in range(steps):
            en, accept = self._mc_step()
            print(accept)
            totalenergies.append(en)
            
        return totalenergies
            

    def _mc_step(self):
        """ Make one Monte Carlo step by swithing two atoms """

        number_of_atoms = len(self.atoms)
        
        rand_a = self.indeces[np.random.randint(0,len(self.indeces))]
        rand_b = self.indeces[np.random.randint(0,len(self.indeces))]
        # At the moment rand_a and rand_b could be the same atom

#        rand_b = np.random.randint(0,number_of_atoms)
        #while (rand_a == rand_b):
        #    rand_b = np.random.randint(0,number_of_atoms)
        
        temp_atom = self.atoms[rand_a].symbol
        self.atoms[rand_a].symbol = self.atoms[rand_b].symbol
        self.atoms[rand_b].symbol = temp_atom
        new_energy = self.atoms.get_potential_energy()
        print(new_energy,self.current_energy)
        accept = False
        if new_energy < self.current_energy:
            self.current_energy = new_energy
            accept = True
        else:
            kT = self.T*units.kB
            energy_diff = new_energy-self.current_energy
            probability = np.exp(-energy_diff/kT)
            probability = min(1.0,probability)
            if np.random.rand() <= probability:
                self.current_energy = new_energy
                accept = True
            else:
                # Reset the sytem back to original 
                self.atoms[rand_a].symbol,self.atoms[rand_b].symbol = self.atoms[rand_b].symbol,self.atoms[rand_a].symbol
        return self.current_energy,accept

        
