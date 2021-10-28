from ase.calculators.calculator import Calculator, all_changes
from ase.io.trajectory import Trajectory
from ase.parallel import broadcast
from ase.parallel import world
import numpy as np
from os.path import exists
from ase.units import fs, mol, kJ, nm


def restart_from_trajectory(prev_traj, *args, prev_steps=None, atoms=None, **kwargs):
    """ This function helps the user to restart a plumed simulation 
    from a trajectory file. 

    Parameters
        ----------  
        calc: Calculator object
            It  computes the unbiased forces

    .. note:: As alternative for restarting a plumed simulation, the user
            has to fix the positions, momenta and Plumed.istep
    """
    atoms.calc = Plumed(*args, atoms=atoms, restart=True, **kwargs)

    with Trajectory(prev_traj) as traj:
        if prev_steps is None:
            atoms.calc.istep = len(traj) - 1
        else:
            atoms.calc.istep = prev_steps
        atoms.set_positions(traj[-1].get_positions())
        atoms.set_momenta(traj[-1].get_momenta())
    return atoms.calc


class Plumed(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    def __init__(self, calc, input, timestep, atoms=None, kT=1., log='', 
                 restart=False, use_charge=False, update_charge=False):
        """
        Plumed calculator is used for simulations of enhanced sampling methods
        with the open-source code PLUMED (plumed.org).

        [1] The PLUMED consortium, Nat. Methods 16, 670 (2019)
        [2] Tribello, Bonomi, Branduardi, Camilloni, and Bussi, 
        Comput. Phys. Commun. 185, 604 (2014)

        Parameters
        ----------  
        calc: Calculator object
            It  computes the unbiased forces

        input: List of strings
            It contains the setup of plumed actions

        timestep: float
            Timestep of the simulated dynamics

        atoms: Atoms
            Atoms object to be attached


        .. note:: For this case, the calculator is defined strictly with the
            object atoms inside. This is necessary for initializing the
            Plumed object. For conserving ASE convention, it can be initialized as
            atoms.calc = (..., atoms=atoms, ...)

        kT: float. Default 1.
            Value of the thermal energy in eV units. It is important for
            some of the methods of plumed like Well-Tempered Metadynamics.

        log: string
            Log file of the plumed calculations

        restart: boolean. Default False
            True if the simulation is restarted.

        use_charge: boolean. Default False
            True if you use some collective variable which needs charges. If 
            use_charges is True and update_charge is False, you have to define 
            initial charges and then this charge will be used during all simulation.

        update_charge: boolean. Default False
            True if you want the carges to be updated each time step. This will
            fail in case that calc does not have 'charges' in its properties. 


        .. note:: In order to guarantee a well restart, the user has to fix momenta,
            positions and Plumed.istep, where the positions and momenta corresponds
            to the last coniguration in the previous simulation, while Plumed.istep 
            is the number of timesteps performed previously. This can be done 
            using ase.calculators.plumed.restart_from_trajectory.

        """

        from plumed import Plumed as pl

        if atoms is None:
            raise TypeError('plumed calculator has to be defined with the object atoms inside.')

        self.istep = 0
        Calculator.__init__(self, atoms=atoms)

        self.input = input
        self.calc = calc
        self.use_charge = use_charge
        self.update_charge = update_charge
        self.name = '{}+Plumed'.format(self.calc.name)

        if world.rank == 0:
            natoms = len(atoms.get_positions())
            self.plumed = pl()

            # Units setup
            # warning: outputs from plumed will still be in plumed units.

            ps = 1000 * fs
            self.plumed.cmd("setMDEnergyUnits", mol/kJ)  # kjoule/mol to eV
            self.plumed.cmd("setMDLengthUnits", 1/nm)    # nm to Angstrom
            self.plumed.cmd("setMDTimeUnits", 1/ps)      # ps to ASE time units 
            self.plumed.cmd("setMDChargeUnits", 1.)      # ASE and plumed - charge unit is in e units
            self.plumed.cmd("setMDMassUnits", 1.)        # ASE and plumed - mass unit is in e units

            #self.plumed.cmd("setPlumedDat", self.fn)

            self.plumed.cmd("setNatoms", natoms)
            self.plumed.cmd("setMDEngine", "ASE")
            self.plumed.cmd("setLogFile", log)

            self.plumed.cmd("setTimestep", float(timestep))
            self.plumed.cmd("setRestart", restart)

            #self.plumed.cmd("setKbT", float(kT)) #niet in yaff
            self.plumed.cmd("init")
            for line in input:
                self.plumed.cmd("readInputLine", line)

        self.atoms = atoms

    def calculate(self, atoms=None, properties=['energy', 'forces', 'stress', 'free_energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)
        energy, forces = self.compute_energy_and_forces(self.atoms.get_positions(), self.istep)

        self.results['energy'], self. results['forces'] = energy, forces
        self.results['free_energy'] = energy
        if 'stress' in properties:
            self.results['stress'] = self.calc.get_stress() + self.calculate_stress_bias(atoms, d=1e-06, voigt=True)
        self.istep += 1

    def compute_energy_and_forces(self, pos, istep):
        unbiased_energy = self.calc.get_potential_energy(self.atoms)
        unbiased_forces = self.calc.get_forces(self.atoms)

        if world.rank == 0:
            ener_forc = self.compute_bias(pos, istep, unbiased_energy)
        else:
            ener_forc = None
        energy_bias, forces_bias = broadcast(ener_forc)
        energy = unbiased_energy + energy_bias
        forces = unbiased_forces + forces_bias
        return energy, forces

    def compute_bias(self, pos, istep, unbiased_energy):
        self.plumed.cmd("setStep", istep)

        if self.use_charge:
            if 'charges' in self.calc.implemented_properties and self.update_charge:
                charges = self.calc.get_charges(atoms=self.atoms.copy()) 

            elif self.atoms.has('initial_charges') and not self.update_charge:
                charges = self.atoms.get_initial_charges()

            else:
                assert not self.update_charge, "Charges cannot be updated"
                assert self.update_charge, "Not initial charges in Atoms"

            self.plumed.cmd("setCharges", charges)

        self.plumed.cmd("setPositions", pos)
        #self.plumed.cmd("setEnergy", unbiased_energy)  #not in yaff
        self.plumed.cmd("setMasses", self.atoms.get_masses())

        cell = self.atoms.get_cell()
        # print(cell[:])
        self.plumed.cmd("setBox", cell[:])

        forces_bias = np.zeros((self.atoms.get_positions()).shape)
        self.plumed.cmd("setForces", forces_bias)
        virial = np.zeros((3, 3))
        self.plumed.cmd("setVirial", virial)
        self.plumed.cmd("prepareCalc")
        self.plumed.cmd("performCalcNoUpdate")
        energy_bias = np.zeros((1,))
        self.plumed.cmd("getBias", energy_bias)

        self.plumed.cmd("update")

        # print("bias forces") 
        # print(forces_bias) 

        return [energy_bias, forces_bias]

    def calculate_stress_bias(self, atoms, d=1e-06, voigt=True):
        '''

        calculate stress due to plumed bias, 
        code adapted from calculator/calculate_numerical_stress

        '''

        unbiased_energy = self.calc.get_potential_energy(self.atoms)
        stress = np.zeros((3, 3), dtype=float)

        cell = atoms.cell.copy()
        V = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] += d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = self.compute_bias(self.atoms.get_positions(), self.istep, unbiased_energy)[0]

            x[i, i] -= 2 * d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = self.compute_bias(self.atoms.get_positions(), self.istep, unbiased_energy)[0]

            stress[i, i] = (eplus - eminus) / (2 * d * V)
            x[i, i] += d

            j = i - 2
            x[i, j] = d
            x[j, i] = d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eplus = self.compute_bias(self.atoms.get_positions(), self.istep, unbiased_energy)[0]

            x[i, j] = -d
            x[j, i] = -d
            atoms.set_cell(np.dot(cell, x), scale_atoms=True)
            eminus = self.compute_bias(self.atoms.get_positions(), self.istep, unbiased_energy)[0]

            stress[i, j] = (eplus - eminus) / (4 * d * V)
            stress[j, i] = stress[i, j]
        atoms.set_cell(cell, scale_atoms=True)

        if voigt:
            return stress.flat[[0, 4, 8, 5, 2, 1]]
        else:
            return stress

    def write_plumed_files(self, images):
        """ This function computes what is required in
        plumed input for some trajectory.

        The outputs are saved in the typical files of
        plumed such as COLVAR, HILLS """
        for i, image in enumerate(images):
            pos = image.get_positions()
            self.compute_energy_and_forces(pos, i)
        return self.read_plumed_files()

    def read_plumed_files(self, file_name=None):
        read_files = {}
        if file_name is not None:
            read_files[file_name] = np.loadtxt(file_name, unpack=True)
        else:
            for line in self.input:
                if line.find('FILE') != -1:
                    ini = line.find('FILE')
                    end = line.find(' ', ini)
                    if end == -1:
                        file_name = line[ini+5:]
                    else:
                        file_name = line[ini+5:end]
                    read_files[file_name] = np.loadtxt(file_name, unpack=True)

            if len(read_files) == 0:
                if exists('COLVAR'):
                    read_files['COLVAR'] = np.loadtxt('COLVAR', unpack=True)
                if exists('HILLS'):
                    read_files['HILLS'] = np.loadtxt('HILLS', unpack=True)
        assert not len(read_files) == 0, "There are not files for reading"
        return read_files

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.plumed.finalize()
