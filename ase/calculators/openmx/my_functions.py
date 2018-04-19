import numpy as np
from ase.spacegroup import crystal
from ase.calculators.calculator import get_calculator
import copy
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory

def curve_fitting(atoms):
    #target = atoms.calc.atoms.get_positions()
    target = atoms[:]
    target_p = target.positions
    import numpy as np
    from scipy.optimize import minimize
    def rosen(xx):
        """The Rosenbrock function"""
        a=xx[0]
        theta = xx[1]
        r = xx[2]
        x = r*np.cos(theta)
        y = r*np.sin(theta)
        z = xx[3]
        my_symprec =1e-1
        while(True):  # Try to construct the atom until we have qualified symprec
            cry = crystal(['Cr','I'],
                    basis = [(1/3,2/3,0.5),( x + y / np.sqrt(3), 2 * np.sqrt(3) / 3 * y, z)],
                    spacegroup = 147,
                    size = (1,1,1),
                    cellpar = [a , a, 19.807, 90, 90, 120],
                    symprec = my_symprec
                    )
            if(len(cry)==len(target)):
                break
            elif(len(cry)>len(target)):
                my_symprec *= 3
            else :
                my_symprec /= 3

        loss =0
        for a in cry:
            j0 = 100
            for t in target_p:
                j1 = np.linalg.norm(a.position - t)
                if(j0 > j1):
                    j0 = j1
                    #print(j0)
            loss += j0
        return loss

    x0 = [6.867, np.pi/3, 1/3, 0.58]
    res = minimize(rosen, x0, method='nelder-mead',
                   options={'xtol': 1e-8, 'disp': True})
    x = res.x
    structure = getAtoms(a = x[0], theta = x[1], r = x[2], z = x[3])
    atoms.set_cell(structure.cell)
    atoms.set_positions(structure.positions)
    return atoms

def optimize(atoms):
    label = atoms.calc.label

    #c = FixAtoms(indices = [atom.index for atom in atoms if atom.symbol=='Cr'])
    c = FixAtoms(indices = [0])
    atoms.set_constraint(c)
    ucf = UnitCellFilter(atoms, mask=[True, True, False, False, False, False])
                                      # xx,   yy,    zz,    yz,   xz,     xy
    dyn = BFGS(ucf, trajectory = label+'.traj', restart = label + '.rst')
    dyn.run(fmax=0.01)
    new_atoms = Trajectory(label + '.traj')[-1]
    new_atoms.calc = atoms.calc
    return new_atoms

def relabel(atoms,label='opt',par=3, pattern='Cr'):
    directory = atoms.calc.directory
    old_prefix = atoms.calc.prefix
    inherit = old_prefix[par:]
    atoms.calc.set_label(directory +'/'+ label + inherit)

    print(atoms.calc.label)
    return atoms

def getAtoms(atom=None, calcName='openmx', a=6.867, charge=None, m=[3, 3],
             M='Cr', X='I',
             theta=1.0470908, r=0.34696889, z=0.58316308, mylabel='bnd',
             band_dispersion=None, band_kpath=None, magmoms_angle=None):
    ''' Get the Variable within the bracket and set the fixed parameters directly
        returns atoms()
    '''
    def getLabel(mylabel=mylabel):
        return '/group1/schinavro/crI3/openMX/'+mylabel+M+X+mlabel

    def getMagmoms(m):
        if len(m) < 3:
            return np.array([m[0],m[1],0,0,0,0,0,0])
        else:
            return np.array([m[0], m[1], 0, 0, 0, 0, 0, 0, m[2], m[3], 0, 0, 0, 0, 0, 0])

    def readPrefix(prefix):
        if('stAF' in prefix):
            mlabel = 'stAF'
            size = (1,2,1) # stripy AFM
            magmoms = getMagmoms([3, 3, -3, -3])
            return (mlabel, size, magmoms)
        elif('zgAF' in prefix):
            mlabel = 'zgAF'
            size = (2,1,1) # zigzag AFM
            magmoms = getMagmoms([-3,3,3,-3])
            return (mlabel, size, magmoms)
        elif('AF' in prefix):
            mlabel = 'AF'
            size = (1,1,1)
            magmoms = getMagmoms([3,-3])
            return (mlabel, size, magmoms)
        else :
            mlabel = 'F'
            size = (1,1,1)
            magmoms = getMagmoms([3, 3])
            return (mlabel, size, magmoms)

    x = r*np.cos(theta)
    y = r*np.sin(theta)

    size = (1,1,1)
    mlabel='F'

    if(atom is not None):
        prefix = atom.calc.prefix
        #magmoms = atom.calc.atoms.get_initial_magnetic_moments() #Want to calculate continuous
        mlabel, size, magmoms = readPrefix(prefix)
        label = getLabel(mylabel = mylabel)
        atom.calc.set_label(label)
        atom.calc.label = label

        a = atom.cell[0,0]
        charge = atom.calc['scf_system_charge']
        fixed_grid = atom.calc['scf_fixed_grid']
        current_iter = atom.calc['md_current_iter']
        restart = 'on'

    else:
        fixed_grid = None
        current_iter = None
        restart = None
        if (m[0] == 3 and len(m) == 4):
            size = (1,2,1) # stripy AFM
            mlabel='stAF'
        elif (m[0] == -3 and len(m) == 4):
            size = (2,1,1) # zigzag AFM
            mlabel = 'zgAF'
        elif (set(m) == set([3, -3]) and len(m)==2):
            mlabel = 'AF'
        magmoms = getMagmoms(m)
        label= getLabel()

    '''
       Adjust the symprec to get the number, which needs to construct the magnetic structure, of atoms.
    '''

    my_symprec =1e-1
    while(True):  # Try to construct the atom until we have qualified symprec
        atoms = crystal([M,X],
                basis = [(1/3,2/3,0.5),( x + y / np.sqrt(3), 2 * np.sqrt(3) / 3 * y, z)],
                spacegroup = 147,
                size = size,
                cellpar = [a , a, 19.807, 90, 90, 120],
                symprec = my_symprec
                )
        if(len(atoms)==len(magmoms)):
            break
        elif(len(atoms)>len(magmoms)):
            my_symprec *= 3
        else :
            my_symprec /= 3

    atoms.set_initial_magnetic_moments(magmoms)          # Set the magmoms
    Calculator = get_calculator(calcName)
    calc = Calculator(pbs={'processes':20, 'walltime':"3:00:00:00"}, nohup=True
                    ,label = label
                    ,energy_cutoff=200*13.6
                    ,kpts=(4,4,1)
                    ,eigenvalue_solver='Band'
                    ,mixing_type='RMM-DIISH'
                    ,xc='GGA'
                    ,scf_criterion=1e-5*13.6
                    ,scf_max_iter=3000
                    ,scf_system_charge=charge
                    ,hubbard_u_values = {'Cr':{'1d':3.0}}
                    ,scf_fixed_grid = fixed_grid
                    ,md_current_iter = current_iter
                    ,scf_restart = restart
                    ,orbital_polarization_enhancement_atom_indices = [0,1,8,9]
                    ,stress = 'on'

                    ,band_dispersion = band_dispersion
                    ,band_kpath = band_kpath

                    #,md_type = 'RFC5'
                    #,md_maxiter= 100
                    #,md_criterion=1.0e-4
                    ,initial_magnetic_moments_euler_angles = magmoms_angle
                    )
    atoms.calc = copy.deepcopy(calc)
    return copy.deepcopy(atoms)


class Functions:

    def __init__(self, readmode=True, subversive=False):
        self.readmode = readmode
        self.subversive = subversive

    def getE(self, atom):
        from pathlib import Path
        '''if subversive is True, it calculate with zero background'''
        atom = copy.deepcopy(atom)
        outfile = Path(atom.calc.label+'.out')
        restartfile = Path(atom.calc.label+'.dat#')
        while True:
            try:
                if(outfile.exists()):
                    print(atom.calc.label + '.out' + 'exist!')
                    if(self.subversive):
                        atom.get_potential_energy()
                    else:
                        atom.calc.read_results()
                        atom.positions = atom.calc.atoms.positions
                        atom.cell = atom.calc.atoms.cell
                elif(restartfile.exists()):
                    print(atom.calc.label + '.dat#' + 'exist!')
                    atom.calc.restart = atom.calc.label+'.dat#'
                    if(self.subversive):
                        atom.get_potential_energy()
                    else:
                        atom.calc.read_results()
                        atom.positions = atom.calc.atoms.positions
                        atom.cell = atom.calc.atoms.cell
                else :
                    print('New Calculation'+atom.calc.label)
                    if(self.readmode):
                        atom.calc.read_results()
                        atom.positions = atom.calc.atoms.positions
                        atom.cell = atom.calc.atoms.cell
                    else:
                        atom.get_potential_energy()
            except :
                print('Something wrong with'+atom.calc.label+'ReIterating')
                #continue
                break
            break
        return copy.deepcopy(atom)
