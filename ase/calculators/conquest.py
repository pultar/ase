"""This module defines an ASE interface to CONQUEST.

http://www.order-n.org/
Authors: J Kane Shenton <jkshenton@gmail.com>
         Jack Baker
         J Poulton
         Zamaan Raza
         L Truflandier <lionel.truflandier@u-bordeaux.fr>
"""
from __future__ import print_function

# import warnings
# import numpy as np
from ase.units import Hartree
from pathlib import Path
from shutil import copy
from ase.calculators.calculator import FileIOCalculator, Parameters, ReadError
from ase.io.conquest import ConquestEnv, ConquestError, ConquestWarning


class Conquest(FileIOCalculator):
    """ASE-Calculator for CONQUEST.
    """

    implemented_properties = ['energy', 'forces', 'stress']
    command = None
    conquest_infile = "Conquest_input"
    conquest_outfile = "Conquest_out_ase"

    # A minimal set of defaults
    default_parameters = {
        'grid_cutoff': 100,     # DFT defaults
        'kpts': None,
        'xc': 'PBE',
        'scf_tolerance': 1.0e-6,
        'nspin': 1,
        'general.pseudopotentialtype': 'Hamann',  # CONQUEST defaults
        'basis.basisset': 'PAOs',
        'io.iprint': 2,
        'io.fractionalatomiccoords': True,
        'sc.maxiters': 50,
        'atommove.typeofrun': 'static',
        'dm.solutionmethod': 'diagon'}

    def __init__(self, restart=None, label=None, atoms=None, basis={},
                 **kwargs):

        """Construct CONQUEST-calculator object.
        Species are sorted by atomic number for purposes of indexing.

        Parameters
        ==========

        directory: str
            directory for calculation

        atoms: ASE atoms object
            An atoms object constructed either via ASE or read from an input.
            A CONQUEST-formatted structure file can be parsed into an atoms
            object using the function ase.io.conquest.read_conquest_structure

        kwargs: keyword arguments
            Critical DFT parameters for the calculator. These have defaults,
            but are *strongly* system-dependent.
            "grid_cutoff"  = 150      Integration grid energy cutoff (Ha)
            "kpts"         = [4,4,4]  K-point mesh etc
            "xc"           = "PBE"    XC functional
            "scf_tolerance = 1.0e-7   Self-consistent tolerance
            "nspin"        = 1        spin polarisation

            The rest can be CONQUEST key/value pairs in a dictionary, e.g.
            other_keywords = {"Basis.MultisiteSF"         : True,
                              "Multisite.LFD"             : True,
                              "Mulisite.LFD.ThreshE       : 1.0e-6
                              "Mulisite.LFD.ThreshD       : 1.0e-6
                              "Multistie.LFD.MaxIteration : 150}

           calc = Conquest(grid_cutoff=50,
                           xc="LDA",
                           self_consistent=True,
                           basis=basis,
                           kpts=[4,4,4],
                           nspin=2,
                           **other_keywords)

        basis: dict
            A dictionary specifying the basis set parameters. These will
            generally be parsed from the .ion file, but can be set as follows:
                 - valence_charge            : Float
                 - number_of_supports        : Int
                 - support_fn_range          : Float
                 - pseudopotential_type      : Str (siesta, hamann)
                 - xc functional use for generation : Str (PBE, LDA)
                 (if not given functional prodiced in Calculator is used)

            For example, for NaCl you might have something like:

                basis = {"Na" : {"valence_charge"       : 1.00,
                                 "number_of_supports"   : 4,
                                 "support_fn_range"     : 7.25,
                                 "pseudopotential_type" : "siesta"},
                        "Cl" :  {"valence_charge"       : 7.00,
                                 "number_of_supports"   : 9,
                                 "support_fn_range"     : 5.00,
                                 "pseudopotential_type" : "siesta"}}

            Above is deprecated you have now 3 options which depend upon
            ConquestEnv by setting up CQ_PP_PATH and CQ_GEN_BASIS_CMD

            CQ_PP_PATH is the absolute path towards a library of ion files
            (from the Conquest root tree located at ~/pseudo-and-pao)

            CQ_GEN_BASIS_CMD is the absolute path towards the the MakeIonFiles
            (from the Conquest root tree located at ~/tools/BasisGeneration)

            option 1) specify the ion file names located in CQ_PP_PATH

                basis = {'Na' : {'file' : 'Na_PBE_DZP_CQ.ion'},
                         'Cl' : {'file' : 'Cl_PBE_DZP_CQ.ion'}}

            option 2)

            Alternatively, if paths to a pseudopotential library and the
            MakeIonFiles tool are set, a basis set will be generated with the
            following input:

                basis = {"Na" : {"gen_basis"            : True,
                                 "basis_size"           : "medium",
                                 "pseudopotential_type" : "hamann"},
                        "Cl" :  {"gen_basis"            : True,
                                 "basis_size"           : "medium",
                                 "pseudopotential_type" : "hamann"}}

            option 3) ... mix of 1) and 2)

        Examples
        ========
        Use default values:
        TODO
        """

        self.initialised = False
        self.label = label
        self.eigenvalues = None
        self.occupancies = None
        self.kpoints = None
        self.weights = None
        self.nspin = None
        self.coorfile = None

        FileIOCalculator.__init__(self, restart=restart, label=self.label,
                                  atoms=atoms, **kwargs)

        self.basis = {}
        for spec in basis:
            self.basis[spec] = \
                Parameters({k.lower(): basis[spec][k] for k in basis[spec]})
            print('input basis:')
            print(spec, self.basis[spec])

        self.parameters = Parameters({})

        for key in kwargs:
            self.parameters[key.lower()] = kwargs[key]

        for key in self.default_parameters:
            if key not in self.parameters:
                self.parameters[key] = self.default_parameters[key]

    def init_basis(self, atoms, basis, ion_dir):
        """
        Determine whether to make basis sets useing MakeIonFiles (part of
        CONQUEST distribution). Parse .ion files to write basis information for
        Conquest_input. Make .ion files if necessary.
        """

        from ase.io.conquest import make_ion_files, sort_by_atomic_number

        # Get the species list
        self.species_list = sort_by_atomic_number(atoms)

        # Get XC for ion generation as input XC ; use PBE if not
        for key in self.parameters:
            if (key == 'xc'):
                ion_xc = self.parameters[key]
                break

            else:
                ion_xc = 'PBE'

        gen_basis = []
        for species in self.species_list:
            if ("gen_basis" in basis[species]):

                if (basis[species]["gen_basis"]):
                    gen_basis.append(True)

                else:
                    gen_basis.append(False)

            else:
                basis[species]["gen_basis"] = False
                gen_basis.append(False)

        count = 0
        # Test weither or not if we need to generate the basis
        for test in gen_basis:
            count += 1

            # TODO rewrite make_ion_files to avoid what is below
            basis_single = {k: basis[k] for k in self.species_list}
            species_single_list = [self.species_list[count -1 ]]
            species_single = self.species_list[count - 1]

            # Yes : run make_ion_files
            if (test):
                if ('xc' in basis_single[species_single]):

                    if (ion_xc != basis_single[species_single]['xc']):
                        ConquestWarning("{} xc {} enforced for basis \
                                        generation".format(species_single,
                                        ion_xc))

                        basis_single[species_single]['xc'] = ion_xc

                print('\nmake_ion_files input:')
                print(species_single, basis_single[species_single])
                print(ion_xc)
                make_ion_files(basis_single, species_single_list,
                               directory=self.directory, xc=ion_xc)

            # No : try to find the ion file
            else:
                self.find_ion_file(species_single, basis, ion_dir, ion_xc)

        self.initialised = True

    def find_ion_file(self, species, basis, ion_dir, ion_xc):

        # Default name used for the input
        dname = species + ".ion"

        # Name of the file from basis dictionary
        if ("file" in basis[species]):
            fname = basis[species]["file"]

        # Default otherwise
        else:
            fname = dname

        # Default working directory
        fullpath = Path(self.directory).joinpath(dname)

        if (not ion_dir or ion_dir == '.' or ion_dir == './'
                or ion_dir == '/'):

            ion_dir = Path('')
        else:
            ion_dir = Path(ion_dir)

        if (species not in basis):
            basis[species] = {}

        ion_file_path_lib = ion_dir.joinpath(Path("lib/" + fname))
        ion_file_path = ion_dir.joinpath(Path(fname))

        if ("xc" in basis[species]):
            ion_file_path_xc = ion_dir.joinpath(Path(basis[species]["xc"] 
                                                     + "/" + species + '/' 
                                                     + fname))
        else:
            ion_file_path_xc = ion_dir.joinpath(ion_xc + "/"+species + '/' 
                                                + fname)
        
        count = 0
        for ion_file in [ion_file_path, 
                         ion_file_path_lib, 
                         ion_file_path_xc, 
                         Path(fname), 
                         Path(self.directory).joinpath(fname)]: 

            print("Try to find {} in {}".format(fname,ion_file), end="")
            if(ion_file.is_file() and ion_file != fullpath):
                print("... Found")                
                copy(ion_file, fullpath)
                ConquestWarning("{} copied into {}".format(ion_file,fullpath))
                count += 1
                break

            elif(ion_file.is_file() and ion_file == fullpath):
                print("... Found")    
                ConquestWarning("{} copied into {}".format(ion_file,fullpath))
                count += 1
                break

            else:
                print("... Not found")

        if (count == 0) :
            raise ConquestError("Ion file {} not found".format(fname))

        # else:                                              
            # Else if .ion file exists in the current directory
        #    print("Try to find {} in current directory".format(dname), end="")
        #    if ( Path(dname).is_file() ):
        #        print("... Found")
        #        copy(fname, fullpath)
        #        ConquestWarning("{} from current directory copied into {}".format(dname,fullpath))
        #    else:
        #        print("... Not found")
        #    # Else if .ion file exists in the working directory    
        #    print("Try to find {} in working directory".format(dname), end="")
        #    if ( fullpath.is_file() ): 
        #        print("... Found")
        #        ConquestWarning("{} taken form working directory {}".format(pname,fullpath))
        #        
        #    else:
        #        raise ConquestError("Ion file {} not found".format(pname))
            

    def write_input(self, atoms, properties=None, system_changes=None):
        """
        Create the directory, make the basis if necessary, write the
        Conquest_input and coordinate files
        """

        from ase.io.conquest import write_conquest_input, write_conquest

        # set io.title
        if (not 'io.title' in self.parameters):
            self.parameters['io.title'] = atoms.get_chemical_formula()

        # set coordfile from io.title
        if (not 'io.coordinates' in self.parameters):
            coordfile = self.parameters['io.title'] + '.in'
            self.parameters['io.coordinates'] = coordfile
        else:
            coordfile = self.parameters['io.coordinates']

        # super().write_input(atoms, properties, system_changes)

        if not self.initialised:  # make the basis once only

            cq_env = ConquestEnv()

            try:
                pseudo_dir = cq_env.get('pseudo_path')

            except:
                pseudo_dir = './'                

            self.init_basis(atoms, self.basis, ion_dir=pseudo_dir)

        cq_in = Path(self.directory).joinpath(Path(self.conquest_infile))
        cq_coord = Path(self.directory).joinpath(Path(coordfile))
        
        with cq_in.open(mode='w') as infile_obj:
            
            write_conquest_input(infile_obj, atoms, self.species_list,
                                 self.parameters,
                                 directory=self.directory, basis=self.basis)

        with cq_coord.open(mode='w') as coordfile_obj:
            write_conquest(coordfile_obj, atoms, self.species_list) 

        # it looks label is needed when using some ASE routines, eg. NEB
        if self.label : 
            self.parameters.write(self.label + '.ase')

    def read_results(self, atoms=None):
        """
        Parse energy, Fermi level, forces, stresses from Conquest_out
        """
        
        from ase.io.conquest import read_conquest_out

        if ( atoms != None ):
            self.atoms = atoms

        cq_out_path = Path(self.directory).joinpath(Path(self.conquest_outfile))

        if (not cq_out_path.is_file()):
            raise ReadError

        with cq_out_path.open(mode='r') as outfile_obj:
            # print(outfile_obj)   
            self.results = read_conquest_out(outfile_obj, self.atoms)

        # Eigenvalues are printed only for IO.Iprint > 2, and diagonalisation
        if (self.parameters['dm.solutionmethod'] == 'diagon'):
            if (self.parameters['io.iprint'] >= 2):
                self.get_bands(cq_out_path)

    def get_bands(self, cq_out_path):
        """
        Parser fermi level, eigenvalues, kpoints and weights from Conquest_out.
        Generates np.array(nspin, nkpoints, nbands) for eigenvalues and
        occupancies, np.array(3, nkpts) and np.array(nkpts) for kpoints and
        weights
        """
        from ase.io.conquest import read_bands, get_fermi_level, get_k_points

        with open(cq_out_path, 'r') as cqout_fileobj:
            self.eFermi = get_fermi_level(cqout_fileobj)

        with open(cq_out_path, 'r') as cqout_fileobj:
            self.eigenvalues, self.occupancies = \
                read_bands(self.parameters['nspin'], cqout_fileobj)
                
        with open(cq_out_path, 'r') as cqout_fileobj:
            self.kpoints, self.weights = get_k_points(cqout_fileobj)
    # 
    # Below functions necessary for ASE band_structure()
    #
    def get_ibz_k_points(self):

        return self.kpoints

    def get_k_point_weights(self):
        
        return self.weights

    def get_eigenvalues(self, kpt=0, spin=0):

        return self.eigenvalues[spin][kpt][:] * Hartree

    def get_fermi_level(self):

        return self.eFermi

    def get_number_of_spins(self):

        for key in self.parameters:

            if ( key == 'nspin'):
                self.nspin = self.parameters[key]

        if ( self.nspin is None ):        
                ConquestWarning('nspin not specified in Conquest input')
                self.nspin = 1

        return self.nspin      


