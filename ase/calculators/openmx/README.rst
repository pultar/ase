.. module:: ase.calculators.openmx

======
OpenMX
======

Introduction
============

OpenMX_ (Open source package for Material eXplorer) is a software 
package for nano-scale material simulations based on density functional 
theories (DFT), norm-conserving pseudopotentials, and pseudo-atomic 
localized basis functions. This interface makes it possible to use 
OpenMX_ as a calculator in ASE, and also to use ASE as a post-processor
for an already performed OpenMX_ calculation.

You should import the OpenMX calculator when writing ASE code.
To import into your python code::

  from ase.calculators.openmx import OpenMX

"Openmx" can also be used (more conventional)

Then you can define a calculator object and set it as the calculator of an
atoms object::

  calc = OpenMX(**kwargs)
  atoms.set_calculator(calc)

.. _OpenMX: http://www.openmx-square.org

Environment variables
=====================

The environment variable :envvar:`OPENMX_COMMAND` must point to that file.

A directory containing the pseudopotential directories :file:`VPS`, and it 
is to be put in the environment variable :envvar:`OPENMX_DFT_DATA_PATH`.

Set both environment variables in your shell configuration file:

.. highlight:: bash

::

  $ export OPENMX_DFT_DATA_PATH=/openmx/DFT_DATA13
  $ export OPENMX_COMMAND='mpirun -np 20 openmx ./%s > ./%s'

.. highlight:: python



Keyword Arguments of OpenMX objects
===================================

The default setting used by the OpenMX interface is

.. autoclass:: OpenMX

Below follows a list with a selection of parameters

+-------------------+-----------+---------+---------------------------------------------+
| keyword           | type      | default | description                                 |
+===================+===========+=========+=============================================+
| ``xc``            | ``dict``  | LDA     | Exchange correlation functional.            | 
|                   |           |         | Options are:                                | 
|                   |           |         | 'LDA' -> Local Density Approximation        | 
|                   |           |         | 'LSDA' or 'CA' -> 'LSDA-CA'                 | 
|                   |           |         | -> Local Spin Density Approximation,        |   
|                   |           |         | Ceperley-Alder                              |  
|                   |           |         | 'PW' -> 'LSDA-PW' -> Local Spin             | 
|                   |           |         | Density Approximation, Perdew-Wang          |  
|                   |           |         | 'GGA' or 'PBE' -> 'GGA-PBE' ->              |  
+-------------------+-----------+---------+---------------------------------------------+             
| ``kpts``          | ``int``   | (4,4,4) | GGA proposed by Perdew, Burke, and Ernzerhof|
|                   |           |         | Integers in a tuple specifying the type of  |
|                   |           |         | Monkhorst Pack                              |     
+-------------------+-----------+---------+---------------------------------------------+             
| ``dft_data_dict`` | ``dict``  | {'H':   | A specification of the atomic orbital       | 
|                   |           |         | basis to be used for each atomic species    |
|                   |           |         | e.g)  dft_data_dict =                       |   
|                   |           |         | {'C': {'cutoff radius': 8*ase.units.Bohr,   |   
|                   |           |         | 'orbitals used': [1,1]}}                    | 
|                   |           |         | means that for carbon species,              | 
|                   |           |         | the spatial extent of the atomic orbitals   | 
|                   |           |         | used is limited to 8 Bohr and one level of s|         
|                   |           |         | orbitals are used and one level of p        |
|                   |           |         | orbitals are used. The default value for    |
|                   |           |         | this is specified in the default_settings.py|
|                   |           |         | file, namely, default_dictionary.           |
+-------------------+-----------+---------+---------------------------------------------+             
| initial_magnetic_m| ``float`` | None    | An iterable containing the initial guess for|
| oments            |           |         | magnetic_moments for each atom. A positive  |
|                   |           |         | value indicates a net magnetic moment in the|  
|                   |           |         | spin up direction. If this argument is      |
|                   |           |         | specified, OpenMX will indentify the system | 
|                   |           |         | as spin polarised and will try to find a    | 
|                   |           |         | stable collinear spin configuration for     |
|                   |           |         | the system which is nearest the initial     |
|                   |           |         | guess specified.  You may then specify      | 
|                   |           |         | hubbard_u_values and hubbard_occupation.    | 
+-------------------+-----------+---------+---------------------------------------------+             
| hubbard_u_values  | ``dict``  | None    | A dictionary of dictionaries.               |
|                   |           |         | The first key specifies the species symbol  |
|                   |           |         | (e.g. 'Fe') and the second key specifies    |
|                   |           |         | the orbital (e.g. '1d'). The value is in    |
|                   |           |         | eV. If the value is not provided it assumed | 
|                   |           |         | to be zero.                                 | 
|                   |           |         | e.g. hubbard_u_values={'Fe': {'1d': 4.0}}   | 
|                   |           |         | => Hubbard U values will be 0 eV            | 
|                   |           |         | except for 1d orbitals in iron atoms.       |
+-------------------+-----------+---------+---------------------------------------------+             
| hubbard_occupation| ``str``   |         | A choice of 'dual', 'onsite' or 'full'      |
+-------------------+-----------+---------+---------------------------------------------+             
| initial_magnetic  | ``list``  |         | Initial guess for orientation of each atom's| 
| _moments_euler_   |           |         | Magnetic moment in degrees. If this argument|  
| angles            |           |         | is specified, OpenMX will allow spins       | 
|                   |           |         | between atoms to be non-collinear.          |         
|                   |           |         | e.g. initial_magnetic_moments_euler_angles  | 
|                   |           |         | =[(45, 0), (90, 45)] => First atom          | 
|                   |           |         | has magnetic moment aligned in theta=45     |  
|                   |           |         | degrees and phi=0 degrees direction and the |    
|                   |           |         | second atom has magneticmoment aligned in   |   
|                   |           |         | theta=90 degrees and phi=45 degrees         |
|                   |           |         | direction.                                  |
+-------------------+-----------+---------+---------------------------------------------+
| nc_spin_constrai  |	        |         | Same format as initial_magnetic_moments_eule|
| nt_euler_angle    |	        |         | r_angles. Specify this if you want to constr| 
|                   |           |         | ain the spins to certain axes. You must also|
|                   |           |         | specify spin_euler_angle and either nc_spin |   
|                   |           |         | _constraint_penalty or magnetic_field.      |  
+-------------------+-----------+---------+---------------------------------------------+
| nc_spin_constrai  |	        |         | if nc_spin_constraint_euler_angle is given, |
| nt_penalty        |           |         | you may specify a prefactor (eV) for the    | 
|                   |           |         | penalty functional to be used.              |
+-------------------+-----------+---------+---------------------------------------------+
| magnetic_field    | 	        |         | quote a magnitude of magnetic field strength|
|                   |           |         | (T) in the direction of orbital_euler_angle.|
|                   |           |         | This will include the Zeeman term for orbita| 
|                   |           |         | l magnetic moments in the DFT calculation.  | 
+-------------------+-----------+---------+---------------------------------------------+
| ``smearing`` 	    |           |         | Specifies the variation of electron occupati|  
|                   |           |         | on with respect to the Fermi level. Default |
|                   |           |         | is ('Fermi-Dirac': 300*ase.units.kB).       |
+-------------------+-----------+---------+---------------------------------------------+
| ``scf_max_iter``  |           |         | the maximum number of iterations the self   |
|                   |           |         | consistent field calculation will make befor| 
|                   |           |         | e finishing. Default is 40.                 |
+-------------------+-----------+---------+---------------------------------------------+
| eigenvalue_solver |           |         | 'DC' (for divide-conquer method), 'Krylov'  |
|                   |           |         | (for Krylov subspace method), 'ON2' (for a  |
|                   |           |         | numerically exact low-order scaling method),| 
|                   |           |         | 'Cluster' or 'Band'. If not specified, this |  
|                   |           |         | will be taken as 'Cluster'.                 |   
+-------------------+-----------+---------+---------------------------------------------+
| ``mixing_type``   |           |         | how the electron density is determined for  |  
|                   |           |         | the next self consistent field step. Options|  
|                   |           |         | are 'Simple', 'GR-Pulay' (Guaranteed Reducti|   
|                   |           |         | on), 'RMM-DIIS', 'Kerker', 'RMM-DIISK',     |  
|                   |           |         | 'DMM-DIISH'.                                |
+-------------------+-----------+---------+---------------------------------------------+
| init_mixing_weigh |           | 0.3     |                                             |    
| t                 |           |         |                                             |
+-------------------+-----------+---------+---------------------------------------------+
| min_mixing_weight |           | 0.001   |                                             |  
+-------------------+-----------+---------+---------------------------------------------+
| max_mixing_weight |           | 0.4     |                                             |   
+-------------------+-----------+---------+---------------------------------------------+
| mixing_history    |           | 5       |                                             |   
+-------------------+-----------+---------+---------------------------------------------+
| mixing_start_pula |           | 6       |                                             | 
| y                 |           |         |                                             |
+-------------------+-----------+---------+---------------------------------------------+
| ``scf_criterion`` |           | 0.000001| Hartrees                                    |    
+-------------------+-----------+---------+---------------------------------------------+              

Molecular Dynamics
==================

================= ======== ============== ============================
keyword           type     default value  description
================= ======== ============== ============================
``md_type``       ``str``                 'Opt', 'NVE', 'NVT_VS' or 'NVT_NH'. If not 
                                          specified, no molecular dynamics calculations 
                                          will be performed.
``md_max_iter``	 ``int``   1              1   
``time_step``    ``float`` 0.5            1         
``md_criterion`` ``float`` 0.0001         Hartrees per Bohr
================ ========= ============== ============================

Density of States
=================

=================  ========= ============== ============================
keyword            type      default value  description
=================  ========= ============== ============================
``dos_fileout``    ``str``   False          if True, density of states will be calculated 
                                            for an energy range given by dos_erange. 
``dos_erange`` 	   ``tuple`` (-25, 20)      Gives the density of states energy range in eV
``dos_kgrid`` 	   ``tuple``                defaults to the value given by kpts.
=================  ========= ============== ============================

Band Structure
==============

======================= ========= ============== ============================
keyword                 type      default value  description
======================= ========= ============== ============================
``band_dispersion``     ``str``   False		 If True, the band structure will be calculated
						 for a path in k-space specified by band_kpath.
``band_kpath_unitcell`` ``float``  		 If given, this specifies the unit cell 
						 (in terms of real space) being used to 
						 calculate the band structure. If not given, 
						 the unit cell given in ASE will be used.
``band_kpath`` 		``float``		 A list of dictionaries giving the properties 
						 of each part of the path in k-space. Each 
						 dictionary should give:
``kpts`` 		``int``			 an integer specifying the number of points 
						 in kspace to calculate energies between the 
						 start and end point.
``start_point``        	``float``  		 i.e. where in k-space relative to the
						 unit cell the part of the path starts from.
``end_point`` 		``float`` 		 i.e. where in k-space relative to the unit
						 cell the part of the path ends.
``path_symbols`` 	``str``			 i.e. the symbol denoting the start 
						 point and the symbol denoting the end point.
======================= ========= ============== ============================

File Management
===============

=========== ======= ============== ============================
keyword     type    default value  description
=========== ======= ============== ============================
``curdir``  ``str`` ./             the current directory of the system. 
``fileout`` ``int``  1             the level of file output. 
``stdout``  ``int``  1             the level of standard output. 
=========== ======= ============== ============================

Molecular Orbitals
==================

========================== ======== ============= ============================
keyword                    type     default value description
========================== ======== ============= ============================
``homos``                                         the number of highest energy occupied 
                                                  molecular orbitals to calculate.
``lumos``                                         the number of lowest energy unoccupied 
                                                  molecular orbitals to calculate.
``mo_kpts``                                       the points in k-space to find HOMOs and LUMOs.
``absolute_path_of_vesta``                        the absolute file path of the system's VESTA
                                                  executable. This is required to produced a 
                                                  graphical output of HOMOs and LUMOs.
========================== ======== ============= ============================

Methods of OpenMX objects 
=========================

get_dos(***kwargs):
key word arguments:

====================  =========  ============== ============================
keyword               type       default value  description
====================  =========  ============== ============================
``energy``         float                        The total energy of the system in eV. 
``forces``                                      An array of tuples describing the forces on an each atom in eV / Ang.
                                                e.g. array([(atom1Fx, atom1Fy, atom1Fz), (atom2Fx, atom2Fy, atom2Fz)]
                                                'dipole': A tuple describing the total dipole moment in Debeye
                                                'chemical_potential': The chemical potential of the system in eV



``atoms``                                       Needs to be specified if system hasn't been calculated with the
                                                parameter, dos_fileout=True.

``erange``                                      e.g. (min_energy, max_energy) with the energy quoted in eV. If not
                                                specified, this will be the same as the dos_erange parameter of the calculator.

``method``                                      'Tetrahedron' or 'Gaussian'. The method of calculating the density of
                                                states from the eigenvalues and eigenvectors.
``gaussian_width``                              If method='Gaussian', then the width of broadening needs to be
                                                specified in eV. The default is 0.1eV.
``spin_polarization``                           If True, each graph plotted will split horizontally with
                                                spin up above the x-axis and spin down below the x-axis. If not specified, this
                                                will be True for spin polarized systems and False for spin non-polarized
                                                systems. You may specify this as False for a spin_polarized system.
``density``                                     If True, the (partial) density of states will be plotted. The default
is True.

cum                                             If True, the cumulative number of states from the minimum energy specified
                                                in the dos_erange parameter will be plotted. The default is False.
``fermi_level``                                 If True, the region of the graph below the fermi level will be
                                                highlighted in yellow. The default is True.
``file_format``                                 If specified, instead of opening a window to the view the plot,
                                                the plot will be saved in a specified format. The following formats are
                                                available: 'pdf', 'png', 'ps', 'eps' or 'svg'.
``pdos``                                        If True, the partial density of states will be calculated and plotted for
                                                the atoms specified in atom_index_list and their orbitals specified by
                                                orbital_list. If False, the total density of states for the whole system will
                                                be calculated and plotted.
``atom_index_list``                             if pdos=True, a list of reference numbers of the atoms to have
                                                their partial density of states calculated and plotted. If not specified, only
                                                the first atom will be used.
``orbital_list``                                if pdos=True, a list of all the orbitals to have their partial
                                                density of states plotted. If '' is in the list, the combined partial density
                                                of states for each desired atom will be plotted. If 's', 'p', 'd' or 'f' is in
                                                the list then all the corresponding orbitals of that type will be plotted. If
                                                the list is not specified, then only the combined density of states will be
                                                plotted.

====================  =========  ============== ============================

get_band(***kwargs):
key-word arguments:

===============  =========  ============== ============================
keyword          type       default value  description
===============  =========  ============== ============================
``erange``                                 e.g. (min_energy, max_energy) with the energy quoted in eV. If not
                                           specified, this defaults to (-10, 10).
``plot``                                   which kind of plot that will be used. Either 'pyplot' (matplotlib) or
                                           'gnuplot'. Default is 'pyplot'.
``atoms``                                  If the calculator has not produced a .Band file already, an atoms object
                                           is required to run the calculation.
``spin``                                   if plot='gnuplot' and spin is 'up' or 'down' then just the specified spin
                                           states will be plotted.
``fermi_level``                            If True, the region of the graph below the fermi level will be
                                           highlighted in yellow. The default is True.
``file_format``                            If specified, instead of opening a window to the view the plot,
                                           the plot will be saved in a specified format. The following formats are
                                           available: 'pdf', 'png', 'ps', 'eps' or 'svg'.
===============  =========  ============== ============================

get_mo(***kwargs):
key-word arguments:

===============  =========  ============== ============================
keyword          type       default value  description
===============  =========  ============== ============================
``homos``                                  a list of HOMO numbers to display. e.g. homos=[0, 1, 5] => HOMO, HOMO-1,
                                           HOMO-5 will be displayed. Defaults to displaying all calculated HOMOs.
``lumos``                                  same as homos but for LUMOs.
``real``                                   If True the real component of the wavefunctions will be displayed.
                                           Defaults to True.
``imaginary``                              If True the imaginary component of the wavefunction will be
                                           displayed. Defaults to False.
``spins``                                  if system is spin polarised you can choose the spins to display in a
                                           list. If 'up' or 'down' is in the list then those spins will be displayed.
                                           Defaults to showing both spins for spin-polarised cases, all just combined
                                           spin states otherwise.
==============  =========  ==============  ============================

