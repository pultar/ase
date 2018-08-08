"""
ASE Calculator for interatomic models compatible with the Knowledgebase of
Interatomic Models (KIM) application programming interface (API). Written by:

Mingjian Wen
University of Minnesota
"""
from __future__ import absolute_import, division, print_function
import numpy as np
from .exceptions import KIMCalculatorError
from ase.calculators.calculator import Calculator
from ase import Atom
from ase.neighborlist import neighbor_list
try:
    import kimpy
except ModuleNotFoundError:
    print('kimpy not found; KIM calculator will not work')


__version__ = '0.1.0'
__author__ = 'Mingjian Wen'


class KIMModelCalculator(Calculator, object):
    """ An ASE calculator to work with KIM interatomic models.

    Parameter
    ---------

    modelname: str
      KIM model name

    neigh_skin_ratio: double
      The neighbor list is build using r_neigh = (1+neigh_skin_ratio)*rcut.

    debug: bool
      Flag to indicate whether to enable debug mode to print extra information.
    """

    implemented_properties = ['energy', 'forces', 'stress']

    def __init__(self, modelname, neigh_skin_ratio=0.2,
                 debug=False, *args, **kwargs):
        super(KIMModelCalculator, self).__init__(*args, **kwargs)

        self.modelname = modelname
        self.debug = debug

        # neigh attributes
        if neigh_skin_ratio < 0:
            neigh_skin_ratio = 0
        self.neigh_skin_ratio = neigh_skin_ratio
        self.neigh = None
        self.skin = None
        self.cutoff = None
        self.last_update_positions = None

        # padding atoms related
        self.padding_need_neigh = None
        self.num_contributing_atoms = None
        self.num_padding_particles = None
        self.neighbor_image_of = None

        # model and compute arguments objects
        self.kim_model = None
        self.compute_arguments = None

        # model input
        self.num_particles = None
        self.species_code = None
        self.contributing_mask = None
        self.coords = None

        # model output
        self.energy = None
        self.forces = None

        # initialization flags
        self.kim_initialized = False
        self.neigh_initialized = False

        # initialize KIM
        self.init_kim()

    def init_kim(self):
        """Initialize KIM.
        """

        if self.kim_initialized:
            return

        # create model
        units_accepted, kim_model, error = kimpy.model.create(
            kimpy.numbering.zeroBased,
            kimpy.length_unit.A,
            kimpy.energy_unit.eV,
            kimpy.charge_unit.e,
            kimpy.temperature_unit.K,
            kimpy.time_unit.ps,
            self.modelname
        )
        check_error(error, 'kimpy.model.create')
        if not units_accepted:
            report_error('requested units not accepted in kimpy.model.create')
        self.kim_model = kim_model

        # units
        if self.debug:
            l_unit, e_unit, c_unit, te_unit, ti_unit = kim_model.get_units()
            check_error(error, 'kim_model.get_units')
            print('Length unit is:', str(l_unit))
            print('Energy unit is:', str(e_unit))
            print('Charge unit is:', str(c_unit))
            print('Temperature unit is:', str(te_unit))
            print('Time unit is:', str(ti_unit))
            print()

        # create compute arguments
        self.compute_arguments, error = kim_model.compute_arguments_create()
        check_error(error, 'kim_model.compute_arguments_create')

        # check compute arguments
        num_compute_arguments = kimpy.compute_argument_name.get_number_of_compute_argument_names()
        if self.debug:
            print('Number of compute_arguments:', num_compute_arguments)

        for i in range(num_compute_arguments):
            name, error = kimpy.compute_argument_name.get_compute_argument_name(
                i)
            check_error(
                error, 'kimpy.compute_argument_name.get_compute_argument_name')

            dtype, error = kimpy.compute_argument_name.get_compute_argument_data_type(
                name)
            check_error(
                error, 'kimpy.compute_argument_name.get_compute_argument_data_type')

            support_status, error = self.compute_arguments.get_argument_support_status(
                name)
            check_error(error, 'compute_arguments.get_argument_support_status')

            if self.debug:
                n_space_1 = 21 - len(str(name))
                n_space_2 = 7 - len(str(dtype))
                print('Compute Argument name "{}" '.format(name) + ' ' * n_space_1 +
                      'is of type "{}" '.format(dtype) + ' ' * n_space_2 +
                      'and has support status "{}".'.format(support_status))

            # the simulator can handle energy and force from a kim model
            # virial is computed within the calculator
            if support_status == kimpy.support_status.required:
                if (name != kimpy.compute_argument_name.partialEnergy and
                        name != kimpy.compute_argument_name.partialForces
                    ):
                    report_error(
                        'Unsupported required ComputeArgument {}'.format(name))

        # check compute callbacks
        num_callbacks = kimpy.compute_callback_name.get_number_of_compute_callback_names()
        if self.debug:
            print()
            print('Number of callbacks:', num_callbacks)

        for i in range(num_callbacks):
            name, error = kimpy.compute_callback_name.get_compute_callback_name(
                i)
            check_error(
                error, 'kimpy.compute_callback_name.get_compute_callback_name')

            support_status, error = self.compute_arguments.get_callback_support_status(
                name)
            check_error(error, 'compute_arguments.get_callback_support_status')

            if self.debug:
                n_space = 18 - len(str(name))
                print('Compute callback "{}"'.format(name) + ' ' * n_space +
                      'has support status "{}".'.format(support_status))

            # cannot handle any "required" callbacks
            if support_status == kimpy.support_status.required:
                report_error(
                    'Unsupported required ComputeCallback: {}'.format(name))

        # set cutoff
        model_influence_dist = kim_model.get_influence_distance()
        self.skin = self.neigh_skin_ratio * model_influence_dist
        self.cutoff = (1 + self.neigh_skin_ratio) * model_influence_dist

        # TODO support multiple cutoffs
        model_cutoffs, padding_hints, half_hints = kim_model.get_neighbor_list_cutoffs_and_hints()
        self.cutoffs = model_cutoffs

        if padding_hints[0] == 0:
            self.padding_need_neigh = True
        else:
            self.padding_need_neigh = False

        if self.debug:
            print()
            print('Model influence distance:', model_influence_dist)
            print('Number of cutoffs:', model_cutoffs.size)
            print('Model cutoffs:', model_cutoffs)
            print('Model padding neighbors hints:', padding_hints)
            print('Model half list hints:', half_hints)
            print('Calculator cutoff (include skin):', self.cutoff)
            print('Calculator cutoff skin:', self.skin)
            print()

        self.kim_initialized = True

    def set_atoms(self, atoms):
        """Initialize KIM and neighbor list.
        This is called by set_calculator() of Atoms instance.

        Note that set_calculator() may be called multiple times by different Atoms instance.

        Parameter
        ---------

        atoms: ASE Atoms instance
        """
        self.init_neigh(atoms)

    def init_neigh(self, atoms):
        """Initialize neighbor list.

        Parameter
        ---------

        atoms: ASE Atoms instance
        """
        # register get neigh callback
        neigh = {}
        self.neigh = neigh
        error = self.compute_arguments.set_callback(
            kimpy.compute_callback_name.GetNeighborList,
            get_neigh,
            neigh)

        check_error(error, 'compute_arguments.set_callback_pointer')

        self.neigh_initialized = True

    def update_neigh(self, atoms):
        """Create the neighbor list along with the other required parameters.

        KIM requires a neighbor list that has indices corresponding to
        positions.
        We first build a neighbor list and get the distance vectors from
        original positions to neighbors, these positions are concatenated to
        the original positions.
        """
        # Information of original atoms
        contributing_coords = atoms.get_positions()
        num_contributing = len(atoms)
        self.num_contributing_atoms = num_contributing

        # species support and code
        contributing_species = atoms.get_chemical_symbols()
        unique_species = list(set(contributing_species))
        species_map = dict()
        for s in unique_species:
            support, code, error = self.kim_model.get_species_support_and_code(
                kimpy.species_name.SpeciesName(s))

            check_error(error or not support,
                        'kim_model.get_species_support_and_code')
            species_map[s] = code
            if self.debug:
                msg = 'Species {} is supported and its code is: {}'
                print(msg.format(s, code))
        contributing_species_code = np.array(
            [species_map[s] for s in contributing_species], dtype=np.intc)

        # Change cutoff to influence distance
        i, j, D, S, dists = neighbor_list('ijDSd', atoms, self.cutoff)

        # Get coordinates for all neighbors (this has overlapping positions)
        A = contributing_coords[i] + D

        # Make the neighbor list ready for KIM
        ac = atoms.copy()
        neighbor_species_code = []
        neighbor_image_of = []
        used = dict()
        neb_dict = dict((k, []) for k in range(num_contributing))
        neb_dists = dict((k, []) for k in range(num_contributing))
        for k in range(len(i)):
            shift_tuple = tuple(S[k])
            if shift_tuple == (0, 0, 0):
                # In unit cell
                neb_dict[i[k]].append(j[k])
                neb_dists[i[k]].append(dists[k])
            else:
                # Not in unit cell
                t = (j[k], ) + shift_tuple
                if t not in used:
                    # Add the neighbor as a padding atom
                    used[t] = len(ac)
                    neb_sym = contributing_species[j[k]]
                    pc = contributing_species_code[j[k]]
                    neighbor_species_code.append(pc)
                    ac.append(Atom(neb_sym, position=A[k]))
                    neighbor_image_of.append(j[k])
                neb_dict[i[k]].append(used[t])
                neb_dists[i[k]].append(dists[k])
        neb_lists = []
        for cut in self.cutoffs:
            neb_list = [np.array(neb_dict[k],
                                 dtype=np.intc)[neb_dists[k] <= cut]
                        for k in range(num_contributing)]
            neb_lists.append(neb_list)

        self.neighbor_image_of = np.array(neighbor_image_of, dtype=np.intc)

        tmp = np.concatenate(
            (contributing_species_code, neighbor_species_code))
        self.species_code = np.asarray(tmp, dtype=np.intc)

        # Save the number of atoms and all their neighbors and positions
        num_padding = len(used)
        self.num_particles = np.array([num_contributing + num_padding],
                                      dtype=np.intc)
        self.coords = ac.get_positions()

        # Save which coordinates are from original atoms and which are from
        # neighbors using a mask
        indices_mask = [1] * num_contributing + [0] * num_padding
        self.contributing_mask = np.array(indices_mask, dtype=np.intc)

        # neb_list now only contains neighbor information for the original
        # atoms. A neighbor is represented as an index in the list of all
        # coordinates in self.coords
        self.neigh['neighbors'] = neb_lists
        self.neigh['cutoff'] = self.cutoff
        self.neigh['num_particles'] = num_contributing

        # Does not support padding needing neighbors at the moment
        # Check the output of padding_need_neigh
        # need_neigh = np.array(indices_mask, dtype=np.intc)
        # if not self.padding_need_neigh:
        #     need_neigh[num_contributing:] = 0

        if self.debug:
            print('Debug: called update_neigh')
            print()

    def update_kim(self):
        """ Register model input and output data pointers.
        """

        # model output
        self.energy = np.array([0.], dtype=np.double)
        self.forces = np.zeros([self.num_particles[0], 3], dtype=np.double)

        # register argument
        error = self.compute_arguments.set_argument_pointer(
            kimpy.compute_argument_name.numberOfParticles, self.num_particles)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_arguments.set_argument_pointer(
            kimpy.compute_argument_name.particleSpeciesCodes, self.species_code)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_arguments.set_argument_pointer(
            kimpy.compute_argument_name.particleContributing, self.contributing_mask)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_arguments.set_argument_pointer(
            kimpy.compute_argument_name.coordinates, self.coords)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_arguments.set_argument_pointer(
            kimpy.compute_argument_name.partialEnergy, self.energy)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        error = self.compute_arguments.set_argument_pointer(
            kimpy.compute_argument_name.partialForces, self.forces)
        check_error(error, 'kimpy.compute_argument_name.set_argument_pointer')

        if self.debug:
            print('Debug: called update_kim')
            print()

    def update_kim_coords(self, atoms):
        """Update the atom positions in self.coords, which is registered in KIM.

        Parameter
        ---------

        atoms: ASE Atoms instance
        """
        if self.neighbor_image_of.size != 0:
            # displacement of contributing atoms
            disp_contrib = atoms.positions - self.last_update_positions
            # displacement of padding atoms
            disp_pad = disp_contrib[self.neighbor_image_of]
            # displacement of all atoms
            disp = np.concatenate((disp_contrib, disp_pad))
            # update coords in KIM
            self.coords += disp
        else:
            np.copyto(self.coords, atoms.positions)

        if self.debug:
            print('Debug: called update_kim_coords')
            print()

    def calculate(self, atoms=None,
                  properties=['energy', 'forces', 'stress'],
                  system_changes=['positions', 'numbers', 'cell', 'pbc']):
        """
        Inherited method from the ase Calculator class that is called by get_property().

        Parameters
        ----------

        atoms: ASE Atoms instance

        properties: list of str
          List of what needs to be calculated.  Can be any combination
          of 'energy', 'forces' and 'stress'.

        system_changes: list of str
          List of what has changed since last calculation.  Can be
          any combination of these six: 'positions', 'numbers', 'cell',
          and 'pbc'.
        """

        Calculator.calculate(self, atoms, properties, system_changes)

        need_update_neigh = True
        if len(system_changes) == 1 and 'positions' in system_changes:  # only pos changes
            if self.last_update_positions is not None:
                a = self.last_update_positions
                b = atoms.positions
                if a.shape == b.shape:
                    delta = np.linalg.norm(a - b, axis=1)
                    # indices of the two largest element
                    ind = np.argpartition(delta, -2)[-2:]
                    if sum(delta[ind]) <= self.skin:
                        need_update_neigh = False

        # update KIM API input data and neighbor list if necessary
        if system_changes:
            if need_update_neigh:
                self.update_neigh(atoms)
                self.update_kim()
                self.last_update_positions = atoms.get_positions()  # should make a copy
            else:
                self.update_kim_coords(atoms)

            release_GIL = False
            if 'GIL' in atoms.info:
                if atoms.info['GIL'] == 'off':
                    release_GIL = True
            error = self.kim_model.compute(self.compute_arguments, release_GIL)
            check_error(error, 'kim_model.compute')

        energy = self.energy[0]
        forces = assemble_padding_forces(self.forces,
                                         self.num_contributing_atoms,
                                         self.neighbor_image_of)
        stress = compute_virial_stress(self.forces, self.coords)

        # return values
        self.results['energy'] = energy
        self.results['forces'] = forces
        self.results['stress'] = stress

    def get_kim_model_supported_species(self):
        """Get all the supported species by a model.

        Return: list of str
          a list of species
        """
        species = []
        num_kim_species = kimpy.species_name.get_number_of_species_names()

        for i in range(num_kim_species):
            species_name, error = kimpy.species_name.get_species_name(i)
            check_error(error, 'kimpy.species_name.get_species_name')
            species_support, code, error = self.kim_model.get_species_support_and_code(
                species_name)
            check_error(error, 'kim_model.get_species_support_and_code')
            if species_support:
                species.append(str(species_name))

        return species

    def __expr__(self):
        """Print this object shows the following message."""
        return 'KIMModelCalculator(modelname = {})'.format(self.modelname)

    def __del__(self):
        """Garbage collection for the KIM model object and neighbor list object."""

        # free neighbor list
        if self.neigh_initialized:
            self.neigh = {}

        # free compute arguments
        if self.kim_initialized:
            error = self.kim_model.compute_arguments_destroy(
                self.compute_arguments)
            check_error(error, 'kim_model.compute_arguments_destroy')

            # free kim model
            kimpy.model.destroy(self.kim_model)


def assemble_padding_forces(forces, num_contributing, neighbor_image_of):
    """
    Assemble forces on padding atoms back to contributing atoms.

    Parameters
    ----------

    forces: 2D array
      forces on both contributing and padding atoms

    num_contributing: int
      number of contributing atoms

    neighbor_image_of: 1D int array
      atom number, of which the padding atom is an image


    Returns
    -------
      Total forces on contributing atoms.
    """
    total_forces = np.array(forces[:num_contributing])

    has_padding = True if neighbor_image_of.size != 0 else False

    if has_padding:

        pad_forces = forces[num_contributing:]
        for f, org_index in zip(pad_forces, neighbor_image_of):
            total_forces[org_index] += f

    return total_forces


def compute_virial_stress(forces, coords):
    """Compute the virial stress in voigt notation.

    Parameters
    ----------
      forces: 2D array
        forces on all atoms (padding included)

      coords: 2D array
        coordinates of all atoms (padding included)

    Returns
    -------
      stress: 1D array
        stress in Voigt order (xx, yy, zz, yz, xz, xy)
    """
    stress = np.zeros(6)
    stress[0] = -np.dot(forces[:, 0], coords[:, 0])
    stress[1] = -np.dot(forces[:, 1], coords[:, 1])
    stress[2] = -np.dot(forces[:, 2], coords[:, 2])
    stress[3] = -np.dot(forces[:, 1], coords[:, 2])
    stress[4] = -np.dot(forces[:, 0], coords[:, 2])
    stress[5] = -np.dot(forces[:, 0], coords[:, 1])

    return stress


def check_error(error, msg):
    if error != 0 and error is not None:
        raise KIMCalculatorError('Calling "{}" failed.\n'.format(msg))


def report_error(msg):
    raise KIMCalculatorError(msg)


def get_neigh(data, cutoffs, neighbor_list_index, particle_number):
    try:
        # we only support one neighbor list
        rcut = data['cutoff']
        if len(cutoffs) != 1 or cutoffs[0] > rcut:
            return(np.array([]), 1)
        if neighbor_list_index != 0:
            return(np.array([]), 1)

        # invalid id
        number_of_particles = data['num_particles']
        if particle_number >= number_of_particles or particle_number < 0:
            return(np.array([]), 1)

        neighbors = data['neighbors'][neighbor_list_index][particle_number]
        return (neighbors, 0)

    except:
        return(np.array([]), 1)
