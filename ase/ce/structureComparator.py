from __future__ import print_function
import sys
from ase.build import tools as asetools
import unittest
import numpy as np
from ase.build import bulk
import copy
from ase.visualize import view
from ase import Atoms
from ase.io import read, write
import time
from matplotlib import pyplot as plt
import itertools
from ase.spacegroup import spacegroup
from ase.lattice import cubic,tetragonal,orthorhombic,monoclinic,triclinic,hexagonal
import time
from scipy.spatial import cKDTree as KDTree
import pickle

try:
    # The code runs perfectly fine without pymatgen
    # PyMatGen is imported just for debugging to verify
    # that the two codes returns the same
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms_to_structure = AseAtomsAdaptor.get_structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    has_pymat_gen = True
except:
    has_pymat_gen = False


class StructureComparator( object ):
    def __init__( self, angleTolDeg=1, position_tolerance=1E-2 ):
        self.s1 = None
        self.s2 = None
        self.angleTolDeg = 1
        self.position_tolerance = position_tolerance

    def niggli_reduce( self ):
        """
        Reduce the two cells to the
        """
        asetools.niggli_reduce(self.s1)
        asetools.niggli_reduce(self.s2)
        self.standarize_cell(self.s1)
        self.standarize_cell(self.s2)

    def standarize_cell( self, atoms ):
        """
        Rotate the first vector such that it points along the x-axis
        """
        cell = atoms.get_cell().T
        total_rot_mat = np.eye(3)
        v1 = cell[:,0]
        l1 = np.sqrt(v1[0]**2+v1[2]**2)
        angle = np.abs( np.arcsin(v1[2]/l1) )
        if ( v1[0] < 0.0 and v1[2] > 0.0 ):
            angle = np.pi-angle
        elif ( v1[0] < 0.0  and v1[2] < 0.0 ):
            angle = np.pi+angle
        elif ( v1[0]>0.0 and v1[2] < 0.0 ):
            angle = -angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        rotmat = np.array( [[ca,0.0,sa],[0.0,1.0,0.0],[-sa,0.0,ca]] )
        total_rot_mat = rotmat.dot(total_rot_mat)
        cell = rotmat.dot(cell)

        v1 = cell[:,0]
        l1 = np.sqrt(v1[0]**2+v1[1]**2)
        angle = np.abs( np.abs( np.arcsin(v1[1]/l1) ) )
        if ( v1[0] < 0.0 and v1[1] > 0.0 ):
            angle = np.pi-angle
        elif ( v1[0] < 0.0 and v1[1] < 0.0 ):
            angle = np.pi+angle
        elif ( v1[0] > 0.0 and v1[1] < 0.0 ):
            angle = -angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        rotmat = np.array( [[ca,sa,0.0],[-sa,ca,0.0],[0.0,0.0,1.0]] )
        total_rot_mat = rotmat.dot(total_rot_mat)
        cell = rotmat.dot(cell)

        # Rotate around x axis such that the second vector is in the xy plane
        v2 = cell[:,1]
        l2 = np.sqrt( v2[1]**2 + v2[2]**2 )
        angle = np.abs( np.arcsin( v2[2]/l2 ) )
        if ( v2[1] < 0.0 and v2[2] > 0.0 ):
            angle = np.pi-angle
        elif ( v2[1] < 0.0 and v2[2] < 0.0 ):
            angle = np.pi+angle
        elif ( v2[1] > 0.0 and v2[2] < 0.0 ):
            angle = -angle
        ca = np.cos(angle)
        sa = np.sin(angle)
        rotmat = np.array( [[1.0,0.0,0.0],[0.0,ca,sa],[0.0,-sa,ca]] )
        total_rot_mat = rotmat.dot(total_rot_mat)
        cell = rotmat.dot(cell)

        atoms.set_cell( cell.T )
        atoms.set_positions( total_rot_mat.dot(atoms.get_positions().T).T )
        atoms.wrap( pbc=[1,1,1] )
        return atoms

    def get_element_count( self ):
        """
        Counts the number of elements in each of the structures
        """
        elem1 = {}
        elem2 = {}

        for atom in self.s1:
            if ( atom.symbol in elem1.keys() ):
                elem1[atom.symbol] += 1
            else:
                elem1[atom.symbol] = 1

        for atom in self.s2:
            if ( atom.symbol in elem2.keys() ):
                elem2[atom.symbol] += 1
            else:
                elem2[atom.symbol] = 1
        return elem1, elem2


    def has_same_elements( self ):
        """
        Check that the element types and the number of each constituent are the same
        """
        elem1, elem2 = self.get_element_count()
        return (elem1 == elem2)

    def has_same_angles( self ):
        """
        Check that the Niggli unit vectors has the same internal angles
        """
        angles1 = []
        angles2 = []
        cell1 = self.s1.get_cell().T
        cell2 = self.s2.get_cell().T

        # Normalize each vector
        for i in range(3):
            cell1[:,i] /= np.sqrt( np.sum(cell1[:,i]**2) )
            cell2[:,i] /= np.sqrt( np.sum(cell2[:,i]**2) )
        dot1 = cell1.T.dot(cell1)
        dot2 = cell2.T.dot(cell2)

        # Extract only the relevant dot products
        dot1 = [ dot1[0,1],dot1[0,2],dot1[1,2] ]
        dot2 = [ dot2[0,1],dot2[0,2],dot2[1,2] ]

        # Convert to angles
        angles1 = [ np.arccos(scalar_prod)*180.0/np.pi for scalar_prod in dot1]
        angles2 = [ np.arccos(scalar_prod)*180.0/np.pi for scalar_prod in dot2]

        for i in range(3):
            closestIndex = np.argmin( np.abs(np.array(angles2)-angles1[i]) )
            if ( np.abs(angles2[closestIndex]-angles1[i]) < self.angleTolDeg ):
                # Remove the entry that matched
                #del dot2[closestIndex]
                del angles2[closestIndex]
            else:
                return False
        return True

    def has_same_volume( self ):
        return np.abs( np.linalg.det(self.s1.get_cell())-np.linalg.det(self.s2.get_cell()) < 1E-5 )

    def compare( self, s1, s2 ):
        """
        Compare the two structures
        """
        self.s1 = s1
        self.s2 = s2

        if ( len(s1) != len(s2) ):
            return False

        if ( not self.has_same_elements() ):
            return False

        self.niggli_reduce()
        if ( not self.has_same_angles() ):
            return False

        if ( not self.has_same_volume() ):
            return False

        matrices = self.get_rotation_reflection_matrices()
        if ( not self.positions_match(matrices, self.s1, self.s2) ):
            return False
        return True

    def get_least_frequent_element( self ):
        """
        Returns the symbol of the least frequent element
        """
        elem1, elem2 = self.get_element_count()
        assert( elem1 == elem2 )
        minimum_value = 2*len(self.s1) # Set the value to a large value
        least_freq_element = "X"
        for key, value in elem1.iteritems():
            if ( value < minimum_value ):
                least_freq_element = key
                minimum_value = value

        if ( least_freq_element == "X" ):
            raise ValueError( "Did not manage to find the least frequent element" )
        return least_freq_element

    def extract_positions_of_least_frequent_element( self ):
        """
        Extracts a dictionary of positions of each element
        """
        elem_pos1 = {}
        elem_pos2 = {}
        pos1 = self.s1.get_positions(wrap=True)
        pos2 = self.s2.get_positions(wrap=True)

        least_freq_element = self.get_least_frequent_element()

        position1 = []
        position2 = []
        atoms1 = None
        atoms2 = None

        for i in range( len(self.s1) ):
            symbol = self.s1[i].symbol
            if ( symbol == least_freq_element ):
                #position1.append( pos1[i,:] )
                if ( atoms1 is None ):
                    atoms1 = Atoms( symbol, positions=[self.s1.get_positions(wrap=True)[i,:]])
                else:
                    atoms1.extend( Atoms(symbol, positions=[self.s1.get_positions(wrap=True)[i,:]]) )

            symbol = self.s2[i].symbol
            if ( symbol == least_freq_element ):
                #position2.append( pos2[i,:] )
                if ( atoms2 is None ):
                    atoms2 = Atoms( symbol, positions=[self.s2.get_positions(wrap=True)[i,:]])
                else:
                    atoms2.extend( Atoms(symbol, positions=[self.s2.get_positions(wrap=True)[i,:]]) )
        atoms1.set_cell( self.s1.get_cell() )
        atoms2.set_cell( self.s2.get_cell() )
        return atoms1, atoms2

    def positions_match( self, rotation_reflection_matrices, atoms1, atoms2 ):
        """
        Check if the position and elements match.
        Note that this function changes self.s1 and self.s2 to the rotation and
        translation that matches best. Hence, it is crucial that this function
        is called before the element comparison
        """
        # Position matching not implemented yet
        pos1_ref = atoms1.get_positions( wrap=True )
        pos2_ref = atoms2.get_positions( wrap=True )

        cell = atoms1.get_cell().T
        delta = 1E-6*(cell[:,0]+cell[:,1]+cell[:,2])
        orig_cell = atoms1.get_cell()

        # Expand the reference object
        exp2, app2 = self.expand(atoms2)

        # Build a KD tree to enable fast look-up of nearest neighbours
        tree = KDTree(exp2.get_positions())

        for matrix in rotation_reflection_matrices:
            pos1 = copy.deepcopy(pos1_ref)
            pos2 = copy.deepcopy(pos2_ref)

            # Translate
            pos1 -= matrix[:3,3]

            # Rotate
            pos1 = matrix[:3,:3].dot(pos1.T).T

            # Update the atoms positions
            atoms1.set_positions( pos1 )
            atoms1.wrap( pbc=[1,1,1] )

            if ( self.elements_match(atoms1,exp2,tree) ):
                return True
        return False

    def expand( self, ref_atoms ):
        """
        This functions adds additional atoms to for atoms close to the cell boundaries
        ensuring that atoms having crossed the cell boundaries due to numerical noise
        are properly detected
        """
        expaned_atoms = copy.deepcopy(ref_atoms)

        cell = ref_atoms.get_cell().T
        normal_vectors = [np.cross(cell[:,0],cell[:,1]), np.cross(cell[:,0],cell[:,2]), np.cross(cell[:,1],cell[:,2])]
        normal_vectors = [vec/np.sqrt(np.sum(vec**2)) for vec in normal_vectors]
        positions = ref_atoms.get_positions(wrap=True)
        tol = 0.0001
        num_faces_close = 0

        appended_atom_pairs = []

        for i in range( len(ref_atoms) ):
            surface_close = [False,False,False,False,False,False]
            # Face 1
            distance = np.abs( positions[i,:].dot(normal_vectors[0]) )
            symbol = ref_atoms[i].symbol
            if ( distance < tol ):
                newpos = positions[i,:]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
                surface_close[0] = True

            # Face 2
            distance = np.abs( (positions[i,:]-cell[:,2]).dot(normal_vectors[0]) )
            if ( distance < tol ):
                newpos = positions[i,:]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
                surface_close[1] = True

            # Face 3
            distance = np.abs( positions[i,:].dot(normal_vectors[1]) )
            if ( distance < tol ):
                newpos = positions[i,:] + cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
                surface_close[2] = True

            # Face 4
            distance = np.abs( (positions[i,:]-cell[:,1]).dot(normal_vectors[1]) )
            if ( distance < tol ):
                newpos = positions[i,:] - cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
                surface_close[3] = True

            # Face 5
            distance = np.abs(positions[i,:].dot(normal_vectors[2]))
            if ( distance < tol ):
                newpos = positions[i,:] + cell[:,0]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
                surface_close[4] = True

            # Face 6
            distance = np.abs( (positions[i,:]-cell[:,0]).dot(normal_vectors[2]) )
            if ( distance < tol ):
                newpos = positions[i,:] - cell[:,0]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
                surface_close[5] = True

            # Take edges into account
            if ( surface_close[0] and surface_close[2] ):
                newpos = positions[i,:] + cell[:,1] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[0] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[0] and surface_close[3] ):
                newpos = positions[i,:] - cell[:,1] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[0] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[3] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] - cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[3] and surface_close[1] ):
                newpos = positions[i,:] - cell[:,1] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[3] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,1] - cell[:,0]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[1] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[2] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0] + cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[2] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] + cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[1] and surface_close[2] ):
                newpos = positions[i,:] + cell[:,1] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            if ( surface_close[1] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)

            # Take corners into account
            if ( surface_close[0] and surface_close[2] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]+cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[0] and surface_close[3] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]-cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[0] and surface_close[2] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]+cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[0] and surface_close[3] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]-cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[3] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]-cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[3] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]-cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[2] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]+cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[2] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]+cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                appended_atom_pairs.append( (i,len(expaned_atoms)) )
                expaned_atoms.extend(newAtom)

        return expaned_atoms, appended_atom_pairs

    def elements_match( self, s1, s2, kdtree ):
        """
        Checks that all the elements in the two atoms match

        NOTE: The unit cells may be in different quadrants
        Hence, try all cyclic permutations of x,y and z
        """

        for order in range(3):
            all_match = True
            used_sites = []
            for i in range( len(s1) ):
                s1pos = np.zeros(3)
                s1pos[0] = s1.get_positions()[i,order]
                s1pos[1] = s1.get_positions()[i, (order+1)%3]
                s1pos[2] = s1.get_positions()[i, (order+2)%3]
                dist, closest = kdtree.query(s1pos)
                if ( (s1[i].symbol != s2[closest].symbol) or (dist > self.position_tolerance)):
                    all_match = False
                    break
            if ( all_match ):
                return True
        return False

    def get_rotation_reflection_matrices( self ):
        """
        Computes the closest rigid body transformation matrix by solving Procrustes problem
        """
        s1_pos_ref = copy.deepcopy( self.s1.get_positions() )
        s2_pos_ref = copy.deepcopy( self.s2.get_positions() )
        atoms1_ref, atoms2_ref = self.extract_positions_of_least_frequent_element()
        rot_reflection_mat = []
        center_of_mass = []
        cell = self.s1.get_cell().T
        angle_tol = self.angleTolDeg*np.pi/180.0

        delta_vec = 1E-6*(cell[:,0]+cell[:,1]+cell[:,2]) # Additional vector that is added to make sure that there always is an atom at the origin

        # Put on of the least frequent elements of structure 2 at the origin
        translation = atoms2_ref.get_positions()[0,:]-delta_vec
        atoms2_ref.set_positions( atoms2_ref.get_positions() - translation )
        atoms2_ref.wrap( pbc=[1,1,1] )
        self.s2.set_positions( self.s2.get_positions()-translation)
        self.s2.wrap( pbc=[1,1,1] )


        # Store three reference vectors
        ref_vec = atoms2_ref.get_cell().T
        ref_vec_lengths = np.sqrt( np.sum( ref_vec**2, axis=0 ) )
        cell_diag = cell[:,0]+cell[:,1]+cell[:,2]

        canditate_trans_mat = []

        # Compute ref vec angles
        angle12_ref = np.arccos( ref_vec[:,0].dot(ref_vec[:,1])/(ref_vec_lengths[0]*ref_vec_lengths[1]) )
        if ( angle12_ref > np.pi/2 ):
            angle12_ref = np.pi-angle12_ref
        angle13_ref = np.arccos( ref_vec[:,0].dot(ref_vec[:,2])/(ref_vec_lengths[0]*ref_vec_lengths[2]))
        if ( angle13_ref > np.pi/2 ):
            angle13_ref = np.pi-angle13_ref
        angle23_ref = np.arccos( ref_vec[:,1].dot(ref_vec[:,2])/(ref_vec_lengths[1]*ref_vec_lengths[2]) )
        if ( angle23_ref > np.pi/2.0 ):
            angle23_ref = np.pi-angle23_ref

        sc_atom_search = atoms1_ref*(3,3,3)
        sc_pos = atoms1_ref.get_positions()+cell[:,0]+cell[:,1]+cell[:,2] # Translate by one cell diagonal
        sc_pos_search = sc_atom_search.get_positions()

        for i in range( len(atoms1_ref) ):
            candidate_vecs = [[],[],[]]
            translation = sc_pos[i,:]-delta_vec

            new_sc_pos = sc_pos_search-translation
            lengths = np.sqrt( np.sum( new_sc_pos**2, axis=1 ) )
            for l in range( len(lengths) ):
                if ( l==i ):
                    continue
                for k in range(3):
                    if ( np.abs(lengths[l]-ref_vec_lengths[k]) < self.position_tolerance ):
                        candidate_vecs[k].append(new_sc_pos[l,:])


            # Check angles
            refined_candidate_list = [[],[],[]]
            for v1 in candidate_vecs[0]:
                for v2 in candidate_vecs[1]:
                    if ( np.allclose(v1,v2, atol=1E-3) ):
                        continue
                    v1len = np.sqrt( np.sum(v1**2) )
                    v2len = np.sqrt( np.sum(v2**2) )
                    angle12 = np.arccos( v1.dot(v2)/(v1len*v2len) )
                    if ( angle12 > np.pi/2.0 ):
                        angle12 = np.pi-angle12
                    for v3 in candidate_vecs[2]:
                        if ( np.allclose(v1,v3, atol=1E-3) or np.allclose(v2,v3, atol=1E-3) ):
                            continue
                        v3len = np.sqrt( np.sum(v3**2) )
                        angle13 = np.arccos( v1.dot(v3)/(v1len*v3len) )
                        if ( angle13 > np.pi/2.0):
                            angle13 = np.pi-angle13

                        angle23 = np.arccos( v2.dot(v3)/(v2len*v3len) )
                        if ( angle23 > np.pi/2.0 ):
                            angle23 = np.pi-angle23

                        if ( np.abs(angle12-angle12_ref) < angle_tol and
                             np.abs(angle13-angle13_ref) < angle_tol and
                             np.abs(angle23-angle23_ref) < angle_tol):
                            refined_candidate_list[0].append(v1)
                            refined_candidate_list[1].append(v2)
                            refined_candidate_list[2].append(v3)

            # Compute rotation/reflection/translation matrices (4x4 matrices)
            for v1,v2,v3 in zip(refined_candidate_list[0],refined_candidate_list[1],refined_candidate_list[2]):
                T = np.zeros((3,3))
                T[:,0] = v1
                T[:,1] = v2
                T[:,2] = v3
                R = ref_vec.dot( np.linalg.inv(T) )

                # Skip the rotation/reflection matrix if it is not unitary
                #if ( not np.allclose(R.dot(R.T),np.eye(3),atol=0.001) ):
                #    continue
                full_matrix = np.zeros((4,4))
                full_matrix[:3,:3] = R
                full_matrix[3,3] = 1
                full_matrix[:3,3] = translation
                canditate_trans_mat.append(full_matrix)
        return canditate_trans_mat

# ============================================================================ #
#                                                                              #
#       BELOW THIS LINE IS ONLY FUNCTIONALITY FOR TESTING THE                  #
#                       STRUCTURE COMPARATOR                                   #
#                                                                              #
# ============================================================================ #
class TestStructureComparator( unittest.TestCase ):
    pymat_code = {
        True:"Equal",
        False:"Different"
    }
    def test_compare( self ):
        s1 = bulk( "Al" )
        s1 = s1*(2,2,2)
        s2 = bulk( "Al" )
        s2 = s2*(2,2,2)

        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

    def test_fcc_bcc( self ):
        s1 = bulk( "Al", crystalstructure="fcc" )
        s2 = bulk( "Al", crystalstructure="bcc", a=4.05 )
        s1 = s1*(2,2,2)
        s2 = s2*(2,2,2)
        comparator = StructureComparator()
        self.assertFalse( comparator.compare(s1,s2) )

    def test_single_impurity( self ):
        s1 = bulk( "Al" )
        s1 = s1*(2,2,2)
        s1[0].symbol = "Mg"
        s2 = bulk( "Al" )
        s2 = s2*(2,2,2)
        s2[3].symbol = "Mg"
        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

    def test_two_impurities( self ):
        s1 = read("test_structures/neigh1.xyz")
        s2 = read("test_structures/neigh2.xyz")
        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )
        s2 = read("test_structures/neigh3.xyz")
        self.assertFalse( comparator.compare(s1,s2) )

    def test_reflection_three_imp(self):
        s1 = read("test_structures/reflection1.xyz")
        s2 = read("test_structures/reflection2.xyz")
        comparator = StructureComparator()

        if ( has_pymat_gen ):
            m = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=5,
                             primitive_cell=True, scale=True)
            str1 = atoms_to_structure(s1)
            str2 = atoms_to_structure(s2)
            print ("PyMatGen says: %s"%(self.pymat_code[m.fit(str1,str2)]))

        self.assertTrue( comparator.compare(s1,s2) )


    def test_translations( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")

        xmax = 2.0*np.max(s1.get_cell().T)
        N = 1
        dx = xmax/N
        pos_ref = s2.get_positions()
        comparator = StructureComparator( position_tolerance=0.01 )
        number_of_correctly_identified = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    displacement = np.array( [ dx*i, dx*j,dx*k ] )
                    new_pos = pos_ref + displacement
                    s2.set_positions(new_pos)
                    if ( comparator.compare(s1,s2) ):
                        number_of_correctly_identified += 1

        msg = "Identified %d of %d as duplicates. All structures are known to be duplicates."%(number_of_correctly_identified,N**3)
        self.assertEqual( number_of_correctly_identified, N**3, msg=msg )

    def test_rot_60_deg( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")
        ca = np.cos(np.pi/3.0)
        sa = np.sin(np.pi/3.0)
        matrix = np.array( [[ca,sa,0.0],[-sa,ca,0.0],[0.0,0.0,1.0]] )
        s2.set_positions( matrix.dot(s2.get_positions().T).T )
        #s2.set_cell( matrix.dot(s2.get_cell().T).T )
        comparator = StructureComparator()
        if ( has_pymat_gen ):
            m = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=5,
                             primitive_cell=True, scale=True)
            str1 = atoms_to_structure(s1)
            str2 = atoms_to_structure(s2)
            code = m.fit(str1,str2)
            print ("PyMatGen says: %s"%(self.pymat_code[code]))

        self.assertTrue( comparator.compare(s1,s2) )

    def test_rot_120_deg(self):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")
        ca = np.cos(2.0*np.pi/3.0)
        sa = np.sin(2.0*np.pi/3.0)
        matrix = np.array( [[ca,sa,0.0],[-sa,ca,0.0],[0.0,0.0,1.0]] )
        s2.set_positions( matrix.dot(s2.get_positions().T).T )
        s2.set_cell( matrix.dot(s2.get_cell().T).T )
        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

    def test_rotations_to_standard( self ):
        s1 = Atoms("Al")
        comparator = StructureComparator()
        for i in range(20):
            cell = np.random.rand(3,3)*4.0-4.0
            s1.set_cell( cell )
            new_cell = comparator.standarize_cell(s1).get_cell().T
            self.assertAlmostEqual( new_cell[1,0], 0.0 )
            self.assertAlmostEqual( new_cell[2,0], 0.0)
            self.assertAlmostEqual( new_cell[2,1], 0.0 )

    def test_point_inversion( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")
        view(s1)
        s2.set_positions( -s2.get_positions() )
        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

    def test_mirror_plane( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")
        comparator = StructureComparator()
        mat = np.array( [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,-1.0]])
        s2.set_positions( mat.dot(s2.get_positions().T).T )
        self.assertTrue( comparator.compare(s1,s2) )

        mat = np.array( [[-1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        s2.set_positions( mat.dot(s1.get_positions().T).T )
        self.assertTrue( comparator.compare(s1,s2))

        mat = np.array( [[1.0,0.0,0.0],[0.0,-1.0,0.0],[0.0,0.0,1.0]])
        s2.set_positions( mat.dot(s1.get_positions().T).T )
        self.assertTrue( comparator.compare(s1,s2) )

    def test_hcp_symmetry_ops( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")
        comparator = StructureComparator()
        sg = spacegroup.Spacegroup(194)
        cell = s2.get_cell().T
        inv_cell = np.linalg.inv(cell)
        for op in sg.get_rotations():
            s1 = read("test_structures/mixStruct.xyz")
            s2 = read("test_structures/mixStruct.xyz")
            transformed_op = cell.dot(op).dot(inv_cell)
            s2.set_positions( transformed_op.dot(s1.get_positions().T).T )
            if ( has_pymat_gen ):
                m = StructureMatcher(ltol=0.3, stol=0.4, angle_tol=5,
                                 primitive_cell=True, scale=True)
                str1 = atoms_to_structure(s1)
                str2 = atoms_to_structure(s2)
            self.assertTrue( comparator.compare(s1,s2) )

    def test_fcc_symmetry_ops( self ):
        s1 = read("test_structures/fccMix.xyz")
        s2 = read("test_structures/fccMix.xyz")
        comparator = StructureComparator()
        sg = spacegroup.Spacegroup(225)
        cell = s2.get_cell().T
        inv_cell = np.linalg.inv(cell)
        for op in sg.get_rotations():
            s1 = read("test_structures/fccMix.xyz")
            s2 = read("test_structures/fccMix.xyz")
            transformed_op = cell.dot(op).dot(inv_cell)
            s2.set_positions( transformed_op.dot(s1.get_positions().T).T )
            self.assertTrue( comparator.compare(s1,s2) )

    def test_bcc_symmetry_ops( self ):
        s1 = read("test_structures/bcc_mix.xyz")
        s2 = read("test_structures/bcc_mix.xyz")
        comparator = StructureComparator()
        sg = spacegroup.Spacegroup(229)
        cell = s2.get_cell().T
        inv_cell = np.linalg.inv(cell)
        for op in sg.get_rotations():
            s1 = read("test_structures/bcc_mix.xyz")
            s2 = read("test_structures/bcc_mix.xyz")
            transformed_op = cell.dot(op).dot(inv_cell)
            s2.set_positions( transformed_op.dot(s1.get_positions().T).T )
            self.assertTrue( comparator.compare(s1,s2) )

    def test_bcc_translation( self ):
        s1 = read("test_structures/bcc_mix.xyz")
        s2 = read("test_structures/bcc_mix.xyz")
        s2.set_positions( s2.get_positions()+np.array([6.0,-2.0,1.0]) )
        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

    def test_one_atom_out_of_pos( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")
        comparator = StructureComparator( angleTolDeg=0.2, position_tolerance=0.01 )
        pos = s1.get_positions()
        pos[0,:] += 0.2
        s2.set_positions(pos)
        self.assertFalse( comparator.compare(s1,s2) )

class LargeTestHCP(object):
    def __init__(self):
        self.structure = read("test_structures/mixStruct.xyz")

if __name__ == "__main__":
    # Run the unittests
    unittest.main()
