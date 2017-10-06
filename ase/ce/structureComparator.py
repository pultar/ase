from ase.build import tools as asetools
import unittest
import numpy as np
from ase.build import bulk
import copy
from ase.visualize import view
from ase import Atoms
from ase.io import read
import time
from matplotlib import pyplot as plt

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
        cell1 = self.s1.get_cell()
        cell2 = self.s2.get_cell()

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

        matrices, translations = self.get_rotation_reflection_matrices()
        #print ("test positions")
        if ( not self.positions_match(matrices, translations) ):
            return False

        #print ("test elements")
        exp1 = self.expand(self.s1)
        exp2 = self.expand(self.s2)
        if ( not self.elements_match(exp1,exp2) ):
            return True
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

        #pos1 -= np.mean(pos1,axis=0)
        #pos2 -= np.mean(pos2,axis=0)

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
        return np.array( position1 ), np.array( position2 )

    def positions_match( self, rotation_reflection_matrices, center_of_mass ):
        """
        Check if the position and elements match.
        Note that this function changes self.s1 and self.s2 to the rotation and
        translation that matches best. Hence, it is crucial that this function
        is called before the element comparison
        """
        # Position matching not implemented yet
        pos1_ref = self.s1.get_positions( wrap=True )
        pos2_ref = self.s2.get_positions( wrap=True )

        for matrix,com in zip(rotation_reflection_matrices,center_of_mass):
            pos1 = copy.deepcopy(pos1_ref)
            pos2 = copy.deepcopy(pos2_ref)

            # Translate to origin of rotation
            pos1 -= com[0]
            pos2 -= com[1]

            self.s1.set_positions( pos1 )
            self.s2.set_positions( pos2 )
            pos1 = self.s1.get_positions( wrap=True )
            pos2 = self.s2.get_positions( wrap=True )

            # Rotate
            pos1 = matrix.dot(pos1.T).T

            # Update the atoms positions
            self.s1.set_positions( pos1 )
            self.s2.set_positions( pos2 )
            pos1 = self.s1.get_positions( wrap=True )
            pos2 = self.s2.get_positions( wrap=True )
            self.s1.set_positions( pos1 )
            self.s2.set_positions( pos2 )

            exp1 = self.expand(self.s1)
            exp2 = self.expand(self.s2)
            pos1 = exp1.get_positions() # NOTE: No wrapping
            pos2 = exp2.get_positions() # NOTE: No wrapping

            # Check that all closest distances match
            used_sites = []
            for i in range(pos1.shape[0]):
                distances = np.sqrt( np.sum( (pos2-pos1[i,:])**2, axis=1 ) )
                closest = np.argmin(distances)
                if ( np.min(distances) > self.position_tolerance or closest in used_sites ):
                    break
                else:
                    used_sites.append(closest)

            if ( len(used_sites) == pos1.shape[0] ):
                return True
        return False

    def expand( self, ref_atoms ):
        """
        This functions adds additional atoms to for atoms close to the cell boundaries
        ensuring that atoms having crossed the cell boundaries due to numerical noise
        are properly detected
        """
        expaned_atoms = copy.deepcopy(ref_atoms)

        cell = ref_atoms.get_cell()
        normal_vectors = [np.cross(cell[:,0],cell[:,1]), np.cross(cell[:,0],cell[:,2]), np.cross(cell[:,1],cell[:,2])]
        normal_vectors = [vec/np.sqrt(np.sum(vec**2)) for vec in normal_vectors]
        positions = ref_atoms.get_positions(wrap=True)
        tol = 0.0001
        num_faces_close = 0

        for i in range( len(ref_atoms) ):
            surface_close = [False,False,False,False,False,False]
            # Face 1
            distance = np.abs( positions[i,:].dot(normal_vectors[0]) )
            symbol = ref_atoms[i].symbol
            if ( distance < tol ):
                newpos = positions[i,:]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
                surface_close[0] = True

            # Face 2
            distance = np.abs( (positions[i,:]-cell[:,2]).dot(normal_vectors[0]) )
            if ( distance < tol ):
                newpos = positions[i,:]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
                surface_close[1] = True

            # Face 3
            distance = np.abs( positions[i,:].dot(normal_vectors[1]) )
            if ( distance < tol ):
                newpos = positions[i,:] + cell[:,1]
                newAtom = Atoms( self.s1[i].symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
                surface_close[2] = True

            # Face 4
            distance = np.abs( (positions[i,:]-cell[:,1]).dot(normal_vectors[1]) )
            if ( distance < tol ):
                newpos = positions[i,:] - cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
                surface_close[3] = True

            # Face 5
            distance = np.abs(positions[i,:].dot(normal_vectors[2]))
            if ( distance < tol ):
                newpos = positions[i,:] + cell[:,0]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
                surface_close[4] = True

            # Face 6
            distance = np.abs( (positions[i,:]-cell[:,0]).dot(normal_vectors[2]) )
            if ( distance < tol ):
                newpos = positions[i,:] - cell[:,0]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
                surface_close[5] = True

            # Take edges into account
            if ( surface_close[0] and surface_close[2] ):
                newpos = positions[i,:] + cell[:,1] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[0] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[0] and surface_close[3] ):
                newpos = positions[i,:] - cell[:,1] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[0] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0] + cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[3] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] - cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[3] and surface_close[1] ):
                newpos = positions[i,:] - cell[:,1] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[3] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,1] - cell[:,0]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[1] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[2] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0] + cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[2] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] + cell[:,1]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[1] and surface_close[2] ):
                newpos = positions[i,:] + cell[:,1] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            if ( surface_close[1] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0] - cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)

            # Take corners into account
            if ( surface_close[0] and surface_close[2] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]+cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[0] and surface_close[3] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]-cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[0] and surface_close[2] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]+cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[0] and surface_close[3] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]-cell[:,1]+cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[3] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]-cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[3] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]-cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[2] and surface_close[4] ):
                newpos = positions[i,:] + cell[:,0]+cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)
            elif ( surface_close[1] and surface_close[2] and surface_close[5] ):
                newpos = positions[i,:] - cell[:,0]+cell[:,1]-cell[:,2]
                newAtom = Atoms( symbol, positions=[newpos] )
                expaned_atoms.extend(newAtom)

        return expaned_atoms

    def elements_match( self, s1, s2 ):
        """
        Checks that all the elements in the two atoms match
        """

        used_sites = []
        for i in range( len(s1) ):
            distances = np.sum ( (s2.get_positions()-s1.get_positions()[i,:])**2, axis=1 )
            closest = np.argmin( distances )
            if ( not s1[i].symbol == s2[closest].symbol or closest in used_sites ):
                return False
            else:
                used_sites.append( closest )
        return True

    def get_rotation_reflection_matrices( self ):
        """
        Computes the closest rigid body transformation matrix by solving Procrustes problem
        """
        s1_pos_ref = copy.deepcopy( self.s1.get_positions() )
        s2_pos_ref = copy.deepcopy( self.s2.get_positions() )
        atoms1_ref, atoms2_ref = self.extract_positions_of_least_frequent_element()
        rot_reflection_mat = []
        center_of_mass = []
        cell = self.s1.get_cell()

        delta_vec = 1E-6*(cell[:,0]+cell[:,1]+cell[:,2]) # Additional vector that is added to make sure that there always is an atom at the origin

        # Put on of the least frequent elements of structure 2 at the origin
        translation = atoms2_ref.get_positions()[0,:]-delta_vec
        atoms2_ref.set_positions( atoms2_ref.get_positions() - translation )
        atoms2_ref.set_positions( atoms2_ref.get_positions(wrap=True) )
        self.s2.set_positions( self.s2.get_positions()-translation)
        self.s2.set_positions( self.s2.get_positions(wrap=True) )

        # Also update structure 1 to have on of the least frequent elements at the corner
        translation = atoms1_ref.get_positions()[0,:]-delta_vec
        atoms1_ref.set_positions( atoms1_ref.get_positions()-translation)
        atoms1_ref.set_positions( atoms1_ref.get_positions(wrap=True))
        self.s1.set_positions( self.s1.get_positions()-translation)
        s1_pos_ref = self.s1.get_positions(wrap=True)
        self.s1.set_positions( s1_pos_ref )

        for i in range( len(atoms1_ref) ):
            # Change which atom is at the origin
            new_pos = s1_pos_ref - atoms1_ref.get_positions()[i,:]+delta_vec
            self.s1.set_positions( new_pos )
            new_pos = self.s1.get_positions(wrap=True)
            self.s1.set_positions(new_pos)
            #pos1, pos2 = self.extract_positions_of_least_frequent_element()
            atoms1, atoms2 = self.extract_positions_of_least_frequent_element()
            #view(atoms1)
            #view(atoms2)
            #time.sleep(1)

            cm1 = np.mean( atoms1.get_positions( wrap=True ), axis=0 )
            cm2 = np.mean( atoms2.get_positions( wrap=True), axis=0 )
            pos1 = atoms1.get_positions(wrap=True)
            pos2 = atoms2.get_positions(wrap=True)

            #cm1 = np.mean( atoms1.get_positions(), axis=0 )
            #cm2 = np.mean( atoms2.get_positions(), axis=0 )
            #pos1 = atoms1.get_positions()
            #pos2 = atoms2.get_positions()
            pos1 -= cm1
            pos2 -= cm2
            M = pos2.T.dot(pos1)
            U, S, V = np.linalg.svd(M)
            bestMatrix = U.dot(V)
            center_of_mass.append( (cm1,cm2) )
            rot_reflection_mat.append( bestMatrix )

        self.s1.set_positions( s1_pos_ref )
        return rot_reflection_mat, center_of_mass


# ======================== UNIT TESTS ==========================================
class TestStructureComparator( unittest.TestCase ):
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
        self.assertTrue( comparator.compare(s1,s2) )

    def test_translations( self ):
        s1 = read("test_structures/mixStruct.xyz")
        s2 = read("test_structures/mixStruct.xyz")

        xmax = 2.0*np.max(s1.get_cell())
        N = 3
        dx = xmax/N
        pos_ref = s2.get_positions()
        comparator = StructureComparator()
        number_of_correctly_identified = 0
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    displacepement = [ dx*i, dx*j,dx*k ]
                    new_pos = pos_ref + displacepement
                    s2.set_positions(new_pos)
                    s2.set_positions( s2.get_positions(wrap=True) )
                    if ( comparator.compare(s1,s2) ):
                        number_of_correctly_identified += 1

        msg = "Identified %d of %d as duplicates. All structures are known to be duplicates."%(number_of_correctly_identified,N**3)
        self.assertEqual( number_of_correctly_identified, N**3, msg=msg )






if __name__ == "__main__":
    unittest.main()
