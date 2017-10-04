from ase.build import tools as asetools
import unittest
import numpy as np
from ase.build import bulk
import copy

class StructureComparator( object ):
    def __init__( self, angleTolDeg=1, position_tolerance=1E-4 ):
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
                del dot2[closestIndex]
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

        matrices = self.get_rotation_reflection_matrices()
        if ( not self.positions_match(matrices) ):
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

        pos1 -= np.mean(pos1,axis=0)
        pos2 -= np.mean(pos2,axis=0)

        least_freq_element = self.get_least_frequent_element()

        position1 = []
        position2 = []

        for i in range( len(self.s1) ):
            symbol = self.s1[i].symbol
            if ( symbol == least_freq_element ):
                position1.append( pos1[i,:] )

            symbol = self.s2[i].symbol
            if ( symbol == least_freq_element ):
                position2.append( pos2[i,:] )
        return np.array( position1 ), np.array( position2 )

    def positions_match( self, rotation_reflection_matrices ):
        """
        Check if the position and elements match
        """
        pos1_ref = self.s1.get_positions()
        pos2 = self.s2.get_positions()
        cm_1 = np.mean( pos1_ref, axis=0 )
        cm_2 = np.mean( pos2, axis=0 )
        counter = 0
        positions_match = False
        for matrix in rotation_reflection_matrices:
            pos1 = copy.deepcopy(pos1_ref)
            d = cm_2 - matrix.dot(cm_1)
            pos1 -= d
            pos1 = matrix.dot(pos1.T).T
            if ( np.all( np.abs(pos1-pos2) < self.position_tolerance ) ):
                # Update the positions of s1 such that they match the ones of s2
                self.s1.set_positions(pos1)
                return True
        return False

    def get_rotation_reflection_matrices( self ):
        """
        Computes the closest rigid body transformation matrix by solving Procrustes problem
        """
        s1_pos_ref = copy.deepcopy( self.s1.get_positions() )
        pos1_ref, pos2 = self.extract_positions_of_least_frequent_element()
        rot_reflection_mat = []
        for i in range( pos1_ref.shape[0] ):
            new_pos = s1_pos_ref - pos1_ref[i]
            self.s1.set_positions( new_pos )
            pos1, pos2 = self.extract_positions_of_least_frequent_element()
            pos1 -= np.mean( pos1, axis=0 )
            pos2 -= np.mean( pos2, axis=0 )
            #M = pos1.T.dot(pos2)
            M = pos2.T.dot(pos1)
            U, S, V = np.linalg.svd(M)
            bestMatrix =U.dot(V)
            rot_reflection_mat.append( bestMatrix )

        self.s1.set_positions( s1_pos_ref )
        return rot_reflection_mat


# ======================== UNIT TESTS ==========================================
class TestStructureComparator( unittest.TestCase ):
    def test_compare( self ):
        s1 = bulk( "Al" )
        s1 = s1*(2,2,2)
        s2 = bulk( "Al" )
        s2 = s2*(2,2,2)

        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

    def test_single_impurity( self ):
        s1 = bulk( "Al" )
        s1 = s1*(2,2,2)
        s1[0].symbol = "Mg"
        s2 = bulk( "Al" )
        s2 = s2*(2,2,2)
        s2[3].symbol = "Mg"
        comparator = StructureComparator()
        self.assertTrue( comparator.compare(s1,s2) )

if __name__ == "__main__":
    unittest.main()
