from ase.neighborlist import neighbor_list
import numpy as np


class TransMatrixConstructor(object):
    """Class for constructor translation matrices.

    Arguments
    ==========
    atoms: Atoms
        ASE atoms object (assumed to be wrapped and sorted)
    cutoff: float
        Cut-off distance in angstrom
    """
    def __init__(self, atoms, cutoff):
        self.num_atoms = len(atoms)
        self.neighbor = self._construct_neighbor_list(atoms, cutoff)

    def _construct_neighbor_list(self, atoms, cutoff):
        """
        Construct neighbour list structure
        """
        i_first, i_second, d_vec = neighbor_list('ijD', atoms, cutoff)
        # Transfer to more convenienct strucutre
        neighbor = [{"neigh_index": [], "dist": []} for _ in range(len(atoms))]

        for i in range(len(i_first)):
            neighbor[i_first[i]]["neigh_index"].append(i_second[i])
            neighbor[i_first[i]]["dist"].append(d_vec[i])
        return neighbor

    def _map_one(self, indx, template_indx):
        mapped = {template_indx: indx}

        neigh_indx = self.neighbor[indx]["neigh_index"]
        neigh_dist = self.neighbor[indx]["dist"]
        ref_dists = self.neighbor[template_indx]["dist"]
        ref_indx = self.neighbor[template_indx]["neigh_index"]

        for i, d in zip(neigh_indx, neigh_dist):
            dist_vec = np.array(ref_dists) - np.array(d)
            lengths = np.sum(dist_vec**2, axis=1)
            corresponding_indx = ref_indx[np.argmin(lengths)]
            mapped[corresponding_indx] = i
        return mapped

    def construct(self, ref_symm_group, symm_group):
        """Construct the translation matrix.

        Arguments
        =========
        ref_symm_group: list
            List of reference indices. If the atoms object has only one
            basis this will be [0], otherwise it can for instance be
            [0, 5, 15] if the atoms object have three basis
        symm_group: list
            List with the symmetry groups of each atoms object. If the object
            has only one basis this will be [0, 0, 0, ...0], if it has two
            basis this can be [0, 0, 1, 1, 0, 1...]. The reference index of the
            symmetry group of atoms k will be ref_symm_group[symm_group[k]]
        """
        tm = []
        for indx in range(self.num_atoms):
            group = symm_group[indx]
            ref_indx = ref_symm_group[group]
            mapped = self._map_one(indx, ref_indx)
            tm.append(mapped)
        return tm
