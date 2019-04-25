from ase.neighborlist import neighbor_list


class TransMatrixConstructor(object):
    """Class for constructor translation matrices."""
    def __init__(self, atoms, cutoff):
        self.num_atoms = len(atoms)
        self.neighbor = self._construct_neighbor_list(atoms, cutoff)    

    def _construct_neighbor_list(self, atoms, cutoff):
        """
        Construct neighbour list structure
        """
        raw_neigh = neighbor_list('ijDS', atoms, cutoff)

        # Transfer to more convenienct strucutre
        neighbor = [{"neigh_index": [], "dist": [], "shift": []}]*len(atoms)

        for i, j, d, s in raw_neigh:
            neighbor[i]["neigh_index"].append(j)
            neighbor[i]["dist"].append(d)
            neighbor[i]["shift"].append(shift)
        return neighbor

    def _map_one(self, indx, template_indx):
        mapped = {template_indx: indx}

        neigh_indx = self.neighbor[indx]["neigh_index"]
        neigh_dist = self.neighbor[indx]["dist"]
        ref_dists = self.neighbor[template_indx]["dist"]
        ref_indx = self.neighbor[template_indx]["neigh_index"]

        for i, d in zip(neigh_indx, neigh_dist):
            dist_vec = np.array(ref_dists) - np.array(d)
            lengths = dist_vec.dot(dist_vec)
            corresponding_indx = ref_indx[np.argmin(lengths)]
            mapped[corresponding_indx] = i
        return mapped

    def construct(self, ref_symm_group, symm_group):
        """Construct the translation matrix."""
        tm = []
        for indx in range(self.num_atoms):
            group = symm_group[indx]
            ref_indx = ref_symm_group[group]
            mapped = self._map_one(indx, ref_indx)
            tm.append(mapped)
        return tm
