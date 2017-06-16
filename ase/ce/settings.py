class CrystalStructure(object):
    def __init__(self, alat, db_name, max_cluster_size, max_cluster_dia):
        self.alat = alat
        self.db_name = db_name
        self.max_cluster_size = max_cluster_size
        self.max_cluster_dia = max_cluster_dia

class Rocksalt(CrystalStructure):
    """
    Class that stores the necessary information about rocksalt structures.
    """
    def __init__(self, cations, anions, c_ratio, a_ratio,\
                 alat, db_name, max_cluster_size, max_cluster_dia):
        """
        cations: chemical symbols of each type of cation
        anions: chemical symbols of each type of anion
        c_ratio: ratio of consituting cations
        a_ratio: ratio of consituting anions
        alat: lattic constant
        db_name: name of the database file
        max_cluster_size: maximum size (number of atoms in a cluster)
        max_cluster_dia: maximum diameter of cluster (in unit of alat)
        """
        # ----------------------
        # Perform sanity check
        # ----------------------
        # check list dimensions
        if len(cations) != len(c_ratio) or len(anions) != len(a_ratio):
            raise ValueError("Dimensions of ion and ratio lists must agree.")
        # -------------------------------
        # Passed tests. Assign parameters
        # -------------------------------
        super(Rocksalt, self).__init__(self, alat, db_name, max_cluster_size, max_cluster_dia)
        self.cations = cations
        self.anions = anions
        self.c_ratio = c_ratio
        self.a_ratio = a_ratio

    @property
    def cluster_list(self):
        """
        Create a list of parameters used to describe the structure.
        """
        cluster_list = []
        # # order = 1 (singlets)
        # all_elements = cat + an 
        # for element in all_elements:
        #     cluster_list.append('c1_%s' %element)
        for size in range(self.max_cluster_size):
            if size < 2:
                cluster_list.append('c%d' %size)
            else:
                cluster_list.append('c%d_' %s)

        return cluster_list

    @property
    def cluster_dict(self):
        """
        Create a dictionary containing all the parameters and their values
        ininitalized to 0.0
        """
        cluster_dict = dict((i, 0.0) for i in self.cluster_list)
        return cluster_dict
