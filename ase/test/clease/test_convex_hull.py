from __future__ import division
from ase.clease import ConvexHull
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.build import bulk
import numpy as np
import os

def binary():
    db_name = "test_binary_cnv_hull.db"
    db = connect(db_name)

    # Create energies that we know are on the 
    # convex hull
    cnv_hull_enegies = [-x*(8-x) - x + 0.2 for x in range(9)]

    for n_cu in range(9):
        atoms = bulk("Au")
        atoms = atoms*(2, 2, 2)

        for i in range(n_cu):
            atoms[i].symbol = "Cu"
        
        calc = SinglePointCalculator(atoms, energy=cnv_hull_enegies[n_cu])
        atoms.set_calculator(calc)
        db.write(atoms, converged=True, expected_cnv_dist=0.0)

        # Create a new structure with exactly the same 
        # composition, but higher energy
        calc = SinglePointCalculator(atoms, energy=cnv_hull_enegies[n_cu] + 0.5)
        atoms.set_calculator(calc)
        db.write(atoms, converged=True, expected_cnv_dist=0.5/len(atoms))

    cnv_hull = ConvexHull(db_name)
    from matplotlib import pyplot as plt
    cnv_hull.plot()
    plt.show()

    energies = []
    expected_dists = []
    comp = []
    for row in db.select():
        energies.append(row.energy/row.natoms)
        count = row.count_atoms()

        for k in ["Au", "Cu"]:
            if k in count.keys():
                count[k] /= row.natoms
            else:
                count[k] = 0.0
        comp.append(count)
        expected_dists.append(row.expected_cnv_dist)

    os.remove(db_name)

    # Calculate distance to the convex hull
    for c, tot_en, exp in zip(comp, energies, expected_dists):
        dist = cnv_hull.distance_to_convex_hull(c, tot_en)
        assert abs(dist - exp) < 1E-6

binary()




