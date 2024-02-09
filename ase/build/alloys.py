from ase.io import read
from ase.formula import non_metals
from ase.visualize import view
import numpy as np
import random


def alloy(template, elements, fractions,
         replace=None, size=None):

    if size is not None:
        if np.array(size).shape == (3,):
            new = template.repeat(size)
        else:
            from ase.build import make_supercell
            new = make_supercell(template, size)
    else:
        new = template.copy()

    if replace is None:
        replace = list(np.unique(
            [s for s in new.get_chemical_symbols() if s not in non_metals]
        ))

    indexes = []
    for i, symbol in enumerate(new.get_chemical_symbols()):
        if symbol in replace:
            indexes.append(i)
    random.shuffle(indexes)

    symbols = []
    for elem, frac in zip(elements, fractions):
        N = round(len(indexes)*frac)
        for n in range(N):
            symbols.append(elem)
    
    for index, symbol in zip(indexes, symbols):
        new[index].symbol = symbol

    return new
