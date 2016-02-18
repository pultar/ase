from __future__ import print_function

import numpy as np
from ase.db import connect
from ase.structure import molecule

db = connect('db_type.db', append=False)

a = molecule('C6H6')
a.set_array('tags', np.array([0.]*len(a)))

i = db.write(a)
row = db.get(i)

assert len(row.get('tags')) == len(a)
