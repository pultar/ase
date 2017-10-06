from __future__ import print_function

import sys
import pickle
import numpy as np

from ase.utils import pickleload


write = True
#write = False

v = sys.version_info[0]

if write:
    fn = 'py{0}.pcl'.format(v)
    print('writing', fn)
    pickle.dump(np.array([0., 0.5, 1.]), open(fn, 'wb'), protocol=2)

if v == 2:
    fn = 'py3.pcl'
else:
    fn = 'py2.pcl'
print('reading', fn)
pickleload(open(fn, 'rb'))

pickleload(open('ir-d0.010.0x-.pckl', 'rb'))
