import os
import struct
import numpy as np


def easyReader(byte, data_type, shape):
    data_size = {'d': 8, 'i': 4}
    data_struct = {'d': float, 'i': int}
    dt = data_type
    ds = data_size[data_type]
    unpack = struct.unpack
    if len(byte) == ds:
        return data_struct[dt].from_bytes(byte, byteorder='little')
    elif shape is not None:
        return np.array(unpack(dt*(len(byte)//ds), byte)).reshape(shape)
    else:
        return np.array(unpack(dt*(len(byte)//ds), byte))


def inte(byte, shape=None):
    return easyReader(byte, 'i', shape)


def floa(byte, shape=None):
    return easyReader(byte, 'd', shape)


fname = './ase/calculators/openmx/test/FeCS-gga.scfout'
myfname = './ase/calculators/openmx/test/testFeCS-gga.scfout'
try:
    f = open(fname, mode='rb')
    atomnum, SpinP_switch = inte(f.read(8))
    Catomnum = inte(f.read(4))
    Latomnum = inte(f.read(4))
    Ratomnum = inte(f.read(4))
    TCpyCell = inte(f.read(4))
    print(atomnum, SpinP_switch, Catomnum, Latomnum, Ratomnum, TCpyCell)
    atv = floa(f.read(8*4*(TCpyCell+1)), shape=(TCpyCell+1, 4))
    print(atv)
    atv_ijk = inte(f.read(4*4*(TCpyCell+1)), shape=(TCpyCell+1, 4))
    Total_NumOrbs = np.array([0, inte(f.read(4*(atomnum)))]).flatten()
    FNAN = np.array([0, inte(f.read(4*(atomnum)))]).flatten()
    _natn = inte(f.read(4*np.sum(FNAN + 1)))
    natn = np.split(_natn, [i+1 for i in FNAN])
    _ncn = inte(f.read(4*np.sum(FNAN + 1)))
    ncn = np.split(_ncn, [i+1 for i in FNAN])
    tv = floa(f.read(8*4*4), shape=(4, 4))
    rtv = floa(f.read(8*4*4), shape=(4, 4))
    Gxyz = floa(f.read(8*(atomnum)*4), shape=(atomnum, 4))
#    Hks = floa(f.read(8*(SpinP_switch+1)*np.sum(FNAN+1) *
#               np.sum(Total_NumOrbs)*np.sum(Total_Num)))
#    atv = dobl

    dt = np.dtype("i4")
    myArray = np.fromfile(fname)
    print(myArray)
except IOError:
    print('Error')
f.close()
