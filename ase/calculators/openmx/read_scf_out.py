"""
The ASE Calculator for OpenMX <http://www.openmx-square.org>: Python interface
to the software package for nano-scale material simulations based on density
functional theories.
    Copyright (C) 2018 Jae Hwan Shim and JaeJun Yu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 2.1 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with ASE.  If not, see <http://www.gnu.org/licenses/>.

Behave like read_scfout.c of OpenMX module but written in python.
"""

import struct
import numpy as np


def easyReader(byte, data_type, shape):
    data_size = {'d': 8, 'i': 4}
    data_struct = {'d': float, 'i': int}
    dt = data_type
    ds = data_size[data_type]
    unpack = struct.unpack
    if len(byte) == ds:
        if dt == 'i':
            return data_struct[dt].from_bytes(byte, byteorder='little')
        elif dt == 'd':
            return np.array(unpack(dt*(len(byte)//ds), byte))[0]
    elif shape is not None:
        return np.array(unpack(dt*(len(byte)//ds), byte)).reshape(shape)
    else:
        return np.array(unpack(dt*(len(byte)//ds), byte))


def inte(byte, shape=None):
    return easyReader(byte, 'i', shape)


def floa(byte, shape=None):
    return easyReader(byte, 'd', shape)


def readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f):
        myOLP = []
        myOLP.append([])
        for ct_AN in range(1, atomnum + 1):
            myOLP.append([])
            TNO1 = Total_NumOrbs[ct_AN]
            for h_AN in range(FNAN[ct_AN] + 1):
                myOLP[ct_AN].append([])
                Gh_AN = natn[ct_AN][h_AN]
                TNO2 = Total_NumOrbs[Gh_AN]
                for i in range(TNO1):
                    myOLP[ct_AN][h_AN].append(floa(f.read(8*TNO2)))
        return myOLP


def read_scf_out(fname='./ase/calculators/openmx/test/FeCS-gga.scfout'):
    from numpy import insert as ins
    try:
        f = open(fname, mode='rb')
        atomnum, SpinP_switch = inte(f.read(8))
        Catomnum = inte(f.read(4))
        Latomnum = inte(f.read(4))
        Ratomnum = inte(f.read(4))
        TCpyCell = inte(f.read(4))
        atv = floa(f.read(8*4*(TCpyCell+1)), shape=(TCpyCell+1, 4))
        atv_ijk = inte(f.read(4*4*(TCpyCell+1)), shape=(TCpyCell+1, 4))
        Total_NumOrbs = np.insert(inte(f.read(4*(atomnum))), 0, 1, axis=0)
        FNAN = np.insert(inte(f.read(4*(atomnum))), 0, 0, axis=0)
        _natn = inte(f.read(4*np.sum(FNAN[1:] + 1)))
        natn = np.insert(np.split(_natn, np.cumsum(FNAN[1:] + 1)),
                         0, np.zeros(FNAN[0] + 1), axis=0)[:-1]
        _ncn = inte(f.read(4*np.sum(FNAN[1:] + 1)))
        ncn = np.insert(np.split(_ncn, np.cumsum(FNAN[1:] + 1)),
                        0, np.zeros(FNAN[0] + 1), axis=0)[:-1]
        tv = ins(floa(f.read(8*3*4), shape=(3, 4)), 0, [0, 0, 0, 0], axis=0)
        rtv = ins(floa(f.read(8*3*4), shape=(3, 4)), 0, [0, 0, 0, 0], axis=0)
        Gxyz = ins(floa(f.read(8*(atomnum)*4), shape=(atomnum, 4)),
                   0, [0., 0., 0., 0.], axis=0)
        Hks = []
        for spin in range(SpinP_switch + 1):
            Hks.append([])
            Hks[spin].append([np.zeros(FNAN[0] + 1)])
            for ct_AN in range(1, atomnum + 1):
                Hks[spin].append([])
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN] + 1):
                    Hks[spin][ct_AN].append([])
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        Hks[spin][ct_AN][h_AN].append(floa(f.read(8*TNO2)))
        iHks = []
        if SpinP_switch == 3:
            for spin in range(SpinP_switch + 1):
                iHks.append([])
                iHks[spin].append([np.zeros(FNAN[0] + 1)])
                for ct_AN in range(1, atomnum + 1):
                    iHks[spin].append([])
                    TNO1 = Total_NumOrbs[ct_AN]
                    for h_AN in range(FNAN[ct_AN] + 1):
                        iHks[spin][ct_AN].append([])
                        Gh_AN = natn[ct_AN][h_AN]
                        TNO2 = Total_NumOrbs[Gh_AN]
                        for i in range(TNO1):
                            dat = floa(f.read(8*TNO2))
                            iHks[spin][ct_AN][h_AN].append(dat)
        OLP = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        OLPpox = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        OLPpoy = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        OLPpoz = readOverlap(atomnum, Total_NumOrbs, FNAN, natn, f)
        DM = []
        for spin in range(SpinP_switch + 1):
            DM.append([])
            DM[spin].append([])
            for ct_AN in range(1, atomnum + 1):
                DM[spin].append([])
                TNO1 = Total_NumOrbs[ct_AN]
                for h_AN in range(FNAN[ct_AN] + 1):
                    DM[spin][ct_AN].append([])
                    Gh_AN = natn[ct_AN][h_AN]
                    TNO2 = Total_NumOrbs[Gh_AN]
                    for i in range(TNO1):
                        DM[spin][ct_AN][h_AN].append(floa(f.read(8*TNO2)))
        Solver = inte(f.read(4))
        ChemP, E_Temp = floa(f.read(8*2))
        dipole_moment_core = floa(f.read(8*3))
        dipole_moment_background = floa(f.read(8*3))
        Valence_Electrons, Total_SpinS = floa(f.read(8*2))

    except IOError:
        raise(IOError('Can not find %s' % fname))
    f.close()
    scf_out = {'atomnum': atomnum, 'SpinP_switch': SpinP_switch,
               'Catomnum': Catomnum, 'Latomnum': Latomnum, 'Hks': Hks,
               'Ratomnum': Ratomnum, 'TCpyCell': TCpyCell, 'atv': atv,
               'Total_NumOrbs': Total_NumOrbs, 'FNAN': FNAN, 'natn': natn,
               'ncn': ncn, 'tv': tv, 'rtv': rtv, 'Gxyz': Gxyz, 'OLP': OLP,
               'OLPpox': OLPpox, 'OLPpoy': OLPpoy, 'OLPpoz': OLPpoz,
               'Solver': Solver, 'ChemP': ChemP, 'E_Temp': E_Temp,
               'dipole_moment_core': dipole_moment_core, 'iHks': iHks,
               'dipole_moment_background': dipole_moment_background,
               'Valence_Electrons': Valence_Electrons, 'atv_ijk': atv_ijk,
               'Total_SpinS': Total_SpinS
               }
    return scf_out
