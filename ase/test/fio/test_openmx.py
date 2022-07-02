import pytest
import numpy as np
from ase.io import openmx
from ase.io.openmx import units


@pytest.fixture
def omxio():
    return openmx


@pytest.fixture
def omxlogtxt():
    logtxt = """
*******************************************************
*******************************************************
 Welcome to OpenMX   Ver. 3.9.2
 Copyright (C), 2002-2019, T. Ozaki
 OpenMX comes with ABSOLUTELY NO WARRANTY.
 This is free software, and you are welcome to
 redistribute it under the constitution of the GNU-GPL.
*******************************************************
*******************************************************

...

*******************************************************
  Allocation of atoms to proccesors at MD_iter=    5
*******************************************************

 proc =   0  # of atoms=   5  estimated weight=         5.00000



TFNAN=      20   Average FNAN=   4.00000
TSNAN=       0   Average SNAN=   0.00000
<truncation> CpyCell= 2 ct_AN=   1 FNAN SNAN   4   0
<truncation> CpyCell= 2 ct_AN=   2 FNAN SNAN   4   0
<truncation> CpyCell= 2 ct_AN=   3 FNAN SNAN   4   0
<truncation> CpyCell= 2 ct_AN=   4 FNAN SNAN   4   0
<truncation> CpyCell= 2 ct_AN=   5 FNAN SNAN   4   0
<UCell_Box> Info. of cutoff energy and num. of grids
lattice vectors (bohr)
A  = -1.0,  0.0,  1.0
B  =  2.0, 3.0, 4.0
C  =  5.0, 6.0, 7.0
reciprocal lattice vectors (bohr^-1)
RA =  0.332491871581,  0.000000000000,  0.000000000000
RB =  0.000000000000,  0.332491871581,  0.000000000000
RC =  0.000000000000,  0.000000000000,  0.332491871581
Required cutoff energy (Ryd) for 3D-grids = 300.0000
    Used cutoff energy (Ryd) for 3D-grids = 304.7058, 304.7058, 304.7058
Num. of grids of a-, b-, and c-axes = 105, 105, 105
Cell_Volume =   6748.333037104149 (Bohr^3)
GridVol     =      0.005829463805 (Bohr^3)
Cell vectors (bohr) of the grid cell (gtv)
  gtv_a =  0.179973903674,  0.000000000000,  0.000000000000
  gtv_b =  0.000000000000,  0.179973903674,  0.000000000000
  gtv_c =  0.000000000000,  0.000000000000,  0.179973903674
  |gtv_a| =  0.179973903674
  |gtv_b| =  0.179973903674
  |gtv_c| =  0.179973903674
Num. of grids overlapping with atom    1 = 89763
Num. of grids overlapping with atom    2 = 89755
Num. of grids overlapping with atom    3 = 89755
Num. of grids overlapping with atom    4 = 89826
Num. of grids overlapping with atom    5 = 89826

*******************************************************
             SCF calculation at MD = 5
*******************************************************

<MD= 5>  Calculation of the overlap matrix
<MD= 5>  Calculation of the nonlocal matrix
<MD= 5>  Calculation of the VNA projector matrix

******************* MD= 5  SCF= 1 *******************
<Restart>  Found restart files
<Poisson>  Poisson's equation using FFT...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.005632
<Band_DFT>  DM, time=0.005873
    1    C  MulP   2.0381  2.0381 sum   4.0762
    2    H  MulP   0.4908  0.4908 sum   0.9815
    3    H  MulP   0.4908  0.4908 sum   0.9815
    4    H  MulP   0.4902  0.4902 sum   0.9804
    5    H  MulP   0.4902  0.4902 sum   0.9804
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.414846617438  dUele     =   1.000000000000
<DFT>  NormRD =    1.000000000000  Criterion =   0.000100000000

******************* MD= 5  SCF= 2 *******************
<Poisson>  Poisson's equation using FFT...
<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.005775
<Band_DFT>  DM, time=0.005961
    1    C  MulP   2.0262  2.0262 sum   4.0523
    2    H  MulP   0.4938  0.4938 sum   0.9875
    3    H  MulP   0.4938  0.4938 sum   0.9875
    4    H  MulP   0.4932  0.4932 sum   0.9863
    5    H  MulP   0.4932  0.4932 sum   0.9863
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.396807466782  dUele     =   0.018039150656
<DFT>  NormRD =    0.009345289617  Criterion =   0.000100000000

******************* MD= 5  SCF= 3 *******************
<Poisson>  Poisson's equation using FFT...
<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.005584
<Band_DFT>  DM, time=0.005872
    1    C  MulP   2.0336  2.0336 sum   4.0671
    2    H  MulP   0.4919  0.4919 sum   0.9838
    3    H  MulP   0.4919  0.4919 sum   0.9838
    4    H  MulP   0.4913  0.4913 sum   0.9826
    5    H  MulP   0.4913  0.4913 sum   0.9826
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.407847791461  dUele     =   0.011040324679
<DFT>  NormRD =    0.000275837642  Criterion =   0.000100000000

******************* MD= 5  SCF= 4 *******************
<Poisson>  Poisson's equation using FFT...
<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.005590
<Band_DFT>  DM, time=0.005857
    1    C  MulP   2.0335  2.0335 sum   4.0669
    2    H  MulP   0.4919  0.4919 sum   0.9839
    3    H  MulP   0.4919  0.4919 sum   0.9839
    4    H  MulP   0.4913  0.4913 sum   0.9827
    5    H  MulP   0.4913  0.4913 sum   0.9827
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.407715585874  dUele     =   0.000132205588
<DFT>  NormRD =    0.000111186819  Criterion =   0.000100000000

******************* MD= 5  SCF= 5 *******************
<Poisson>  Poisson's equation using FFT...
<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.006177
<Band_DFT>  DM, time=0.006386
    1    C  MulP   2.0335  2.0335 sum   4.0671
    2    H  MulP   0.4919  0.4919 sum   0.9838
    3    H  MulP   0.4919  0.4919 sum   0.9838
    4    H  MulP   0.4913  0.4913 sum   0.9826
    5    H  MulP   0.4913  0.4913 sum   0.9826
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.407836870209  dUele     =   0.000121284335
<DFT>  NormRD =    0.000094670492  Criterion =   0.000100000000

******************* MD= 5  SCF= 6 *******************
<Poisson>  Poisson's equation using FFT...
<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.006300
<Band_DFT>  DM, time=0.006405
    1    C  MulP   2.0335  2.0335 sum   4.0669
    2    H  MulP   0.4919  0.4919 sum   0.9839
    3    H  MulP   0.4919  0.4919 sum   0.9839
    4    H  MulP   0.4913  0.4913 sum   0.9827
    5    H  MulP   0.4913  0.4913 sum   0.9827
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.407736644483  dUele     =   0.000100225726
<DFT>  NormRD =    0.000008150615  Criterion =   0.000100000000

******************* MD= 5  SCF= 7 *******************
<Poisson>  Poisson's equation using FFT...
<Set_Hamiltonian>  Hamiltonian matrix for VNA+dVH+Vxc...
<Band>  Solving the eigenvalue problem...
 KGrids1:  -0.37500  -0.12500   0.12500   0.37500
 KGrids2:  -0.37500  -0.12500   0.12500   0.37500
 KGrids3:  -0.37500  -0.12500   0.12500   0.37500
<Band_DFT>  Eigen, time=0.005642
<Band_DFT>  DM, time=0.005858
    1    C  MulP   2.0335  2.0335 sum   4.0669
    2    H  MulP   0.4919  0.4919 sum   0.9839
    3    H  MulP   0.4919  0.4919 sum   0.9839
    4    H  MulP   0.4913  0.4913 sum   0.9827
    5    H  MulP   0.4913  0.4913 sum   0.9827
 Sum of MulP: up   =     4.00000 down          =     4.00000
              total=     8.00000 ideal(neutral)=     8.00000
<DFT>  Total Spin Moment (muB) =  0.000000000000
<DFT>  Mixing_weight= 0.400000000000
<DFT>  Uele   =   -3.407727267511  dUele     =   0.000009376973
<DFT>  NormRD =    0.000000372693  Criterion =   0.000100000000
<MD= 5>  Force calculation
  Force calculation #1
  Force calculation #2
  Force calculation #3
  Force calculation #4
  Force calculation #5
  Stress calculation #1
  Stress calculation #2
  Stress calculation #3
  Stress calculation #4
  Stress calculation #5
<MD= 5>  Total Energy
  Force calculation #6
  Force calculation #7

*******************************************************
                  Dipole moment (Debye)
*******************************************************

 Absolute D        0.01185441

                      Dx                Dy                Dz
 Total             -0.00000016       -0.00000016       -0.01185441
 Core               0.00000307        0.00000307        0.89175352
 Electron          -0.00000323       -0.00000323       -0.90360794
 Back ground       -0.00000000        0.00000000       -0.00000000

*******************************************************
               Stress tensor (Hartree/bohr^3)
*******************************************************

       0.10000000        0.20000000        0.30000000
       0.40000000        0.50000000        0.60000600
       0.70000070        0.80000008        0.90000000

*******************************************************
                Total Energy (Hartree) at MD = 5
*******************************************************

  Uele  =      -3.407727267511

  Ukin  =       6.066893303510
  UH0   =     -14.574635266516
  UH1   =       0.027937933843
  Una   =      -5.320733485564
  Unl   =      -0.210370130142
  Uxc0  =      -1.608676794246
  Uxc1  =      -1.608676794246
  Ucore =       9.091616130928
  Uhub  =       0.000000000000
  Ucs   =       0.000000000000
  Uzs   =       0.000000000000
  Uzo   =       0.000000000000
  Uef   =       0.000000000000
  UvdW  =       0.000000000000
  Uch   =       0.000000000000
  Utot  =      -8.136645102433

  UpV   =       0.000000000000
  Enpy  =      -8.136645102433
  Note:

  Uele:   band energy
  Ukin:   kinetic energy
  UH0:    electric part of screened Coulomb energy
  UH1:    difference electron-electron Coulomb energy
  Una:    neutral atom potential energy
  Unl:    non-local potential energy
  Uxc0:   exchange-correlation energy for alpha spin
  Uxc1:   exchange-correlation energy for beta spin
  Ucore:  core-core Coulomb energy
  Uhub:   LDA+U energy
  Ucs:    constraint energy for spin orientation
  Uzs:    Zeeman term for spin magnetic moment
  Uzo:    Zeeman term for orbital magnetic moment
  Uef:    electric energy by electric field
  UvdW:   semi-empirical vdW energy
  Uch:    penalty term to create a core hole
  UpV:    pressure times volume
  Enpy:   Enthalpy = Utot + UpV
  (see also PRB 72, 045121(2005) for the energy contributions)


*******************************************************
           Computational times (s) at MD = 5
*******************************************************

  DFT in total      =    5.38968

  Set_OLP_Kin       =    0.07500
  Set_Nonlocal      =    0.05567
  Set_ProExpn_VNA   =    0.14758
  Set_Hamiltonian   =    1.73793
  Poisson           =    1.02132
  diagonalization   =    0.08732
  Mixing_DM         =    0.00016
  Force             =    0.24400
  Total_Energy      =    0.47798
  Set_Aden_Grid     =    0.09901
  Set_Orbitals_Grid =    0.07927
  Set_Density_Grid  =    0.34696
  RestartFileDFT    =    0.04283
  Mulliken_Charge   =    0.00041
  FFT(2D)_Density   =    0.00000

*******************************************************
             MD or geometry opt. at MD = 5
*******************************************************

<DIIS>  |Maximum force| (Hartree/Bohr) = 0.004208356187
<DIIS>  Criterion       (Hartree/Bohr) = 0.000100000000

     atom=   1, XYZ(ang) Fxyz(a.u.)=   0.1 0.2 0.3 0.4 0.5 0.6
     atom=   2, XYZ(ang) Fxyz(a.u.)=   -0.1 -0.2 -0.3 -0.4 -0.5 -0.6
     atom=   3, XYZ(ang) Fxyz(a.u.)=   1.0 2.0 3.0 4.0 5.0 6.0
     atom=   4, XYZ(ang) Fxyz(a.u.)=   -1.0 -2.0 -3.0 -4.0 -5.0 -6.0
     atom=   5, XYZ(ang) Fxyz(a.u.)=   0.0 0.01 0.001 -0.0 -0.01 -0.001

outputting data on grids to files...

    """
    return logtxt


def test_parse_openmx_log_cell(omxio, omxlogtxt):
    unit = units['bohr']
    res = np.array(
        [[-1.0, 0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0]]) * unit
    ans = omxio.parse_openmx_log_cell(omxlogtxt)
    assert np.all(np.isclose(ans, res))


def test_parse_openmx_log_pbc(omxio, omxlogtxt):
    assert omxio.parse_openmx_log_pbc(omxlogtxt)


def test_parse_openmx_log_symbols(omxio, omxlogtxt):
    ans = ['C', 'H', 'H', 'H', 'H']
    assert omxio.parse_openmx_log_symbols(omxlogtxt) == ans


def test_parse_openmx_log_positions(omxio, omxlogtxt):
    ans = np.array([[0.1, 0.2, 0.3],
                    [-0.1, -0.2, -0.3],
                    [1.0, 2.0, 3.],
                    [-1., -2., -3.],
                    [0.0, 0.01, 0.001]])
    assert np.all(np.isclose(omxio.parse_openmx_log_positions(omxlogtxt), ans))


def test_parse_openmx_log_energy(omxio, omxlogtxt):
    assert np.isclose(omxio.parse_openmx_log_energy(omxlogtxt), -221.409390825)


def test_parse_openmx_log_forces(omxio, omxlogtxt):
    unit = units['hartree/bohr']
    res = np.array(
        [[0.4, 0.5, 0.6],
        [-0.4, -0.5, -0.6],
        [4.0, 5.0, 6.0],
        [-4.0, -5.0, -6.0],
        [-0.0, -0.01, -0.001]]) * unit
    assert np.all(np.isclose(omxio.parse_openmx_log_forces(omxlogtxt), res))


def test_parse_openmx_log_stress(omxio, omxlogtxt):
    unit = units['hartree/bohr^3']
    res = np.array([[0.1, 0.2, 0.3],
                    [0.4, 0.5, 0.6],
                    [0.7, 0.8, 0.9]]) * unit
    assert np.all(np.isclose(omxio.parse_openmx_log_stress(omxlogtxt), res))


def test_parse_openmx_log_version(omxio, omxlogtxt):
    assert omxio.parse_openmx_log_version(omxlogtxt) == '3.9.2'
