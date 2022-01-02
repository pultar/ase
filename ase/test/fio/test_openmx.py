import pytest
import numpy as np
from ase.io import openmx


@pytest.fixture
def omxio():
    return openmx


@pytest.fixture
def omxlogtxt():
    logtxt = """

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
    A  = 18.897259885789,  0.000000000000,  0.000000000000
    B  =  0.000000000000, 18.897259885789,  0.000000000000
    C  =  0.000000000000,  0.000000000000, 18.897259885789
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

           0.00000216        0.00000104        0.00000000
           0.00000104        0.00000216        0.00000000
           0.00000000        0.00000000        0.00000207

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

         atom=   1, XYZ(ang) Fxyz(a.u.)=   0.0000    0.0000    0.0283     0.0000    0.0000    0.0017
         atom=   2, XYZ(ang) Fxyz(a.u.)=   0.6673    0.6673    0.6781     0.0042    0.0042    0.0023
         atom=   3, XYZ(ang) Fxyz(a.u.)=  -0.6673   -0.6673    0.6781    -0.0042   -0.0042    0.0023
         atom=   4, XYZ(ang) Fxyz(a.u.)=  -0.6545    0.6545   -0.6423    -0.0015    0.0015   -0.0031
         atom=   5, XYZ(ang) Fxyz(a.u.)=   0.6545   -0.6545   -0.6423     0.0015   -0.0015   -0.0031

    outputting data on grids to files...

    """
    return logtxt


@pytest.mark.calculator_lite
def test_parse_openmx_log_cell(omxio, txt):
    ans = omxio.parse_openmx_log_cell(txt)
    assert np.all(np.isclose(ans, 10. * np.identity(3)))


@pytest.mark.calculator_lite
def test_parse_openmx_log_pbc(omxio, txt):
    assert omxio.parse_openmx_log_pbc(txt)


@pytest.mark.calculator_lite
def test_parse_openmx_log_symbols(omxio, txt):
    assert omxio.parse_openmx_log_symbols(txt) == ['C', 'H', 'H', 'H', 'H']


@pytest.mark.calculator_lite
def test_parse_openmx_log_positions(omxio, txt):
    ans = np.array([[0.0000, 0.0000, 0.0283],
                    [0.6673, 0.6673, 0.6781],
                    [-0.6673, -0.6673, 0.6781],
                    [-0.6545, 0.6545, -0.6423],
                    [0.6545, -0.6545, -0.6423]])
    assert np.all(np.isclose(omxio.parse_openmx_log_positions(txt), ans))


@pytest.mark.calculator_lite
def test_parse_openmx_log_energy(omxio, txt):
    assert np.isclose(omxio.parse_openmx_log_energy(txt), -221.40939082558154)


@pytest.mark.calculator_lite
def test_parse_openmx_log_forces(omxio, txt):
    res = np.array(
        [[0.0, 0.0, 0.12341296101715354],
        [0.3033901958338358, 0.3033901958338358, 0.1902616482347784],
        [-0.3033901958338358, -0.3033901958338358, 0.1902616482347784],
        [-0.14912399456239386, 0.14912399456239386, -0.25196812874335517],
        [0.14912399456239386, -0.14912399456239386, -0.25196812874335517]])
    assert np.all(np.isclose(omxio.parse_openmx_log_forces(txt), res))


@pytest.mark.calculator_lite
def test_parse_openmx_log_stress(omxio, txt):
    res = np.array([[0.0015829, -0.00158841, 0.],
                    [-0.00158841, 0.0015829, 0.],
                    [0., 0., 0.00216869]])
    assert np.all(np.isclose(omxio.parse_openmx_log_stress(txt), res))


@pytest.mark.calculator_lite
def test_parse_openmx_log_version(omxio, txt):
    assert omxio.parse_openmx_log_version(txt) == '3.9.2'
