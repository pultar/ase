"""
This file contains helper functions used by the XrayDebye class.

Also contains routine for calculation of atomic form factors and
X-ray wavelength dict.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist, pdist

# Table (1) of
# D. WAASMAIER AND A. KIRFEL, Acta Cryst. (1995). A51, 416-431
# 10.1107/s0108767394013292
waasmaier = {
    #       a1         b1         a2         b2         a3         b3          a4         b4         a5         b5         c
    "C": [
        2.657506,
        14.780758,
        1.078079,
        0.776775,
        1.490909,
        42.086843,
        -4.241070,
        -0.000294,
        0.713791,
        0.239535,
        4.297983,
    ],
    "N": [
        11.893780,
        0.000158,
        3.277479,
        10.232723,
        1.858092,
        30.344690,
        0.858927,
        0.656065,
        0.912985,
        0.217287,
        -11.804902,
    ],
    "O": [
        2.960427,
        14.182259,
        2.5088111,
        5.936858,
        0.637053,
        0.112726,
        0.722838,
        34.958481,
        1.142756,
        0.390240,
        0.027014,
    ],
    "P": [
        1.950541,
        0.908139,
        4.146930,
        27.044953,
        1.494560,
        0.071280,
        1.522042,
        67.520190,
        5.729711,
        1.981173,
        0.155233,
    ],
    "S": [
        6.372157,
        1.514347,
        5.154568,
        22.092528,
        1.473732,
        0.061373,
        1.635073,
        55.445176,
        1.209372,
        0.646925,
        0.154722,
    ],
    "Cl": [
        1.446071,
        0.052357,
        6.870609,
        1.193165,
        6.151801,
        18.343416,
        1.750347,
        46.398394,
        0.634168,
        0.401005,
        0.146773,
    ],
    "Ni": [
        13.521865,
        4.077277,
        6.947285,
        0.286763,
        3.866028,
        14.622634,
        2.135900,
        71.966078,
        4.284731,
        0.004437,
        -2.762697,
    ],
    "Cu": [
        14.014192,
        3.738280,
        4.784577,
        0.003744,
        5.056806,
        13.034982,
        1.457971,
        72.554793,
        6.932996,
        0.265666,
        -3.774477,
    ],
    "Pd": [
        6.121511,
        0.062549,
        4.784063,
        0.784031,
        16.631683,
        8.751391,
        4.318258,
        34.489983,
        13.246773,
        0.784031,
        0.883099,
    ],
    "Ag": [
        6.073874,
        0.055333,
        17.155437,
        7.896512,
        4.173344,
        28.443739,
        0.852238,
        110.376108,
        17.988685,
        0.716809,
        0.756603,
    ],
    "Pt": [
        31.273891,
        1.316992,
        18.445441,
        8.797154,
        17.063745,
        0.124741,
        5.555933,
        40.177994,
        1.575270,
        1.316997,
        4.050394,
    ],
    "Au": [
        16.777389,
        0.122737,
        19.317156,
        8.621570,
        32.979682,
        1.256902,
        5.595453,
        38.008821,
        10.576854,
        0.000601,
        -6.279078,
    ],
    "Fe": [
        12.311098,
        5.009415,
        1.876623,
        0.014461,
        3.066177,
        18.743041,
        2.070451,
        82.767874,
        6.975185,
        0.346506,
        -0.304931 
    ],
    "In": [
        6.196477,
        0.042072,
        18.816183,
        6.695665,
        4.050479,
        31.009791,
        1.638929,
        103.284350,
        17.962912,
        0.610714,
        0.333097 
    ],
    "Ga": [
        15.758946,
        3.121754,
        6.841123,
        0.226057,
        4.121016,
        12.482196,
        2.714681,
        66.203622,
        2395246,
        0.007238,
        -0.847395 
    ],

    "K": [
        8.163991,
        12.816323,
        7.146945,
        0.808945,
        1.070140,
        210327009,
        0.877316,
        39.597651,
        1.486434,
        0.052821,
        0.253614 
    ],

    "Mo": [
        6.236218,
        0.090780,
        17.987711,
        1.108310,
        12.973127,
        11.468720,
        3.451426,
        66.684153,
        0.210899,
        0.090780,
        1.108770 
    ]
}

wavelengths = {
    "CuKa1": 1.5405981,
    "CuKa2": 1.54443,
    "CuKb1": 1.39225,
    "WLa1": 1.47642,
    "WLa2": 1.48748,
}


# Support Functions for Spline-based approached, by A. Haldar


def initialize_atomic_form_factor_splines(symbols_list, s) -> dict:

    atomic_form_factor_dict: dict = {}
    for symbol in set(symbols_list):
        atomic_form_factor_dict[symbol] = construct_atomic_form_factor_spline(symbol, s)

    return atomic_form_factor_dict


def construct_atomic_form_factor_spline(symbol: str, s: np.ndarray):
    coeffs = waasmaier[symbol]
    ff = (
        np.sum(
            coeffs[:-1:2] * np.exp(-s[:, None] * s[:, None] * coeffs[1:-1:2]), axis=1
        )
        + coeffs[-1]
    )
    return UnivariateSpline(x=s, y=ff, s=0, k=5)


def pdist_in_chunks(arr1, chunksize = 5000):
    arr1len = arr1.shape[0]
    
    i = 0
    while (i * chunksize) < arr1len:
        j = i
        while (j * chunksize)< arr1len:
            if i == j:
                yield pdist(arr1[i * chunksize: (i + 1) * chunksize])
            else:
                yield cdist(arr1[i * chunksize: (i + 1) * chunksize], arr1[j * chunksize: (j + 1) * chunksize]).ravel()
                
            j += 1
        i += 1

def cdist_in_chunks(arr1, arr2, chunksize = 5000):
    arr1len = arr1.shape[0]
    arr2len = arr2.shape[0]
    
    i = 0
    while (i * chunksize) < arr1len:
        j = 0
        while (j * chunksize) < arr2len:
            yield cdist(arr1[i * chunksize: (i + 1) * chunksize], arr2[j * chunksize: (j + 1) * chunksize]).ravel()
            j += 1
        i += 1