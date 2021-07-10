"""
This file contains helper functions used by the XrayDebye class.

Also contains routine for calculation of atomic form factors and
X-ray wavelength dict.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.spatial.distance import cdist, pdist

from ase.crystallography.xrddata import waasmaier


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