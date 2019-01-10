import os
from ase.clease import BayesianCompressiveSensing
import numpy as np
from scipy.special import polygamma
from copy import deepcopy

# Construct 30 measurements with 400 basis functions
np.random.seed(0)
X = np.random.rand(30, 400)
y = 60.0*X[:, 20] - 80.0*X[:, 2]
fname = "test_bayes_compr_sens.json"

bayes = BayesianCompressiveSensing(fname=fname, output_rate_sec=2, 
                                   maxiter=100)
   
def test_optimize_shape_parameter(bayes):
    bayes.lamb = 1.0
    opt = bayes.optimal_shape_lamb()
    assert abs(np.log(opt/2.0) - polygamma(0, opt/2.0)) < 1E-6

def test_fit(bayes):
    bayes.fit(X, y)

    expected_eci = np.zeros(X.shape[1])
    expected_eci[20] = 60.0
    expected_eci[2] = -80.0
    assert np.allclose(bayes.eci, expected_eci)

def test_sparse_solution(bayes):
    # Check that the algorithm produce sparse
    # solution
    rand_y = np.random.rand(len(y))
    bayes.fit(X, rand_y)

    # The point is that we should not get 30 ECIs
    # just set 10 here
    assert bayes.num_ecis < 10

def test_save_load(bayes):
    bayes.save()

    bayes2 = BayesianCompressiveSensing.load(fname)
    assert bayes == bayes2

test_optimize_shape_parameter(bayes)
test_fit(bayes)
test_sparse_solution(deepcopy(bayes))
test_save_load(bayes)
os.remove(fname)
