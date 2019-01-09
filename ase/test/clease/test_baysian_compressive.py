from ase.clease import BayesianCompressiveSensing
import numpy as np
from scipy.special import polygamma
from copy import deepcopy

# Construct 30 measurements with 400 basis functions
np.random.seed(0)
X = np.random.rand(30, 400)
y = 60.0*X[:, 20] - 80.0*X[:, 2]

bayes = BayesianCompressiveSensing(X=X, y=y)
   
def test_optimize_shape_parameter(bayes):
    bayes.lamb = 1.0
    opt = bayes.optimal_shape_lamb()
    assert abs(np.log(opt/2.0) - polygamma(0, opt/2.0)) < 1E-6

def test_fit(bayes):
    bayes.fit(output_rate_sec=2)

    expected_eci = np.zeros(X.shape[1])
    expected_eci[20] = 60.0
    expected_eci[2] = -80.0
    assert np.allclose(bayes.eci, expected_eci)

def test_sparse_solution(bayes):
    # Check that the algorithm produce sparse
    # solution
    rand_y = np.random.rand(len(y))
    bayes.y = rand_y
    bayes.fit(output_rate_sec=2)
    assert bayes.num_ecis == 0

#test_optimize_shape_parameter(bayes)
#test_fit(bayes)
test_sparse_solution(deepcopy(bayes))