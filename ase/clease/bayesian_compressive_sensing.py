import numpy as np
from scipy.special import polygamma
from scipy.optimize import brentq
import time
import json

class BayesianCompressiveSensing(object):
    def __init__(self, shape_var=0.5, rate_var=0.5, shape_lamb=0.5, 
                 variance_opt_start=100, fname="bayes_compr_sens.json"):
        self.shape_var = shape_var
        self.rate_var = rate_var
        self.shape_lamb = shape_lamb
        self.variance_opt_start = variance_opt_start
        self.fname = fname
        self.X = None
        self.y = None

        self.gammas = None
        self.eci = None
        self.inv_variance = None
        self.lamb = 1E-6
        self.inverse_sigma = None
        self.current_mu = None

        # Quantities used for fast updates
        self.S = None
        self.Q = None
        self.ss = None
        self.qq = None

    def _initialize(self):
        num_features = self.X.shape[1]
        if self.gammas is None:
            self.gammas = np.zeros(num_features)
        self.eci = np.zeros_like(self.gammas)

        if self.inv_variance is None and self.y is not None:
            self.inv_variance = 0.01*np.var(self.y)
        
        if self.lamb is None:
            self.lamb = 1E-6

        self.inverse_sigma = np.zeros((num_features, num_features))
        self.current_mu = np.zeros(num_features)

        if self.X is not None:
            # Quantities used for fast updates
            self.S = np.diag(self.inv_variance*self.X.T.dot(self.X))
            self.Q = self.X.T.dot(self.y)
            self.ss = self.S/(1.0 - self.gammas*self.S)
            self.qq = self.Q/(1.0 - self.gammas*self.S)

    def mu(self):
        sel = self.selected
        return self.inv_variance*self.inverse_sigma.dot(self.X[:, sel].T.dot(self.y))

    def optimal_gamma(self, indx):
        s = self.ss[indx]
        qsq = self.qq[indx]**2
        term1 = -s*(s + 2*self.lamb)

        delta = (s+2*self.lamb)**2 - 4*self.lamb*(s - qsq + self.lamb)
        assert delta >= 0.0

        term2 = s*np.sqrt(delta)
        gamma = (term1 + term2)/(2*self.lamb*s**2)
        return gamma

    def optimal_lamb(self):
        N = self.X.shape[1]
        return (N - 1 + 0.5*self.shape_lamb)/(0.5*np.sum(self.gammas) + self.shape_lamb*0.5)

    def optimal_inv_variance(self):
        N = self.X.shape[1]
        a = 1.0
        b = 0.0
        mse = np.sum((self.y - self.X.dot(self.eci)**2))
        return (0.5*N + a)/(0.5*mse + b)

    def optimal_shape_lamb(self):
        res = brentq(shape_parameter_equation, 1E-30, 1E100, args=(self.lamb,), maxiter=10000)
        return res

    def sherman_morrison(self, A_inv, u, v):
        return A_inv - A_inv.dot(np.outer(u, v)).dot(A_inv)/(1 + v.T.dot(A_inv).dot(u))

    def is_included(self, n):
        return self.gammas[n] > 0.0

    def update_quantities(self):
        sel = self.selected
        X_sel = self.X[:, sel]
        prec = X_sel.dot(self.inverse_sigma).dot(X_sel.T)


        self.S = np.diag(self.inv_variance*self.X.T.dot(self.X) - self.inv_variance**2 * self.X.T.dot(prec).dot(self.X))
        self.Q = self.inv_variance*self.X.T.dot(self.y) - self.inv_variance**2 * self.X.T.dot(prec).dot(self.y)

        self.ss = self.S/(1.0 - self.gammas*self.S)
        self.qq = self.Q/(1.0 - self.gammas*self.S)

    @property
    def selected(self):
        return np.argwhere(self.gammas > 0.0)[:, 0]

    def update_sigma_mu(self, n, gamma, include=True):
        """Update sigma and mu."""
        X_sel = self.X[:, self.selected]
        self.inverse_sigma = np.linalg.inv(self.inv_variance*X_sel.T.dot(X_sel) + np.diag(1.0/self.gammas[self.selected]))
        self.current_mu[self.selected] = self.mu()

    def get_basis_function_index(self, select_strategy):

        if select_strategy == "random":
            return np.random.randint(low=0, high=len(self.gammas))
        elif select_strategy == "max_increase":
            return self._get_bf_with_max_increase()

    def _get_bf_with_max_increase(self):
        new_gammas = np.array([self.optimal_gamma(i) for i in range(len(self.gammas))])
        new_gammas[new_gammas < 0.0] = 0.0
        l = np.log(1/(1+new_gammas*self.ss)) + self.qq**2*new_gammas/(1+new_gammas*self.ss) - \
                self.lamb*new_gammas
        return np.argmax(l)

    def obtain_ecis(self):
        """Update the ECIs."""

        if len(self.selected) == 0:
            return
        X_sel = self.X[:, self.selected]
        self.eci[self.selected] = np.linalg.inv(X_sel.T.dot(X_sel)).dot(X_sel.T).dot(self.y)

    def log_posterior_mass(self):
        
        if len(self.selected) == 0:
            return 0.0
        sel = self.selected
        diff = self.current_mu[sel] - self.X[:, sel].dot(self.eci[sel])
        z = diff.dot(self.inverse_sigma.dot(diff))
        return z

    def rmse(self):
        indx = self.selected
        pred = self.X[:, indx].dot(self.eci[indx])
        return np.sqrt(np.mean((pred - self.y)**2))

    def log(self, msg):
        print(msg)

    @property
    def num_ecis(self):
        return np.count_nonzero(self.gammas)

    def to_dict(self):
        data = {}
        data["inv_variance"] = self.inv_variance
        data["gammas"] = self.gammas.tolist()
        data["shape_var"] = self.shape_var
        data["rate_var"] = self.rate_var
        data["shape_lamb"] = self.shape_lamb
        data["lamb"] = self.lamb
        return data

    def save(self):
        """Save the results from file."""
        with open(self.fname, 'w') as outfile:
            json.dump(self.to_dict(), outfile)
        print("Backup data written to {}".format(self.fname))

    @staticmethod
    def load(fname):
        bayes = BayesianCompressiveSensing()
        bayes.fname = fname
        with open(fname, 'r') as infile:
            data = json.load(infile)
        
        bayes.inv_variance = data["inv_variance"]
        bayes.gammas = np.array(data["gammas"])
        bayes.shape_var = data["shape_var"]
        bayes.rate_var = data["rate_var"]
        bayes.shape_lamb = data["shape_lamb"]
        bayes.lamb = data["lamb"]
        return bayes
    
    def __eq__(self, other):
        equal = True

        # Required fields to be equal if two objects
        # should be considered equal
        items = ["fname", "gammas", "inv_variance", "lamb",
                 "shape_var", "rate_var", "shape_lamb", "lamb"]
        for k in items:
            v = self.__dict__[k]
            if isinstance(v, np.ndarray):
                equal = equal and np.allclose(v, other.__dict__[k])
            elif isinstance(v, float):
                equal = equal and abs(v - other.__dict__[k]) < 1E-6
            else:
                equal = equal and (v == other.__dict__[k])
        return equal

    def fit(self, X, y, min_change=1E-8, maxiter=100000, output_rate_sec=10,
            select_strategy="max_increase"):
        allowed_strategies = ["random", "max_increase"]

        if select_strategy not in allowed_strategies:
            raise ValueError("select_strategy has to be one of {}"
                             "".format(allowed_strategies))

        self.X = X
        self.y = y
        self._initialize()

        is_first = True
        iteration = 0
        now = time.time()
        while iteration < maxiter:
            if time.time() - now > output_rate_sec:
                msg = "Iter: {} ".format(iteration)
                msg += "RMSE: {:3E} ".format(1000.0*self.rmse())
                msg += "Num ECI: {}".format(self.num_ecis)
                self.log(msg)
                now = time.time()

            iteration += 1
            already_excluded = False

            if is_first:
                indx = np.argmax(self.qq**2 - self.ss)
                is_first = False
            else:
                indx = self.get_basis_function_index(select_strategy)

            gamma = self.optimal_gamma(indx)
            if gamma > 0.0:
                self.gammas[indx] = gamma
                include = True
            else:
                gamma = self.gammas[indx]

                if abs(gamma) < 1E-6:
                    already_excluded = True

                self.gammas[indx] = 0.0
                self.eci[indx] = 0.0
                self.current_mu[indx] = 0.0
                include = False

            if already_excluded:
                continue
            
            self.update_sigma_mu(indx, gamma, include=include)
            self.update_quantities()
            self.lamb = self.optimal_lamb()
            self.shape_lamb = self.optimal_shape_lamb()

            if iteration > self.variance_opt_start:
                self.inv_variance = self.optimal_inv_variance()
            self.obtain_ecis()

    def show_shape_parameter(self):
        from matplotlib import pyplot as plt
        x = np.logspace(-10, 10)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, shape_parameter_equation(x, self.lamb))
        ax.axhline(0, ls="--")
        ax.set_xscale("log")
        plt.show()


def shape_parameter_equation(x, lamb):
    return np.log(x/2.0) + 1 - polygamma(0, x/2) + np.log(lamb) - lamb