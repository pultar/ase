"""Collection of classess to perform regression."""
import numpy as np
from numpy.linalg import inv, pinv


class LinearRegression(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        """Fit a linear model by performing ordinary least squares

        y = Xc
        Arguments
        ==========
        X: Design matrix (NxM)
        y: Datapints (vector of length N)
        """
        precision = pinv(X.T.dot(X))
        coeff = precision.dot(X.T.dot(y))
        return coeff

    @staticmethod
    def get_instance_array():
        return [LinearRegression()]

    def is_scalar(self):
        return False

    def get_scalar_parameter(self):
        raise ValueError("Fitting scheme is not described by a scalar "
                         "parameter!")


class Tikhonov(LinearRegression):
    """Ridge regularization.

    Arguments:
    =========
    alpha: float, 1D or 2D numpy array
        regularization term
        - float: A single regularization coefficient is used for all features.
                 Tikhonov matrix is T = alpha * I (I = identity matrix).
        - 1D array: Regularization coefficient is defined for each feature.
                    Tikhonov matrix is T = diag(alpha) (the alpha values are
                    put on the diagonal).
                    The length of array should match the number of features.
        - 2D array: Full Tikhonov matrix supplied by a user.
                    The dimensions of the matrix should be M * M where M is the
                    number of features.
    """
    def __init__(self, alpha=1E-5):
        LinearRegression.__init__(self)
        self.alpha = alpha

    def _get_tikhonov_matrix(self, num_clusters):
        if isinstance(self.alpha, np.ndarray):
            if len(self.alpha.shape) == 1:
                tikhonov = np.diag(self.alpha)
            elif len(self.alpha.shape) == 2:
                tikhonov = self.alpha
            else:
                raise ValueError("Matrix have to have dimension 1 or 2")
        else:
            # Alpha is a floating point number
            tikhonov = np.identity(num_clusters)
            tikhonov[0, 0] = 0.0
            tikhonov *= np.sqrt(self.alpha)
        return tikhonov

    def fit(self, X, y):
        """Fit coefficients based on Ridge regularizeation."""
        num_features = X.shape[1]
        tikhonov = self._get_tikhonov_matrix(num_features)

        # Make sure that the tikhonov matrix has the correct dimention
        if tikhonov.shape != (num_features, num_features):
            raise ValueError("The dimensions of Tikhonov matrix do not match "
                             "the number of clusters!")
        precision = inv(X.T.dot(X) + tikhonov.T.dot(tikhonov))
        coeff = precision.dot(X.T.dot(y))
        return coeff

    @staticmethod
    def get_instance_array(alpha_min, alpha_max, num_alpha=10, scale='log'):
        if scale == 'log':
            alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                                int(num_alpha), endpoint=True)
        else:
            alpha = np.linspace(alpha_min, alpha_max, int(num_alpha),
                                endpoint=True)
        return [Tikhonov(alpha=a) for a in alpha]

    def is_scalar(self):
        return isinstance(self.alpha, float)

    def get_scalar_parameter(self):
        if self.is_scalar():
            return self.alpha
        LinearRegression.get_scalar_parameter(self)


class Lasso(LinearRegression):
    """LASSO regularization.

    Arguments:
    =========
    alpha: float
        regularization coefficient
    """
    def __init__(self, alpha=1E-5):
        LinearRegression.__init__(self)
        self.alpha = alpha

    def fit(self, X, y):
        """Fit coefficients based on LASSO regularizeation."""
        from sklearn.linear_model import Lasso
        lasso = Lasso(alpha=self.alpha, fit_intercept=False, copy_X=True,
                      normalize=True, max_iter=1e6)
        lasso.fit(X, y)
        return lasso.coef_

    @staticmethod
    def get_instance_array(alpha_min, alpha_max, num_alpha=10, scale='log'):
        if scale == 'log':
            alpha = np.logspace(np.log10(alpha_min), np.log10(alpha_max), int(num_alpha),
                                endpoint=True)
        else:
            alpha = np.linspace(alpha_min, alpha_max, int(num_alpha),
                                endpoint=True)
        return [Lasso(alpha=a) for a in alpha]

    def is_scalar(self):
        return True

    def get_scalar_parameter(self):
        return self.alpha
