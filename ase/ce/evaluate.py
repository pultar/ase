"""Module that fits ECIs to energy data."""
import os
import sys
from copy import deepcopy
import numpy as np
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
from ase.utils import basestring
from ase.ce import BulkCrystal, BulkSpacegroup
from ase.db import connect
import multiprocessing as mp
import logging as lg
from ase.ce import MultiprocessHandler

try:
    # dependency on sklearn is to be removed due to some of technical problems
    from sklearn.linear_model import Lasso
    has_sklearn = True
except Exception:
    has_sklearn = False

# Initialize a module wide logger
logger = lg.getLogger(__name__)
logger.setLevel(lg.INFO)


class Evaluate(object):
    """Evaluate RMSE/MAE of the fit and CV scores.

    Arguments:
    =========
    setting: BulkCrystal or BulkSapcegroup object

    cluster_names: list
        Names of clusters to include in the evalutation.
        If None, all of the possible clusters are included.

    select_cond: tuple or list of tuples
        Extra selection condition specified by user.
        Default only includes "converged=True".

    penalty: str
        Type of regularization to be used.
        -*None*: no regularization
        -'lasso' or 'l1': L1 regularization
        -'euclidean' or 'l2': L2 regularization
    """

    def __init__(self, setting, cluster_names=None, select_cond=None,
                 penalty=None):
        """Initialize the Evaluate class."""
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            msg = "setting must be BulkCrystal or BulkSpacegroup object"
            raise TypeError(msg)

        self.setting = setting
        self.cluster_names = setting.cluster_names
        self.num_elements = setting.num_elements

        # Define the selection conditions
        self.select_cond = [('converged', '=', True)]
        if select_cond is not None:
            if isinstance(select_cond, list):
                self.select_cond += select_cond
            else:
                self.select_cond.append(select_cond)

        # Determine the type of penalty for compressed sensing
        if penalty is None or penalty is False:
            self.penalty = False
        elif penalty.lower() == 'lasso' or penalty.lower() == 'l1':
            if not has_sklearn:
                msg = "L1 regularization relies on scikit-learn package, "
                msg += "which was not found..."
                raise ValueError(msg)
            self.penalty = 'l1'
        elif penalty.lower() == 'ridge' or penalty.lower() == 'l2':
            self.penalty = 'l2'
        else:
            raise TypeError("The penalty type, {},".format(penalty) +
                            " is not supported")

        if cluster_names is None:
            self.cluster_names = self.setting.full_cluster_names
        else:
            self.cluster_names = cluster_names
        self.cf_matrix = self._make_cf_matrix()
        self.e_dft = self._get_dft_energy_per_atom()
        self.eci = None
        self.alpha = None
        self.e_pred_loo = None

    def get_eci(self, alpha):
        """Determine and return ECIs for a given alpha.

        This method also saves the last value of alpha used (self.alpha) and
        the corresponding ECIs (self.eci) such that ECIs are not calculated
        repeated if alpha value is unchanged.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        # Check the regression coefficient alpha
        if not isinstance(alpha, (int, float)):

            raise TypeError("The penalty coefficient 'lamb' must be either"
                            " int or float type.")
        self.alpha = float(alpha)

        n_col = self.cf_matrix.shape[1]

        if not self.penalty:
            if matrix_rank(self.cf_matrix) < n_col:
                print("Rank of the design matrix is smaller than the number "
                      "of its columns. Reducing the matrix to make it "
                      "invertible.")
                self._reduce_matrix()
                print("Linearly independent clusters are:")
                print("{}".format(self.cluster_names))
            a = inv(self.cf_matrix.T.dot(self.cf_matrix)).dot(self.cf_matrix.T)
            eci = a.dot(self.e_dft)

        elif self.penalty == 'l2':
            identity = np.identity(n_col)
            identity[0][0] = 0.
            a = inv(self.cf_matrix.T.dot(self.cf_matrix) +
                    alpha * identity).dot(self.cf_matrix.T)
            eci = a.dot(self.e_dft)

        elif self.penalty == 'l1':
            lasso = Lasso(alpha=alpha, fit_intercept=False, copy_X=True,
                          normalize=True, max_iter=1e6)
            lasso.fit(self.cf_matrix, self.e_dft)
            eci = lasso.coef_
            # print('# of nonzero ECIs: {}'.format(len(np.nonzero(eci)[0])))
        else:
            raise ValueError("Unknown penalty type.")

        self.eci = eci
        return eci

    def get_cluster_name_eci(self, alpha, return_type='tuple'):
        """Determine cluster names and their corresponding ECI value.

        Arguments:
        =========
        alpha: int or float
            regularization parameter.

        return_type: str
            'tuple': return an array of cluster_name-ECI tuples.
                     e.g., [(name_1, ECI_1), (name_2, ECI_2)]
            'dict': return a dictionary.
                    e.g., {name_1: ECI_1, name_2: ECI_2}
        """
        if float(alpha) != self.alpha:
            self.get_eci(alpha)
        # sanity check
        if len(self.cluster_names) != len(self.eci):
            raise ValueError('lengths of cluster_names and ECIs are not same')

        i_nonzeros = np.nonzero(self.eci)[0]
        pairs = []
        for i, cname in enumerate(self.cluster_names):
            if i not in i_nonzeros:
                continue
            pairs.append((cname, self.eci[i]))

        if return_type == 'dict':
            return dict(pairs)
        return pairs

    def plot_fit(self, alpha):
        """Plot calculated (DFT) and predicted energies for a given alpha.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        if float(alpha) != self.alpha:
            self.get_eci(alpha)
        e_pred = self.cf_matrix.dot(self.eci)

        rmin = min(np.append(self.e_dft, e_pred)) - 0.1
        rmax = max(np.append(self.e_dft, e_pred)) + 0.1

        t = np.arange(rmin - 10, rmax + 10, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Fit using {} data points'.format(self.e_dft.shape[0]))
        ax.plot(e_pred, self.e_dft, 'bo', mfc='none')
        ax.plot(t, t, 'r')
        ax.axis([rmin, rmax, rmin, rmax])
        ax.set_ylabel(r'$E_{DFT}$ (eV/atom)')
        ax.set_xlabel(r'$E_{pred}$ (eV/atom)')
        ax.text(0.95, 0.01,
                "CV = {0:.3f} meV/atom \n"
                "RMSE = {1:.3f} meV/atom".format(self.cv_loo(alpha) * 1000,
                                                 self.rmse(alpha) * 1000),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=12)
        ax.plot(self.e_pred_loo, self.e_dft, 'ro', mfc='none')
        plt.show()

    def plot_CV(self, alpha_min, alpha_max, num_alpha=10, scale='log',
                logfile=None):
        """Plot CV for a given range of alpha.

        In addition to plotting CV with respect to alpha, logfile can be used
        to extend the range of alpha or add more alpha values in a given range.
        Returns an alpha value that leads to the minimum CV score within the
        pool of evaluated alpha values.

        Arguments:
        =========
        alpha_min: int or float
            minimum value of regularization parameter alpha.

        alpha_max: int or float
            maximum value of regularization parameter alpha.

        num_alpha: int
            number of alpha values to be used in the plot.

        scale: str
            -'log'(default): alpha values are evenly spaced on a log scale.
            -'linear': alpha values are evenly spaced on a linear scale.

        logfile: file object, str or None.
            - None: logging is disabled
            - str: a file with that name will be opened. If '-', stdout used.
            - file object: use the file object for logging

        Note: If the file with the same name exists, it first checks if the
              alpha value already exists in the logfile and evalutes the CV of
              the alpha values that are absent. The newly evaluated CVs are
              appended to the existing file.
        """
        # set up alpha values
        if scale == 'log':
            alphas = np.logspace(np.log10(alpha_min), np.log10(alpha_max),
                                 int(num_alpha), endpoint=True)
        elif scale == 'linear':
            alphas = np.linspace(alpha_min, alpha_max, int(num_alpha),
                                 endpoint=True)

        # logfile setup
        if isinstance(logfile, basestring):
            if logfile == '-':
                handler = lg.StreamHandler(sys.stdout)
                handler.setLevel(lg.INFO)
                logger.addHandler(handler)
            else:
                handler = MultiprocessHandler(logfile)
                handler.setLevel(lg.INFO)
                logger.addHandler(handler)
                # create a log file and make a header line if the file does not
                # exist.
                # if not os.path.isfile(logfile):
                if os.stat(logfile).st_size == 0:
                    logger.info("alpha \t\t # ECI \t CV")
                # if the file exists, read the alpha values that are already
                # evaluated.
                else:
                    existing_alpha = []
                    with open(logfile) as f:
                        next(f)
                        for line in f:
                            existing_alpha.append(float(line.split()[0]))
                    index = []
                    for i, alpha in enumerate(alphas):
                        if np.isclose(existing_alpha, alpha, atol=1e-9).any():
                            index.append(i)
                    # remove redundant alpha values
                    alphas = np.delete(alphas, index)

        # get CV scores
        try:
            nproc = int(max(mp.cpu_count() / 2, 1))
            workers = mp.Pool(nproc)
            args = [(self, alpha) for alpha in alphas]
            cv = workers.map(cv_loo_mp, args)
            cv = np.array(cv)
        except NotImplementedError:
            # NotImplementedError can be raised by mp.cpu_count()
            # In that case execute on one CPU
            cv = np.ones(len(alphas))
            for i, alpha in enumerate(alphas):
                cv[i] = self.cv_loo(alpha)
                num_eci = len(np.nonzero(self.get_eci(alpha))[0])
                logger.info('{:.10f}\t {}\t {:.10f}'.format(alpha, num_eci,
                                                            cv[i]))

        # --------------- #
        # Generate a plot #
        # --------------- #
        # if logfile is present, read all entries from the file
        if logfile:
            alphas = []
            cv = []
            with open(logfile) as log:
                next(log)
                for line in log:
                    alphas.append(float(line.split()[0]))
                    cv.append(float(line.split()[-1]))
                alphas = np.array(alphas)
                cv = np.array(cv)
                # sort alphas and cv based on the values of alphas
                ind = alphas.argsort()
                alphas = alphas[ind]
                cv = cv[ind]

        # get the minimum CV score and the corresponding alpha value
        ind = cv.argmin()
        min_alpha = alphas[ind]
        min_cv = cv[ind]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('CV score vs. alpha')
        ax.semilogx(alphas, cv * 1000)
        ax.semilogx(min_alpha, min_cv * 1000, 'bo', mfc='none')
        ax.set_ylabel('CV score (meV/atom)')
        ax.set_xlabel('alpha')
        ax.text(0.65, 0.01, "min. CV score:\n"
                "alpha = {0:.10f} \n"
                "CV = {1:.3f} meV/atom".format(min_alpha, min_cv * 1000),
                verticalalignment='bottom', horizontalalignment='left',
                transform=ax.transAxes, fontsize=10)
        plt.show()

        return min_alpha

    def mae(self, alpha):
        """Calculate mean absolute error (MAE) of the fit.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        if float(alpha) != self.alpha:
            self.get_eci(alpha)
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        return sum(np.absolute(delta_e)) / len(delta_e)

    def rmse(self, alpha):
        """Calculate root-mean-square error (RMSE) of the fit.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        if float(alpha) != self.alpha:
            self.get_eci(alpha)
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        num_entries = len(delta_e)
        rmse_sq = 0.0
        for i in range(num_entries):
            rmse_sq += (delta_e[i])**2
        rmse_sq /= num_entries
        return np.sqrt(rmse_sq)

    def cv_loo(self, alpha):
        """Determine the CV score for the Leave-One-Out case.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        cv_sq = 0.
        e_pred_loo = []
        for i in range(self.cf_matrix.shape[0]):
            eci = self._get_eci_loo(i, alpha)
            e_pred = self.cf_matrix[i][:].dot(eci)
            delta_e = self.e_dft[i] - e_pred
            cv_sq += (delta_e)**2
            e_pred_loo.append(e_pred)
        cv_sq /= self.cf_matrix.shape[0]
        self.e_pred_loo = e_pred_loo
        return np.sqrt(cv_sq)

    def _get_eci_loo(self, i, alpha):
        """Determine ECI values for the Leave-One-Out case.

        Eliminate the ith row of the cf_matrix when determining the ECIs.
        Returns the determined ECIs.

        Arguments:
        =========
        i: int
            iterator passed from the self._cv_loo method.

        alpha: int or float
            regularization parameter.
        """
        n_col = self.cf_matrix.shape[1]
        cfm = np.delete(self.cf_matrix, i, 0)
        e_dft = np.delete(self.e_dft, i, 0)
        if not self.penalty:
            a = inv(cfm.T.dot(cfm)).dot(cfm.T)
            eci = a.dot(e_dft)
        elif self.penalty == 'l2':
            identity = np.identity(n_col)
            identity[0][0] = 0.
            a = inv(cfm.T.dot(cfm) + alpha * identity).dot(cfm.T)
            eci = a.dot(e_dft)
        elif self.penalty == 'l1':
            lasso = Lasso(alpha=alpha, fit_intercept=False, copy_X=True,
                          normalize=True, max_iter=1e5)
            lasso.fit(cfm, e_dft)
            eci = lasso.coef_
        else:
            raise TypeError("Unknown penalty type.")
        return eci

    def _make_cf_matrix(self):
        """Return a matrix containing the correlation functions.

        Only selects all of the converged structures by default, but further
        constraints can be imposed using *select_cond* argument in the
        initialization step.
        """
        cf_matrix = []
        db = connect(self.setting.db_name)
        for row in db.select(self.select_cond):
            cf_matrix.append([row[x] for x in self.cluster_names])
        return np.array(cf_matrix, dtype=float)

    def _get_dft_energy_per_atom(self):
        """Retrieve DFT energy and convert it to eV/atom unit."""
        e_dft = []
        db = connect(self.setting.db_name)
        for row in db.select(self.select_cond):
            e_dft.append(row.energy / row.natoms)
        return np.array(e_dft)

    def _reduce_matrix(self):
        """Reduce the correlation function matrix.

        Eliminate the columns that are not linearly independent in order to
        match the rank of the matrix. Only used when no regularzation is
        selected (Ordinary Least Squares) where the matrix is not necessarily
        invertible.
        """
        offset = 0
        rank = matrix_rank(self.cf_matrix)
        while self.cf_matrix.shape[1] > rank:
            temp = np.delete(self.cf_matrix, -1 - offset, axis=1)
            cname_temp = deepcopy(self.cluster_names)
            del cname_temp[-1 - offset]
            if matrix_rank(temp) < rank:
                offset += 1
            else:
                self.cf_matrix = temp
                self.cluster_names = deepcopy(cname_temp)
                offset = 0

    def __cv_loo_fast(self, alpha):
        """CV score based on the method in J. Phase Equilib. 23, 348 (2002).

        This method has a computational complexity of order n^1.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        # For each structure i, predict energy based on the ECIs determined
        # using (N-1) structures and the parameters corresponding to the
        # structure i.
        # CV^2 = N^{-1} * Sum((E_DFT-E_pred) / (1 - X_i (X^T X)^{-1} X_u^T))^2
        if float(alpha) != self.alpha:
            self.get_eci(alpha)
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        cfm = self.cf_matrix
        # precision matrix
        prec = inv(cfm.T.dot(cfm))
        cv_sq = 0.0
        for i in range(cfm.shape[0]):
            cv_sq += (delta_e[i] / (1 - cfm[i].dot(prec).dot(cfm[i].T)))**2
        cv_sq /= cfm.shape[0]
        return np.sqrt(cv_sq)


def cv_loo_mp(args):
    """Need to wrap this function in order to use it with multiprocessing.

    Arguments
    =========
    args: Tuple where the first entry is an instance of Evaluate
        and the second is the penalization value
    """
    evaluator = args[0]
    alpha = args[1]
    cv = evaluator.cv_loo(alpha)
    num_eci = len(np.nonzero(evaluator.get_eci(alpha))[0])
    logger.info('{:.10f}\t {}\t {:.10f}'.format(alpha, num_eci, cv))
    return cv
