from copy import deepcopy
import numpy as np
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
from ase.ce import BulkCrystal, BulkSpacegroup
from ase.db import connect

try:
    # dependency on sklearn is to be removed due to some of technical problems
    from sklearn.linear_model import Lasso
    has_sklearn = True
except:
    has_sklearn = False


class Evaluate(object):
    """Class that evaluates RMSE and CV scores. It also generates a plot that
    compares the energies predicted by CE and that are obtained from DFT
    calculations.

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
        if not isinstance(setting, (BulkCrystal, BulkSpacegroup)):
            raise TypeError("setting must be BulkCrystal or BulkSpacegroup "
                            "object")
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
            if ( not has_sklearn ):
                msg = "At the moment the L1 regularization relies on the "
                msg += "sklearn package, which was not found..."
                raise ValueError( msg )
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

    @property
    def full_cf_matrix(self):
        """Get correlation function of every entry.
        This method is used for evaluating the variance when creating probe
        structures.
        """
        cfm = []
        for row in self.setting.db.select([('name', '!=', 'information')]):
            cfm.append([row[x] for x in self.cluster_names])
        cfm = np.array(cfm, dtype=float)
        return cfm

    def _make_cf_matrix(self):
        """
        Returns the matrix containing the correlation functions.
        Only selects all of the converged structures by default.
        """
        cf_matrix = []
        db = connect(self.setting.db_name)
        for row in db.select(self.select_cond):
            cf_matrix.append([row[x] for x in self.cluster_names])
        return np.array(cf_matrix, dtype=float)

    @property
    def full_cf_matrix(self):
        """
        Get correlation function of every entry (other than the one to store
        information). This method is used for evaluating the variance when
        creating probe structures.
        """
        cfm = []
        db = connect(self.setting.db_name)
        for row in db.select([('name', '!=', 'information')]):
            cfm.append([row[x] for x in self.cluster_names])
        cfm = np.array(cfm, dtype=float)
        return cfm

    def _get_dft_energy_per_atom(self):
        e_dft = []
        db = connect(self.setting.db_name)
        for row in db.select(self.select_cond):
            e_dft.append(row.energy / row.natoms)
        return np.array(e_dft)

    def get_eci(self, alpha):
        """Determine and return ECIs for a given alpha.

        This method also saves the last value of alpha used (self.alpha) and
        the corresponding ECIs (self.eci) such that ECIs are not calculated
        repeated if alpha value is unchanged.

        Argument:
        ========
        alpha: int or float
            regression coefficient.


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
                self._reduce_matrix(self.cf_matrix)
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
            print('Number of nonzero ECIs: {}'.format(len(np.nonzero(eci)[0])))

        else:
            raise ValueError("Unknown penalty type.")

        self.eci = eci
        return eci

    def get_cluster_name_eci(self, alpha, return_type='tuple'):
        """Determine and return the cluster names and their corresponding ECI
        values for the provided alpha (regression coefficient) value.


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

        if type == 'dict':
            return dict(pairs)
        return pairs

    def plot_energy(self, alpha):
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
                "RMSE = {1:.3f} meV/atom".format(self._cv_loo(alpha) * 1000,
                                                 self.rmse(alpha) * 1000),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=12)
        ax.plot(self.e_pred_loo, self.e_dft, 'ro', mfc='none')
        plt.show()
        return True

    def _reduce_matrix(self, cfm):
        """
        Reduces the correlation function matrix (eliminates the columns that
        are not linearly independent) to match the rank of the matrix.
        Changes self.cf_matrix and self.cluster_names directly instead of
        returning the reduced values.
        """
        cname_list = deepcopy(self.cluster_names)
        offset = 0
        rank = matrix_rank(cfm)
        while cfm.shape[1] > rank:
            temp = np.delete(cfm, -1 - offset, axis=1)
            cname_temp = deepcopy(cname_list)
            del cname_temp[-1 - offset]
            if matrix_rank(temp) < rank:
                offset += 1
            else:
                cfm = temp
                cname_list = deepcopy(cname_temp)
                offset = 0
        self.cf_matrix = cfm
        self.cluster_names = cname_list
        return True

    def _get_eci_loo(self, i, alpha):
        """
        Determines the ECIs for Leave-one-out case. Eliminates the ith row of
        the cf_matrix when determining the ECIs.
        Returns the determined ECIs.
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

    def _cv_loo(self, alpha):
        """Determines the CV score for the Leave-One-Out case.

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

    def __cv_loo_fast(self, alpha):
        """Calculate cross-validation (CV) score of the model based on the
        method presented in J. Phase Equilib. 23, 348 (2002).
        This method has a computational complexity of order n^1.
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

    def mae(self, alpha):
        if float(alpha) != self.alpha:
            self.get_eci(alpha)
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        return sum(np.absolute(delta_e)) / len(delta_e)

    def rmse(self, alpha):
        """Generate a RMSE of the fit.

        Arguments:
        =========
        alpha: int or float
            regression coefficient.
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
