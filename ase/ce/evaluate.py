from ase.db import connect
from ase.ce.settings import BulkCrystal
import numpy as np
from numpy.linalg import matrix_rank, inv
import matplotlib.pyplot as plt
from copy import deepcopy

# dependency on sklearn is to be removed due to some of technical problems
from sklearn.linear_model import Lasso


class Evaluate(object):
    """
    Class that evaluates RMSE and CV scores. It also generates a plot that
    compares the energies predicted by CE and that are obtained from DFT
    calculations.
    """
    def __init__(self, BC, cluster_names=None, select_cond = None, lamb=0.0,
                 penalty=None):
        """
        BC: BulkCrystal object
        cluster_names: names of clusters to include in the evalutation.
                       If None, all of the possible clusters are included.
        select_cond: extra selection condition specified by user. Default only
                     includes "converged=True".
        lamb: regression coefficient lambda
        penalty: type of regularization (compressed sensing).
                 None - no regularization
                 'lasso' or 'l1' - L1 regularization
                 'euclidean' or 'l2' - L2 regularization
        """
        if type(BC) is not BulkCrystal:
            raise TypeError("Passed object should be BulkCrystal type")
        self.db_name = BC.db_name
        self.cluster_names = BC.cluster_names
        self.num_elements = BC.num_elements

        # Define the selection conditions
        self.select_cond = [('converged', '=', True)]
        if select_cond is not None:
            for cond in select_cond:
                self.select_cond.append(cond)

        # Check the regression coefficient lambda
        if type(lamb) is not int and type(lamb) is not float:
            raise TypeError("The penalty coefficient 'lamb' must be either"
                            " int or float type.")
        self.lamb = float(lamb)

        # Determine the type of penalty for compressed sensing
        if penalty is None or penalty is False:
            self.penalty = False
        elif penalty.lower() == 'lasso' or penalty.lower() == 'l1':
            self.penalty = 'l1'
        elif penalty.lower() == 'euclidean' or penalty.lower() == 'l2':
            self.penalty = 'l2'
        else:
            raise TypeError("The penalty type, {},".format(penalty) +
                            " is not supported")

        self.db = connect(self.db_name)
        if cluster_names is None:
            self.cluster_names = self._get_full_cluster_names()
        else:
            self.cluster_names = cluster_names
        self.cf_matrix = self._make_cf_matrix()
        self.eci = None
        self.e_dft = None
        self.e_pred = None

    def _get_full_cluster_names(self):
        """
        Returns the all possible cluster names.
        Used only when the cluster_names is None.
        """
        full_names = []
        full_names.append(self.cluster_names[0][0])
        for k in range(1,len(self.cluster_names)):
            cases = (self.num_elements-1)**k
            for name in self.cluster_names[k][:]:
                for i in range(1,cases+1):
                    full_names.append('{}_{}'.format(name,i))
        return full_names

    def _make_cf_matrix(self):
        """
        Returns the matrix containing the correlation functions.
        Only selects all of the converged structures by default.
        """
        cf_matrix = []
        for row in self.db.select(self.select_cond):
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
        for row in self.db.select([('name','!=','information')]):
            cfm.append([row[x] for x in self.cluster_names])
        cfm = np.array(cfm, dtype=float)
        #cfm = self.reduce_matrix(cfm)
        return cfm

    def _get_dft_energy_per_atom(self):
        e_dft = []
        for row in self.db.select(self.select_cond):
            e_dft.append(row.energy/row.natoms)
        self.e_dft = np.array(e_dft)
        return True

    @property
    def get_eci(self):
        if self.e_dft is None:
            self._get_dft_energy_per_atom()

        n_col = self.cf_matrix.shape[1]

        if not self.penalty:
            if matrix_rank(self.cf_matrix) < n_col:
                print("Rank of the design matrix is smaller than the number of "
                      "its columns. Reducing the matrix to make it invertible.")
                self._reduce_matrix(self.cf_matrix)
                print("Linearly independent clusters are:")
                print("{}".format(self.cluster_names))
            a = inv(self.cf_matrix.T.dot(self.cf_matrix)).dot(self.cf_matrix.T)
            eci = a.dot(self.e_dft)

        elif self.penalty == 'l2':
            identity = np.identity(n_col)
            identity[0][0] = 0.
            a = inv(self.cf_matrix.T.dot(self.cf_matrix) +
                    self.lamb*identity).dot(self.cf_matrix.T)
            eci = a.dot(self.e_dft)

        elif self.penalty == 'l1':
            lasso = Lasso(alpha=self.lamb, fit_intercept=False, copy_X=True,
                          normalize=True, max_iter=1e5)
            lasso.fit(self.cf_matrix, self.e_dft)
            eci = lasso.coef_
            print('Number of nonzero ECIs: {}'.format(len(np.nonzero(eci)[0])))

        else:
            raise TypeError("Unknown penalty type.")

        self.eci = eci
        return eci

    @property
    def get_cluster_name_eci_tuple(self):
        if self.eci is None:
            self.get_eci
        # sanity check
        if len(self.cluster_names) != len(self.eci):
            raise ValueError('lengths of cluster_names and ECIs are not same')

        i_nonzeros = np.nonzero(self.eci)[0]
        tuples = []
        for i, cname in enumerate(self.cluster_names):
            if i not in i_nonzeros:
                continue
            tuples.append((cname, self.eci[i]))
        return tuples

    @property
    def get_cluster_name_eci_dict(self):
        if self.eci is None:
            self.get_eci
        # sanity check
        if len(self.cluster_names) != len(self.eci):
            raise ValueError('lengths of cluster_names and ECIs are not same')

        i_nonzeros = np.nonzero(self.eci)[0]
        dict = {}
        for i, cname in enumerate(self.cluster_names):
            if i not in i_nonzeros:
                continue
            dict[cname] = self.eci[i]
        return dict

    def _get_e_predict(self):
        """
        Energy predicted by the CE model.
        E_{predicted} = cf_matrix * eci
        """
        if self.eci is None:
            self.get_eci
        self.e_pred = self.cf_matrix.dot(self.eci)
        return True

    def plot_energy(self):
        if self.e_pred is None:
            self._get_e_predict()

        rmin = min(np.append(self.e_dft, self.e_pred)) - 0.1
        rmax = max(np.append(self.e_dft, self.e_pred)) + 0.1

        t = np.arange(rmin-10, rmax+10, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Fit using %d data points' %self.e_dft.shape[0])
        ax.plot(self.e_pred, self.e_dft, 'bo', mfc='none')
        ax.plot(t, t, 'r')
        ax.axis([rmin, rmax, rmin, rmax])
        ax.set_ylabel(r'$E_{DFT}$ (eV/atom)')
        ax.set_xlabel(r'$E_{pred}$ (eV/atom)')
        ax.text(0.95, 0.01, 'CV = %.4f eV/atom \n RMSE = %.4f eV/atom'
                %(self._cv_loo(), self.rmse()),
                  verticalalignment='bottom',
                  horizontalalignment='right',
                  transform=ax.transAxes,
                  fontsize=12)
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
        # cfm = cfm[:, ~np.all(cfm == 0., axis=0)]
        offset = 0
        rank = matrix_rank(cfm)
        while cfm.shape[1] > rank:
            temp = np.delete(cfm,-1-offset,axis=1)
            cname_temp = deepcopy(cname_list)
            del cname_temp[-1-offset]
            if matrix_rank(temp) < rank:
                offset += 1
            else:
                cfm = temp
                cname_list = deepcopy(cname_temp)
                offset = 0
        self.cf_matrix = cfm
        self.cluster_names = cname_list
        return True

    def _get_eci_loo(self, i):
        """
        Determines the ECIs for Leave-one-out case. Eliminates the ith row of
        the cf_matrix when determining the ECIs.
        Returns the determined ECIs.
        """
        if self.e_dft is None:
            self._get_dft_energy_per_atom()
        n_col = self.cf_matrix.shape[1]
        cfm = np.delete(self.cf_matrix, i, 0)
        e_dft = np.delete(self.e_dft, i, 0)
        if not self.penalty:
            a = inv(cfm.T.dot(cfm)).dot(cfm.T)
            eci = a.dot(e_dft)
        elif self.penalty == 'l2':
            identity = np.identity(n_col)
            identity[0][0] = 0.
            a = inv(cfm.T.dot(cfm) + self.lamb*identity).dot(cfm.T)
            eci = a.dot(e_dft)
        elif self.penalty == 'l1':
            lasso = Lasso(alpha=self.lamb, fit_intercept=False, copy_X=True,
                          normalize=True, max_iter=1e5)
            lasso.fit(cfm, e_dft)
            eci = lasso.coef_
        else:
            raise TypeError("Unknown penalty type.")
        return eci

    def _cv_loo(self):
        """
        Determines the CV score for the Leave-One-Out case.
        """
        cv_sq = 0.
        e_pred_loo = []
        for i in range(self.cf_matrix.shape[0]):
            eci = self._get_eci_loo(i)
            e_pred = self.cf_matrix[i][:].dot(eci)
            delta_e = self.e_dft[i] - e_pred
            cv_sq += (delta_e)**2
            e_pred_loo.append(e_pred)
        cv_sq /= self.cf_matrix.shape[0]
        self.e_pred_loo = e_pred_loo
        return np.sqrt(cv_sq)

    def __cv_loo_fast(self):
        """
        Calculate cross-validation (CV) score of the model based on the method
        presented in J. Phase Equilib. 23, 348 (2002).
        This method has a computational complexity of order n^1.
        """
        # For each structure i, predict energy based on the ECIs determined
        # using (N-1) structures and the parameters corresponding to the
        # structure i.
        # CV^2 = N^{-1} * Sum((E_DFT-E_pred) / (1 - X_i (X^T X)^{-1} X_u^T))^2
        delta_e = self.e_dft - self.e_pred
        cfm = self.cf_matrix
        # precision matrix
        prec = inv(cfm.T.dot(cfm))
        cv_sq = 0.0
        for i in range(cfm.shape[0]):
            cv_sq += (delta_e[i]/(1 - cfm[i].dot(prec).dot(cfm[i].T)))**2
        cv_sq /= cfm.shape[0]
        return np.sqrt(cv_sq)

    def mae(self):
        delta_e = self.e_dft - self.e_pred
        return sum(np.absolute(delta_e))/len(delta_e)

    def rmse(self):
        delta_e = self.e_dft - self.e_pred
        rmse_sq = 0.0
        for i in range(len(delta_e)):
            rmse_sq += (delta_e[i])**2
        rmse_sq /= len(delta_e)
        return np.sqrt(rmse_sq)
