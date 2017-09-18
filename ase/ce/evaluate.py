from ase.db import connect
import numpy as np
from numpy.linalg import matrix_rank, inv, lstsq
import matplotlib.pyplot as plt

class Evaluate(object):
    def __init__(self, db_name, cluster_names=None, lamb=0.0, eci=None):
        self.db_name = db_name
        self.lamb = lamb
        self.db = connect(db_name)
        if cluster_names is None:
            self.cluster_names = self._get_cluster_names()
        else:
            self.cluster_names = cluster_names
        self.cf_matrix = self._make_cf_matrix()
        self.cf_matrix = self.reduce_matrix(self.cf_matrix)
        self.e_dft = self._get_dft_energy_per_atom()
        self.eci = eci
        self.e_pred = self._get_e_predict()

    def _get_cluster_names(self):
        cluster_names = self.db.get(name='information').data.cluster_names
        flattened = [item for sublist in cluster_names for item in sublist]
        return flattened

    def _make_cf_matrix(self):
        cf_matrix = []
        # for row in self.db.select(converged=True):
        for row in self.db.select(collapsed=False):
            cf_matrix.append([row[x] for x in self.cluster_names])
        return np.array(cf_matrix, dtype=float)

    @property
    def full_cf_matrix(self):
        cfm = []
        for row in self.db.select([('name','!=','information')]):
            cfm.append([row[x] for x in self.cluster_names])
        cfm = np.array(cfm, dtype=float)
        #cfm = self.reduce_matrix(cfm)
        return cfm

    def _get_dft_energy_per_atom(self):
        e_dft = []
        # for row in self.db.select(converged=True):
        for row in self.db.select(collapsed=False):
            e_dft.append(row.energy/row.natoms)
        return np.array(e_dft)

    def get_eci(self):
        print(self.cf_matrix.shape)
        n_col = self.cf_matrix.shape[1]
        print("# of types of clusters: {}".format(n_col))
#        identity = np.identity(n_col)
#        identity[0][0] = 0.
#        a = inv(self.cf_matrix.T.dot(self.cf_matrix) +
#                self.lamb*identity).dot(self.cf_matrix.T)
        a = inv(self.cf_matrix.T.dot(self.cf_matrix)).dot(self.cf_matrix.T)
        eci = a.dot(self.e_dft)
#        eci = lstsq(self.cf_matrix, self.e_dft)[0]
        #print(eci)
        if self.eci is not None:
            eci = self.eci
        return eci

    def _get_e_predict(self):
        """
        Energy of predicted by the CE model calculated by
        E_{predicted} = cf_matrix * eci
        """
        eci = self.get_eci()
        return self.cf_matrix.dot(eci)

    def plot_energy(self):
        e_pred = self.e_pred
        e_dft = self.e_dft

        rmin = min(np.append(e_dft,e_pred)) - 0.1
        rmax = max(np.append(e_dft,e_pred)) + 0.1

        t = np.arange(rmin-10, rmax+10, 1)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Fit using %d data points' %e_dft.shape[0])
        ax.plot(e_pred, e_dft, 'bo', mfc='none')
        ax.plot(t, t, 'r')
        ax.axis([rmin, rmax, rmin, rmax])
        ax.set_ylabel(r'$E_{DFT}$ (eV/atom)')
        ax.set_xlabel(r'$E_{pred}$ (eV/atom)')
        ax.text(0.95, 0.01, 'CV = %.4f eV/atom \n RMSE = %.4f eV/atom'
                %(self.cv_loo(), self.rmse()),
        # ax.text(0.95, 0.01, 'RMSE = %.4f eV/atom'
                # %(self.rmse()),
                  verticalalignment='bottom',
                  horizontalalignment='right',
                  transform=ax.transAxes,
                  fontsize=12)
        ax.plot(self.e_pred_loo, e_dft, 'ro', mfc='none')
        plt.show()
        return True

    def reduce_matrix(self, cfm):
        cfm = cfm[:, ~np.all(cfm == 0., axis=0)]
        offset = 0
        rank = matrix_rank(cfm)
        while cfm.shape[1] > rank:
            temp = np.delete(cfm,-1-offset,axis=1)
            if matrix_rank(temp) < rank:
                offset += 1
            else:
                cfm = temp
                offset = 0
        self.cf_matrix = cfm
        return cfm

    def get_eci_loo(self, i):
        n_col = self.cf_matrix.shape[1]
        cfm = self.cf_matrix
        cfm = np.delete(cfm, i, 0)
        e_dft = self.e_dft
        e_dft = np.delete(e_dft, i, 0)
        identity = np.identity(n_col)
        identity[0][0] = 0.
        a = inv(cfm.T.dot(cfm) + self.lamb*identity).dot(cfm.T)
        eci = a.dot(e_dft)
        return eci

    def cv_loo(self):
        cv_sq = 0.
        e_pred_loo = []
        for i in range(self.cf_matrix.shape[0]):
            eci = self.get_eci_loo(i)
            e_pred = self.cf_matrix[i][:].dot(eci)
            delta_e = self.e_dft[i] - e_pred
            cv_sq += (delta_e)**2
            e_pred_loo.append(e_pred)
        cv_sq /= self.cf_matrix.shape[0]
        self.e_pred_loo = e_pred_loo
        return np.sqrt(cv_sq)

    def cv_loo_fast(self):
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
        cv_sq /= cf.shape[0]
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
