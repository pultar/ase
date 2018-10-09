"""Module that fits ECIs to energy data."""
import os
import sys
import numpy as np
import multiprocessing as mp
import logging as lg
import json
from ase.utils import basestring
from ase.clease import CEBulk, CECrystal
from ase.clease.mp_logger import MultiprocessHandler
from ase.db import connect


# Initialize a module-wide logger
logger = lg.getLogger(__name__)
logger.setLevel(lg.INFO)


class Evaluate(object):
    """Evaluate RMSE/MAE of the fit and CV scores.

    Arguments:
    =========
    setting: CEBulk or BulkSapcegroup object

    cluster_names: list
        Names of clusters to include in the evalutation.
        If None, all of the possible clusters are included.

    select_cond: tuple or list of tuples
        Extra selection condition specified by user.
        Default only includes "converged=True".

    max_cluster_size: int
        Maximum number of atoms in the cluster to include in the fit.
        If this is None then all clusters in the DB are included.

    max_cluster_size: int
        maximum number of atoms in the cluster to include in the fit.
        If *None*, no restriction on the number of atoms will be imposed.

    max_cluster_dia: float or int
        maximum diameter of the cluster (in angstrom) to include in the fit.
        If *None*, no restriction on the diameter.

    scoring_scheme: str
        should be one of 'loocv' or 'loocv_fast'
    """

    def __init__(self, setting, cluster_names=None, select_cond=None,
                 parallel=False, num_core="all", fitting_scheme="ridge",
                 alpha=1E-5, max_cluster_size=None, max_cluster_dia=None,
                 scoring_scheme='loocv'):
        """Initialize the Evaluate class."""
        if not isinstance(setting, (CEBulk, CECrystal)):
            msg = "setting must be CEBulk or CECrystal object"
            raise TypeError(msg)

        self.setting = setting
        self.cluster_names = setting.cluster_names
        self.num_elements = setting.num_elements
        self.scoring_scheme = scoring_scheme
        if max_cluster_size is None:
            self.max_cluster_size = self.setting.max_cluster_size
        else:
            self.max_cluster_size = max_cluster_size
        if max_cluster_dia is None:
            self.max_cluster_dia = self.setting.max_cluster_dia
        else:
            self.max_cluster_dia = self._get_max_cluster_dia(max_cluster_dia)

        self.scheme = None
        self.scheme_string = None
        self.set_fitting_scheme(fitting_scheme, alpha)
        # Define the selection conditions
        self.select_cond = [('converged', '=', True)]
        if select_cond is not None:
            if isinstance(select_cond, list):
                self.select_cond += select_cond
            else:
                self.select_cond.append(select_cond)

        if cluster_names is None:
            self.cluster_names = self.setting.cluster_names
        else:
            self.cluster_names = cluster_names

        # Remove the cluster names that correspond to clusters larger than the
        # specified size and diameter.
        self._filter_cluster_name()

        self.cf_matrix = self._make_cf_matrix()
        self.e_dft, self.names = self._get_dft_energy_per_atom()
        self.multiplicity_factor = self.setting.multiplicity_factor
        self.eci = None
        self.alpha = None
        self.e_pred_loo = None
        self.parallel = parallel
        if parallel:
            if num_core == "all":
                self.num_core = int(mp.cpu_count() / 2)
            else:
                self.num_core = int(num_core)

    def set_fitting_scheme(self, fitting_scheme="ridge", alpha=1E-9):
        from ase.clease.regression import LinearRegression
        allowed_fitting_schemes = ["ridge", "tikhonov", "lasso", "l1", "l2"]
        if isinstance(fitting_scheme, LinearRegression):
            self.scheme = fitting_scheme
        elif isinstance(fitting_scheme, str):
            fitting_scheme = fitting_scheme.lower()
            self.scheme_string = fitting_scheme
            if fitting_scheme not in allowed_fitting_schemes:
                raise ValueError("Fitting scheme has to be one of "
                                 "{}".format(allowed_fitting_schemes))
            if fitting_scheme in ["ridge", "tikhonov", "l2"]:
                from ase.clease.regression import Tikhonov
                self.scheme = Tikhonov(alpha=alpha)
            elif fitting_scheme in ["lasso", "l1"]:
                from ase.clease.regression import Lasso
                self.scheme = Lasso(alpha=alpha)
            else:
                # Perform ordinary least squares
                self.scheme = LinearRegression()
        else:
            raise ValueError("Fitting scheme has to be one of {} "
                             "or an LinearRegression instance."
                             "".format(allowed_fitting_schemes))

        # If the fitting scheme is changed, the ECIs computed are no
        # longer consistent with the scheme
        # By setting it to None, a new calculation is performed
        # when the ECIs are requested
        self.eci = None

    def _get_max_cluster_dia(self, max_cluster_dia):
        """Make max_cluster_dia in a numpy array form."""
        if isinstance(max_cluster_dia, (list, np.ndarray)):
            if len(max_cluster_dia) == self.max_cluster_size + 1:
                for i in range(2):
                    max_cluster_dia[i] = 0.
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
            elif len(max_cluster_dia) == self.max_cluster_size - 1:
                max_cluster_dia = np.array(max_cluster_dia, dtype=float)
                max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])
            else:
                raise ValueError("Invalid length for max_cluster_dia.")
        # max_cluster_dia is int or float
        elif isinstance(max_cluster_dia, (int, float)):
            max_cluster_dia *= np.ones(self.max_cluster_size - 1,
                                       dtype=float)
            max_cluster_dia = np.insert(max_cluster_dia, 0, [0., 0.])

        return max_cluster_dia

    def get_eci(self):
        """Determine and return ECIs for a given alpha.

        This method also saves the last value of alpha used (self.alpha) and
        the corresponding ECIs (self.eci) such that ECIs are not calculated
        repeated if alpha value is unchanged.
        """
        self.eci = self.scheme.fit(self.cf_matrix, self.e_dft)
        return self.eci

    def get_cluster_name_eci(self, return_type='tuple'):
        """Determine cluster names and their corresponding ECI value.

        Arguments:
        =========
        return_type: str
            'tuple': return an array of cluster_name-ECI tuples.
                     e.g., [(name_1, ECI_1), (name_2, ECI_2)]
            'dict': return a dictionary.
                    e.g., {name_1: ECI_1, name_2: ECI_2}
        """
        self.get_eci()

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

    def save_cluster_name_eci(self, fitting_scheme="ridge", alpha=1E-5,
                              filename='cluster_eci.json'):
        """Determine cluster names and their corresponding ECI value.

        Arguments:
        =========
        alpha: int or float
            regularization parameter.

        return_type: str
            the file name should end with either .json or .txt.
        """
        eci_dict = self.get_cluster_name_eci(
            fitting_scheme=fitting_scheme, alpha=alpha, return_type='dict')

        extension = filename.split(".")[-1]

        if extension == 'json':
            with open(filename, 'w') as outfile:
                json.dump(eci_dict, outfile, indent=2, separators=(",", ": "))
        elif extension == 'txt':
            with open(filename, 'r') as outfile:
                outfile.write(eci_dict)
        else:
            raise TypeError('extension {} is not supported'.format(extension))

    def plot_fit(self, fitting_scheme="ridge", alpha=1E-5, interactive=True):
        """Plot calculated (DFT) and predicted energies for a given alpha.

        Argument:
        ========
        alpha: int or float
            regularization parameter.
        """
        import matplotlib.pyplot as plt
        from ase.clease.interactive_plot import ShowStructureOnClick

        if self.eci is None:
            self.get_eci(fitting_scheme=fitting_scheme, alpha=alpha)
        e_pred = self.cf_matrix.dot(self.eci)

        rmin = min(np.append(self.e_dft, e_pred)) - 0.1
        rmax = max(np.append(self.e_dft, e_pred)) + 0.1

        cv = None
        if self.scoring_scheme == "loocv":
            cv = self.loocv()*1000
        elif self.scoring_scheme == "loocv_fast":
            cv = self.loocv_fast()*1000
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
                "RMSE = {1:.3f} meV/atom".format(cv, self.rmse() * 1000),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax.transAxes, fontsize=12)
        ax.plot(self.e_pred_loo, self.e_dft, 'ro', mfc='none')
        if interactive:
            lines = ax.get_lines()
            data_points = [lines[0], lines[2]]
            annotations = [self.names, self.names]
            db_name = self.setting.db_name
            ShowStructureOnClick(fig, ax, data_points, annotations, db_name)
        else:
            plt.show()

    def plot_CV(self, alpha_min=1E-7, alpha_max=1.0, num_alpha=10, scale='log',
                logfile=None, fitting_schemes=None):
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

        fitting_schemes: None or array of instance of LinearRegression

        Note: If the file with the same name exists, it first checks if the
              alpha value already exists in the logfile and evalutes the CV of
              the alpha values that are absent. The newly evaluated CVs are
              appended to the existing file.
        """
        import matplotlib.pyplot as plt
        from ase.clease.regression import LinearRegression

        if fitting_schemes is None:
            if self.scheme_string is None:
                raise ValueError("No fitting scheme supplied!")
            if self.scheme_string in ["lasso", "l1"]:
                from ase.clease.regression import Lasso
                fitting_schemes = Lasso.get_instance_array(
                    alpha_min, alpha_max, num_alpha=num_alpha, scale=scale)
            elif self.scheme_string in ["ridge", "l2", "tikhonov"]:
                from ase.clease.regression import Tikhonov
                fitting_schemes = Tikhonov.get_instance_array(
                    alpha_min, alpha_max, num_alpha=num_alpha, scale=scale)

        for scheme in fitting_schemes:
            if not isinstance(scheme, LinearRegression):
                raise TypeError("Each entry in fitting_schemes should be an "
                                "instance of LinearRegression")
            elif not scheme.is_scalar():
                raise TypeError("plot_CV only supports the fitting schemes "
                                "with a scalar paramater.")

        # if the file exists, read the alpha values that are already evaluated.
        self._initialize_logfile(logfile)
        fitting_schemes = self._remove_existing_alphas(logfile,
                                                       fitting_schemes)

        # get CV scores
        alphas = []
        if self.parallel:
            workers = mp.Pool(self.num_core)
            args = [(self, scheme) for scheme in fitting_schemes]
            alphas = [s.get_scalar_parameter() for s in fitting_schemes]
            cv = workers.map(loocv_mp, args)
            cv = np.array(cv)
        else:
            cv = np.ones(len(fitting_schemes))
            for i, scheme in enumerate(fitting_schemes):
                self.set_fitting_scheme(fitting_scheme=scheme)
                if self.scoring_scheme == "loocv":
                    cv[i] = self.loocv()
                elif self.scoring_scheme == "loocv_fast":
                    cv[i] = self.loocv_fast()
                num_eci = len(np.nonzero(self.get_eci())[0])
                alpha = scheme.get_scalar_parameter()
                alphas.append(alpha)
                logger.info('{:.10f}\t {}\t {:.10f}'.format(alpha, num_eci,
                                                            cv[i]))

        # --------------- #
        # Generate a plot #
        # --------------- #
        # if logfile is present, read all entries from the file
        if logfile:
            alphas, cv = self._get_alphas_cv_from_file(logfile)

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

    def _get_alphas_cv_from_file(self, logfile):
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
        return alphas, cv

    def _remove_existing_alphas(self, logfile, fitting_schemes):
        if not isinstance(logfile, str):
            return fitting_schemes
        elif logfile == "-":
            return fitting_schemes

        existing_alpha = []
        with open(logfile) as f:
            next(f)
            for line in f:
                existing_alpha.append(float(line.split()[0]))
        schemes = []
        for scheme in fitting_schemes:
            exists = np.isclose(existing_alpha, scheme.get_scalar_parameter(),
                                atol=1E-9).any()
            if not exists:
                schemes.append(schemes)
        return schemes

    def _initialize_logfile(self, logfile):
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
                if os.stat(logfile).st_size == 0:
                    logger.info("alpha \t\t # ECI \t CV")

    def plot_ECI(self, ignore_sizes=[0], interactive=True):
        """Plot the all the ECI.

        Argument
        =========
        ignore_sizes: list of ints
            Sizes listed in this list will not be plotted.
            Default is to ignore the emptry cluster.
        """
        import matplotlib.pyplot as plt
        from ase.clease.interactive_plot import InteractivePlot

        if self.eci is None:
            raise ValueError("ECI is None. You have to call get_eci first!")
        distances = self._distance_from_names()

        # Structure the ECIs in terms by size
        eci_by_size = {}
        for name, d, eci in zip(self.cluster_names, distances, self.eci):
            size = int(name[1])
            if size not in eci_by_size.keys():
                eci_by_size[size] = {"d": [], "eci": [], "name": []}
            eci_by_size[size]["d"].append(d)
            eci_by_size[size]["eci"].append(eci)
            eci_by_size[size]["name"].append(name)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.axhline(0.0, ls="--", color="grey")
        markers = ["o", "v", "x", "D", "^", "h", "s", "p"]
        annotations = []
        lines = []
        for size, data in eci_by_size.items():
            if size in ignore_sizes:
                continue
            data["d"] = np.array(data["d"])
            data["eci"] = np.array(data["eci"])
            sort_index = np.argsort(data["d"])
            data["d"] = data["d"][sort_index]
            data["eci"] = data["eci"][sort_index]
            annotations.append([data["name"][indx] for indx in sort_index])
            mrk = markers[size % len(markers)]
            line = ax.plot(data["d"], data["eci"],
                           label="{}-body".format(size), marker=mrk,
                           mfc="none", ls="", markersize=8)
            lines.append(line[0])
        ax.set_xlabel("Cluster diameter")
        ax.set_ylabel("ECI (eV/atom)")
        ax.legend()
        if interactive:
            # Note: Internally this calls plt.show()
            InteractivePlot(fig, ax, lines, annotations)

    def _distance_from_names(self):
        """Get a list with all the distances for each name."""
        dists = []
        for name in self.cluster_names:
            if name == "c0" or name.startswith("c1"):
                dists.append(0.0)
                continue
            dist_str = name.split("_")[1]
            dist_str = dist_str.replace("p", ".")
            dists.append(float(dist_str))
        return dists

    def mae(self):
        """Calculate mean absolute error (MAE) of the fit."""
        if self.eci is None:
            self.get_eci()
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        return sum(np.absolute(delta_e)) / len(delta_e)

    def rmse(self):
        """Calculate root-mean-square error (RMSE) of the fit."""
        if self.eci is None:
            self.get_eci()
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        num_entries = len(delta_e)
        rmse_sq = 0.0
        for i in range(num_entries):
            rmse_sq += (delta_e[i])**2
        rmse_sq /= num_entries
        return np.sqrt(rmse_sq)

    def loocv_fast(self):
        """CV score based on the method in J. Phase Equilib. 23, 348 (2002).

        This method has a computational complexity of order n^1.
        """
        # For each structure i, predict energy based on the ECIs determined
        # using (N-1) structures and the parameters corresponding to the
        # structure i.
        # CV^2 = N^{-1} * Sum((E_DFT-E_pred) / (1 - X_i (X^T X)^{-1} X_u^T))^2
        if not self.scheme.support_fast_loocv:
            return self.loocv()

        if self.eci is None:
            self.get_eci()
        e_pred = self.cf_matrix.dot(self.eci)
        delta_e = self.e_dft - e_pred
        cfm = self.cf_matrix
        # precision matrix
        prec = self.scheme.precision_matrix(cfm)
        cv_sq = np.mean((delta_e / (1 - np.diag(cfm.dot(prec).dot(cfm.T))))**2)
        return np.sqrt(cv_sq)

    def loocv(self):
        """Determine the CV score for the Leave-One-Out case."""
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

    def _get_eci_loo(self, i):
        """Determine ECI values for the Leave-One-Out case.

        Eliminate the ith row of the cf_matrix when determining the ECIs.
        Returns the determined ECIs.

        Arguments:
        =========
        i: int
            iterator passed from the self.loocv method.
        """
        cfm = np.delete(self.cf_matrix, i, 0)
        e_dft = np.delete(self.e_dft, i, 0)
        eci = self.scheme.fit(cfm, e_dft)
        return eci

    def _filter_cluster_name(self):
        """Filter the cluster names based on size and diameter."""
        if self.max_cluster_size is None and self.max_cluster_dia is None:
            return

        filtered_cnames = []
        for name in self.cluster_names:
            size = int(name[1])
            if size < 2:
                dia = -1
            else:
                dia = float(name.split('_')[1].replace('p', '.'))
            if (size <= self.max_cluster_size and
                    dia < self.max_cluster_dia[size]):
                filtered_cnames.append(name)
        self.cluster_names = filtered_cnames

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
        names = []
        db = connect(self.setting.db_name)
        for row in db.select(self.select_cond):
            e_dft.append(row.energy / row.natoms)
            names.append(row.name)
        return np.array(e_dft), names


def loocv_mp(args):
    """Need to wrap this function in order to use it with multiprocessing.

    Arguments
    =========
    args: Tuple where the first entry is an instance of Evaluate
        and the second is the penalization value
    """
    evaluator = args[0]
    scheme = args[1]
    evaluator.set_fitting_scheme(fitting_scheme=scheme)
    alpha = scheme.get_scalar_parameter()

    if evaluator.scoring_scheme == "loocv":
        cv = evaluator.loocv()
    elif evaluator.scoring_scheme == "loocv_fast":
        cv = evaluator.loocv_fast()
    num_eci = len(np.nonzero(evaluator.get_eci())[0])
    logger.info('{:.10f}\t {}\t {:.10f}'.format(alpha, num_eci, cv))
    return cv
