from __future__ import print_function
from ase.clease import Tikhonov
import numpy as np
import multiprocessing as mp
import os
from random import shuffle
os.environ["OPENBLAS_MAIN_FREE"] = "1"


class GAFit(object):
    """
    Genetic Algorithm for selecting relevant clusters

    Arguments:
    =========
    setting: ClusterExpansionSetting
        Setting object used for the cluster expanion

    max_cluster_dia: float
        Maximum diameter included in the population

    max_cluster_size: int
        Maximum number of atoms included in the largest cluster

    alpha: float
        Regularization parameter for ridge regression which is used internally
        to obtain the coefficient

    elitism: int
        Number of best structures that will be passed unaltered on to the next
        generation

    fname: str
        File name used to backup the population. If this file exists, the next
        run will load the population from the file and start from there.
        Another file named 'fname'_cluster_names.txt is created to store the
        names of selected clusters.

    num_individuals: int or str
        Integer with the number of inidivuals or it is equal to "auto",
        in which case 10 times the number of candidate clusters is used

    change_prob: float
        If a mutation is selected this denotes the probability of a mutating
        a given gene.

    max_num_in_init_pool: int
        If given the maximum clusters included in the initial population
        is given by this number. If max_num_in_init_pool=150, then
        solution with maximum 150 will be present in the initial pool.

    parallel: bool
        If *True*, multiprocessing will be used to parallelize over the
        individuals in the population.
        NOTE: One of the most CPU intensive tasks involves matrix
        manipulations using Numpy. If your Numpy installation uses
        hyperthreading, it is possible that running with parallel=True
        actually leads to lower performance.

    num_core: int
        Number of cores to use during parallelization.
        If not given (and parallel=True) then mp.cpu_count()/2
        will be used

    select_cond: list
        Select condition passed to Evaluate to select which 
        data points from the database the should be included

    cost_func: str
        Use the inverse as fitness measure. 
        Possible cost functions:
        bic - Bayes Information Criterion
        aic - Afaike Information Criterion
        loocv - Leave one out cross validation (average)
        max_loocv - Leave one out cross valdition (maximum)

    sparsity_slope: int
        Ad hoc parameter that can be used to tune the sparsity
        of the model GA selects. The higher it is, the
        sparser models will be prefered. Has only impact
        if cost_func is bic or aic. Default value is 1.

    min_weight: float
        Weight given to the point furthest away from the
        convex hull.

    Example:
    =======
    from ase.clease import Evaluate
    from ase.clease import GAFit
    setting = None # Should be an ASE ClusterExpansionSetting object
    ga_fit = GAFit(setting)
    ga_fit.run()
    """
    def __init__(self, setting=None, max_cluster_size=None,
                 max_cluster_dia=None, mutation_prob=0.001, alpha=1E-5,
                 elitism=1, fname="ga_fit.csv", num_individuals="auto",
                 change_prob=0.2, local_decline=True,
                 max_num_in_init_pool=None, parallel=False, num_core=None,
                 select_cond=None, cost_func="bic", sparsity_slope=1.0,
                 min_weight=1.0, include_subclusters=True):
        from ase.clease import Evaluate
        evaluator = Evaluate(setting, max_cluster_dia=max_cluster_dia,
                             max_cluster_size=max_cluster_size,
                             select_cond=select_cond, min_weight=min_weight)
        self.setting = setting
        self.eff_num = evaluator.effective_num_data_pts
        self.W = np.diag(evaluator.weight_matrix)
        self.cluster_names = evaluator.cluster_names
        self.include_subclusters = include_subclusters
        self.sub_constraint = None
        if self.include_subclusters:
            self.sub_constraint = self._initialize_sub_cluster_constraint()

        allowed_cost_funcs = ["loocv", "bic", "aic", "max_loocv"]

        if cost_func not in allowed_cost_funcs:
            raise ValueError("Cost func has to be one of {}"
                             "".format(allowed_cost_funcs))

        self.cost_func = cost_func
        self.sparsity_slope = sparsity_slope
        # Read required attributes from evaluate
        self.cf_matrix = evaluator.cf_matrix
        self.cluster_names = evaluator.cluster_names
        self.e_dft = evaluator.e_dft
        self.fname = fname
        self.fname_cluster_names = \
            fname.rpartition(".")[0] + "_cluster_names.txt"
        if num_individuals == "auto":
            self.pop_size = 10*self.cf_matrix.shape[1]
        else:
            self.pop_size = int(num_individuals)
        self.change_prob = change_prob
        self.num_genes = self.cf_matrix.shape[1]
        self.individuals = self._initialize_individuals(max_num_in_init_pool)
        self.fitness = np.zeros(len(self.individuals))
        self.regression = Tikhonov(alpha=alpha, penalize_bias_term=True)
        self.elitism = elitism
        self.mutation_prob = mutation_prob
        self.parallel = parallel
        self.num_core = num_core
        self.statistics = {
            "best_cv": [],
            "worst_cv": []
        }
        self.evaluate_fitness()
        self.local_decline = local_decline

    def _initialize_individuals(self, max_num):
        """Initialize a random population."""
        individuals = []
        if os.path.exists(self.fname):
            individuals = self._init_from_file()
        else:
            max_num = max_num or self.num_genes
            indices = list(range(self.num_genes))
            num_non_zero = np.array(list(range(0, self.pop_size)))
            num_non_zero %= max_num
            num_non_zero[num_non_zero < 3] = 3
            for i in range(self.pop_size):
                shuffle(indices)
                individual = np.zeros(self.num_genes, dtype=np.uint8)
                indx = indices[:num_non_zero[i]]
                individual[np.array(indx)] = 1
                individuals.append(individual)
        return individuals

    def _init_from_file(self):
        """Initialize the population from file."""
        print("Initializing population from {}".format(self.fname))
        individuals = []
        with open(self.fname, 'r') as infile:
            for line in infile:
                individual = np.zeros(self.num_genes, dtype=np.uint8)
                indices = np.array([int(x.strip()) for x in line.split(",")])
                individual[indices] = 1
                individuals.append(individual)
        return individuals

    def bic(self, mse, nsel):
        """Return the Bayes Information Criteria."""
        N = len(self.e_dft)
        sparsity_cost = max((N, self.cf_matrix.shape[1]))
        return N*np.log(mse) + nsel*np.log(sparsity_cost)*self.sparsity_slope

    def aic(self, mse, num_features):
        """Return Afaike information criterion."""
        N = N = len(self.e_dft)
        return N*np.log(mse) + 2*num_features*self.sparsity_slope

    def fit_individual(self, individual):
        X = self.cf_matrix[:, individual == 1]
        y = self.e_dft
        coeff = self.regression.fit(X, y)

        e_pred = X.dot(coeff)
        delta_e = y - e_pred

        info_measure = None
        n_selected = np.sum(individual)
        mse = np.sum(self.W*delta_e**2)/self.eff_num

        if self.cost_func == "bic":
            info_measure = self.bic(mse, n_selected)
        elif self.cost_func == "aic":
            info_measure = self.aic(mse, n_selected)
        elif "loocv" in self.cost_func:
            prec = self.regression.precision_matrix(X)
            loo_dev = (self.W*delta_e / (1 - np.diag(X.dot(prec).dot(X.T))))**2
            cv_sq = np.sum(loo_dev)/self.eff_num
            cv = 1000.0*np.sqrt(cv_sq)

            if self.cost_func == "loocv":
                info_measure = cv
            elif self.cost_func == "max_loocv":
                info_measure = np.sqrt(np.max(loo_dev))*1000.0
            else:
                raise ValueError("Unknown LOOCV measure!")
        else:
            raise ValueError("Unknown cost function {}!"
                             "".format(self.cost_func))
        return coeff, info_measure

    def evaluate_fitness(self):
        """Evaluate fitness of all species."""

        if self.parallel:
            num_core = self.num_core or int(mp.cpu_count()/2)
            args = [(self, indx) for indx in range(len(self.individuals))]
            workers = mp.Pool(num_core)
            self.fitness[:] = workers.map(eval_fitness, args)
        else:
            for i, ind in enumerate(self.individuals):
                _, fit = self.fit_individual(ind)
                self.fitness[i] = -fit

    def flip_mutation(self, individual):
        """Apply mutation operation."""
        rand_num = np.random.rand(len(individual))
        flip_indx = (rand_num < self.change_prob)
        individual[flip_indx] = (individual[flip_indx]+1) % 2
        return individual

    def flip_one_mutation(self, individual):
        """Apply mutation where one bit flips."""
        indx = np.random.randint(0, high=len(individual))
        individual[indx] = (individual[indx] + 1) % 2
        return individual

    def sparsify_mutation(self, individual):
        """Change one 1 to 0."""
        indx = np.argwhere(individual == 1)
        rand_num = np.random.rand(len(indx))
        flip_indx = (rand_num < self.change_prob)
        individual[indx[flip_indx]] = 0
        return individual

    @property
    def sparsest_individual(self):
        num_coeff = len(self.individuals[0])
        sprs = self.individuals[0].copy()
        for ind in self.individuals:
            if np.sum(ind) < num_coeff:
                sprs = ind.copy()
                num_coeff = np.sum(ind)
        return sprs

    def get_random_sparse_individual(self):
        """Return new random individual that is
            sparser than the current sparsest"""
        sprs = self.sparsest_individual
        num_non_zero = np.sum(sprs)
        indices = list(range(0, len(sprs)))
        shuffle(indices)
        new = np.zeros_like(sprs)
        indices = np.array(indices)
        new[indices[:num_non_zero]] = 1
        return new

    def make_valid(self, individual):
        """Make sure that there is at least two active ECIs."""
        if np.sum(individual) < 2:
            while np.sum(individual) < 2:
                indx = np.random.randint(low=0, high=len(individual))
                individual[indx] = 1

        # Check if subclusters should be included
        if self.include_subclusters:
            individual = self._activate_all_subclusters(individual)
        return individual

    def create_new_generation(self):
        """Create a new generation."""
        from random import choice
        new_generation = []
        srt_indx = np.argsort(self.fitness)[::-1]

        assert self.fitness[srt_indx[0]] >= self.fitness[srt_indx[1]]
        mutation_type = ["flip", "sparsify"]

        # Pass the fittest to the next generation
        elitism_fit_selected = np.zeros(self.elitism)
        num_transfered = 0
        counter = 0
        while num_transfered < self.elitism and counter < len(srt_indx):
            indx = srt_indx[counter]
            diff = np.abs(self.fitness[indx] - elitism_fit_selected)
            if np.any(diff < 1E-4):
                counter += 1
                continue

            individual = self.individuals[indx].copy()

            # Transfer the best
            new_generation.append(individual)

            # Transfer the best individual with a mutation
            new_ind = self.flip_one_mutation(individual.copy())
            new_generation.append(self.make_valid(new_ind))
            elitism_fit_selected[num_transfered] = self.fitness[indx]
            num_transfered += 1
            counter += 1

        if counter >= len(srt_indx):
            raise RuntimeError("The entrie population has saturated!")

        only_positive = self.fitness - np.min(self.fitness)
        cumulative_sum = np.cumsum(only_positive)
        cumulative_sum /= cumulative_sum[-1]
        num_inserted = len(new_generation)

        # Insert two sparse solutions
        new_generation.append(self.make_valid(self.get_random_sparse_individual()))
        new_generation.append(self.make_valid(self.get_random_sparse_individual()))
        num_inserted += 2

        # Create new generation by mergin existing
        for i in range(num_inserted, self.pop_size):
            rand_num = np.random.rand()
            p1 = np.argmax(cumulative_sum > rand_num)
            p1 = np.random.randint(0, high=10)
            p2 = p1
            while p2 == p1:
                rand_num = np.random.rand()
                p2 = np.argmax(cumulative_sum > rand_num)

            new_individual = self.individuals[p1].copy()
            new_individual2 = self.individuals[p2].copy()

            mask = np.random.randint(0, high=2, dtype=np.uint8)
            new_individual[mask] = self.individuals[p2][mask]
            new_individual2[mask] = self.individuals[p1][mask]

            if np.random.rand() < self.mutation_prob:
                mut_type = choice(mutation_type)
                if mut_type == "flip":
                    new_individual = self.flip_one_mutation(new_individual)
                    new_individual2 = self.flip_one_mutation(new_individual2)
                else:
                    new_individual = self.sparsify_mutation(new_individual)
                    new_individual2 = self.sparsify_mutation(new_individual2)

            if len(new_generation) <= len(self.individuals)-2:
                new_generation.append(self.make_valid(new_individual))
                new_generation.append(self.make_valid(new_individual2))
            elif len(new_generation) == len(self.individuals)-1:
                new_generation.append(self.make_valid(new_individual))
            else:
                break
        self.individuals = new_generation

    def population_diversity(self):
        """Check the diversity of the population."""
        std = np.std(self.individuals)
        return np.mean(std)

    def log(self, msg, end="\n"):
        """Log messages."""
        print(msg, end=end)

    @property
    def best_individual(self):
        best_indx = np.argmax(self.fitness)
        individual = self.individuals[best_indx]
        return individual

    @property
    def best_cv(self):
        return 1.0/np.max(self.fitness)

    @property
    def best_individual_indx(self):
        best_indx = np.argmax(self.fitness)
        return best_indx

    @staticmethod
    def get_instance_array():
        raise TypeError("Does not make sense to create an instance array GA.")

    @property
    def selected_cluster_names(self):
        from itertools import compress
        individual = self.best_individual
        return list(compress(self.cluster_names, individual))

    def index_of_selected_clusters(self, individual):
        """Return the indices of the selected clusters

        Arguments
        ==========
        individual: int
            Index of the individual
        """
        return list(np.nonzero(self.individuals[individual])[0])

    def save_population(self):
        # Save population
        with open(self.fname, 'w') as out:
            for i in range(len(self.individuals)):
                out.write(",".join(str(x) for x in
                                   self.index_of_selected_clusters(i)))
                out.write("\n")
        print("\nPopulation written to {}".format(self.fname))

    def save_cluster_names(self):
        """Store cluster names of best population to file."""
        with open(self.fname_cluster_names, 'w') as out:
            for name in self.selected_cluster_names:
                out.write(name+"\n")
        print("Selected cluster names saved to "
              "{}".format(self.fname_cluster_names))

    def plot_evolution(self):
        """Create a plot of the evolution."""
        from matplotlib import pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(self.statistics["best_cv"], label="best")
        ax.plot(self.statistics["worst_cv"], label="worst")
        ax.set_xlabel("Generation")
        ax.set_ylabel("CV score (meV/atom)")
        plt.show()

    def run(self, gen_without_change=1000, min_change=0.01, save_interval=100):
        """Run the genetic algorithm.

        Return a list consisting of the names of selected clusters at the end
        of the run.

        Arguments:
        =========
        gen_without_change: int
            Terminate if gen_without_change are created without sufficient
            improvement

        min_change: float
            Changes a larger than this value is considered "sufficient"
            improvement

        save_interval: int
            Rate at which all the populations are backed up in a file
        """
        num_gen_without_change = 0
        current_best = 0.0
        gen = 0
        while(True):
            self.evaluate_fitness()

            best_indx = np.argmax(self.fitness)

            # If best individual is repeated: Perform local
            # optimization
            if best_indx != 0 and self.local_decline:
                self._local_optimization()

            num_eci = np.sum(self.individuals[best_indx])
            diversity = self.population_diversity()
            self.statistics["best_cv"].append(np.max(self.fitness))
            self.statistics["worst_cv"].append(np.min(self.fitness))

            self.log("Generation: {}. {}: {:.2e} "
                     "Num ECI: {}. Pop. div: {:.2f}"
                     "".format(gen, self.cost_func,
                               -self.fitness[best_indx],
                               num_eci, diversity), end="\r")
            self.create_new_generation()
            if abs(current_best - self.fitness[best_indx]) > min_change:
                num_gen_without_change = 0
            else:
                num_gen_without_change += 1
            current_best = self.fitness[best_indx]

            if gen % save_interval == 0:
                self.save_population()
                self.save_cluster_names()

            if num_gen_without_change >= gen_without_change:
                self.log("\nReached {} generations without sufficient "
                         "improvement".format(gen_without_change))
                break
            gen += 1

        if self.local_decline:
            # Perform a last local optimization
            self._local_optimization()
        self.save_population()
        self.save_cluster_names()
        return self.selected_cluster_names

    def _local_optimization(self, indx=None):
        """Perform a local optimization strategy to the best individual."""
        from random import choice
        from copy import deepcopy
        if indx is None:
            individual = self.best_individual
        else:
            individual = self.individuals[indx]

        num_steps = 10*len(individual)
        cv_min = -np.max(self.fitness)
        for _ in range(num_steps):
            flip_indx = choice(range(len(individual)))
            individual_cpy = deepcopy(individual)
            individual_cpy[flip_indx] = (individual_cpy[flip_indx]+1) % 2
            _, cv = self.fit_individual(individual_cpy)

            if cv < cv_min:
                cv_min = cv
                individual = individual_cpy

        for i in range(len(self.individuals)):
            if np.allclose(individual, self.individuals[i]):
                # The individual already exists in the population so we don't
                # insert it
                return

        self.individuals[self.best_individual_indx] = individual
        self.fitness[self.best_individual_indx] = -cv_min

    def _initialize_sub_cluster_constraint(self):
        """Initialize the sub-cluster constraint."""
        must_be_active = []
        for name in self.cluster_names:
            prefix = name.rpartition("_")[0]
            sub = self.setting.subclusters(prefix)
            indx = []
            for sub_name in sub:
                indices = [i for i, name in enumerate(self.cluster_names) 
                           if name.startswith(sub_name)]
                indx += indices
            must_be_active.append(indx)
        return must_be_active

    def _activate_all_subclusters(self, individual):
        """Activate all sub-clusters of the individual."""
        selected_indx = np.nonzero(individual)[0].tolist()
        active = set()
        for indx in selected_indx:
            active.union(self.sub_constraint[indx])

        active = list(active)
        if not active:
            return individual
        print(active)
        individual[active] = 1
        return individual


def eval_fitness(args):
    ga = args[0]
    indx = args[1]
    _, cv = ga.fit_individual(ga.individuals[indx])
    return -cv
