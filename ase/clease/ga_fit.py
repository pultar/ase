from __future__ import print_function
from ase.clease import LinearRegression, Tikhonov
import numpy as np
import os

class GAFit(LinearRegression):
    """
    Genetic Algorithm for selecting relevant clusters

    Arguments
    =========
    evaluator: Evaluate
        Instance of the Evaluate class. The GA needs 
        cf_matrix and e_dft from the evaluate class.
    alpha: float
        Regularization parameter for ridge regression which is 
        used internally to obtain the coefficient
    elitism: int
        Number of best structures that will be passed
        unaltered on to the next generation
    fname: str
        Filename used to backup the population. If this file
        exists, the next run will load the population from 
        the file and start from there.
    num_individuals: int or str
        Integer with the number of inidivuals or it is equal to "auto",
        in which case 10 times the number of candidate clusters is used
    change_prob: float
        If a mutation is selected this denotes the probability of a mutating
        a given gene.

    Examples:

    from ase.clease import Evaluate
    from ase.clease import GAFit
    setting = None # Should be an ASE ClusterExpansionSetting object
    evaluator = Evaluate(setting)
    ga_fit = GAFit(evaluator)
    ga_fit.run()
    evaluator.get_cluster_name_eci()

    """
    def __init__(self, evaluator=None, mutation_prob=0.001, alpha=1E-5, elitism=3,
                 fname="ga_fit.csv", num_individuals="auto", change_prob=0.2):
        from ase.clease import Evaluate
        if not isinstance(evaluator, Evaluate):
            raise TypeError("evaluator has to be of type Evaluate")
        self.evaluator = evaluator
        self.evaluator.set_fitting_scheme(fitting_scheme=self)

        self.fname = fname
        if num_individuals == "auto":
            self.pop_size = 10*self.evaluator.cf_matrix.shape[0]
        else:
            self.pop_size = int(num_individuals)
        self.change_prob = change_prob
        self.num_genes = self.evaluator.cf_matrix.shape[1]
        self.individuals = self._initialize_individuals()
        self.fitness = np.zeros(len(self.individuals))
        self.regression = Tikhonov(alpha=alpha)
        self.elitism = elitism
        self.mutation_prob = mutation_prob
        self.statistics = {
            "best_cv": [],
            "worst_cv": []
        }

    def _initialize_individuals(self):
        """Initialize a random population."""
        individuals = []
        if os.path.exists(self.fname):
            individ_from_file = np.loadtxt(self.fname, delimiter=",")
            for i in range(individ_from_file.shape[0]):
                individuals.append(individ_from_file[i,:])
        else:
            for _ in range(self.pop_size):
                individual = np.random.choice(
                        [0, 1], size=self.num_genes)
                individual.astype(np.uint8)
                individuals.append(individual)
        return individuals

    def fit_individual(self, individual):
        X = self.evaluator.cf_matrix[:, individual==1]
        y = self.evaluator.e_dft
        coeff = self.regression.fit(X, y)

        e_pred = X.dot(coeff)
        delta_e = y - e_pred

        # precision matrix
        prec = self.regression.precision_matrix(X)
        cv_sq = np.mean((delta_e / (1 - np.diag(X.dot(prec).dot(X.T))))**2)
        return coeff, 1000.0*np.sqrt(cv_sq)

    def evaluate_fitness(self):
        """Evaluate fitness of all species."""
        for i, ind in enumerate(self.individuals):
            _, cv = self.fit_individual(ind)
            self.fitness[i] = 1.0/cv

    def flip_mutation(self, individual):
        """Apply mutation operation."""
        rand_num = np.random.rand(len(individual))
        flip_indx = (rand_num<self.change_prob)
        individual[flip_indx] = (individual[flip_indx]+1)%2
        return individual

    def sparsify_mutation(self, individual):
        """Change one 1 to 0."""
        indx = np.argwhere(individual==1)
        rand_num = np.random.rand(len(indx))
        flip_indx = (rand_num < self.change_prob)
        individual[indx[flip_indx]] = 0
        return individual
        
    def make_valid(self, individual):
        """Make sure that there is at least two active ECIs."""
        if np.sum(individual) < 2:
            while np.sum(individual) < 2:
                indx = np.random.randint(low=0, high=len(individual))
                individual[indx] = 1
        return individual

    def create_new_generation(self):
        """Create a new generation."""
        from random import choice
        new_generation = []
        srt_indx = np.argsort(self.fitness)[::-1]

        assert self.fitness[srt_indx[0]] >= self.fitness[srt_indx[1]]
        mutation_type = ["flip", "sparsify"]

        # Pass the fittest to the next generation
        for i in range(self.elitism):
            individual = self.individuals[srt_indx[i]].copy()
            new_generation.append(individual)

            # Try to insert mutated versions of the best
            # solutions
            mut_type = choice(mutation_type)
            if mut_type == "flip":
                individual = self.flip_mutation(individual.copy())
            else:
                individual = self.sparsify_mutation(individual.copy())
            new_generation.append(self.make_valid(individual))
        

        cumulative_sum = np.cumsum(self.fitness)
        cumulative_sum /= cumulative_sum[-1]
        num_inserted = len(new_generation)
        # Create new generation by mergin existing
        for i in range(num_inserted, self.pop_size):
            rand_num = np.random.rand()
            p1 = np.argmax(cumulative_sum>rand_num)
            p2 = p1
            while p2 == p1:
                rand_num = np.random.rand()
                p2 = np.argmax(cumulative_sum>rand_num)

            crossing_point = np.random.randint(low=0, high=self.num_genes)
            new_individual = self.individuals[p1].copy()
            new_individual[crossing_point:] = self.individuals[p2][crossing_point:]

            new_individual2 = self.individuals[p2].copy()
            new_individual2[crossing_point:] = self.individuals[p1][crossing_point:]
            if np.random.rand() < self.mutation_prob:
                mut_type = choice(mutation_type)
                if mut_type == "flip":
                    new_individual = self.flip_mutation(new_individual)
                    new_individual2 = self.flip_mutation(new_individual2)
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

    def fit(self, X, y):
        """Perform fit using the best individual."""
        individual = self.best_individual
        if not np.allclose(X, self.evaluator.cf_matrix):
            raise RuntimeError("Design matrix X has to match "
                               "the cf_matrix in Evaluate!")
        coeff, _ = self.fit_individual(individual)
        all_coeff = np.zeros(X.shape[1])
        all_coeff[individual==1] = coeff

        self.evaluator.cf_matrix[:, individual==0] = 0.0
        return all_coeff

    @staticmethod
    def get_instance_array():
        raise TypeError("Does not make sense to create an instance array "
                        "GA.")

    def save_population(self):
        # Save population
        np.savetxt(self.fname, self.individuals, delimiter=",")
        print("\nPopulation written to {}".format(self.fname))

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
        
        Arguments
        ===========
        gen_without_change: int
            Terminate if gen_without_change are created without
            sufficient improvement
        min_change: float
            Changes a larger than this value is considered
            "sufficient" improvement
        save_interval: int
            Rate at which all the populations are backed up 
            in a file
        """
        num_gen_without_change = 0
        current_best = 0.0
        gen = 0
        while(True):
            self.evaluate_fitness()

            best_indx = np.argmax(self.fitness)
            cv = 1.0/self.fitness[best_indx]
            num_eci = np.sum(self.individuals[best_indx])
            diversity = self.population_diversity()
            self.statistics["best_cv"].append(1.0/np.max(self.fitness))
            self.statistics["worst_cv"].append(1.0/np.min(self.fitness))
            self.log("Generation: {}. Best CV: {:.2f} meV/atom "
                     "Num ECI: {}. Pop. div: {:.2f}"
                     "".format(gen, cv, num_eci, diversity), end="\r")
            self.create_new_generation()

            if abs(current_best - cv) > min_change:
                num_gen_without_change = 0
            else:
                num_gen_without_change += 1
            current_best = cv

            if gen%save_interval == 0:
                self.save_population()

            if num_gen_without_change >= gen_without_change:
                self.log("\nReached {} generations, without sufficient improvement"
                         "".format(gen_without_change))
                break
            gen += 1
        self.save_population()
        

