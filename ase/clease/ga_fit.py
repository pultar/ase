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
    """
    def __init__(self, evaluator=None, mutation_prob=0.001, alpha=1E-5, elitism=3,
                 fname="ga_fit.csv", num_individuals="auto"):
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
        self.num_genes = self.evaluator.cf_matrix.shape[1]
        self.individuals = self._initialize_individuals()
        self.fitness = np.zeros(len(self.individuals))
        self.regression = Tikhonov(alpha=alpha)
        self.elitism = elitism
        self.mutation_prob = mutation_prob

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
        flip_indx = (rand_num<1.0/len(individual))
        individual[flip_indx] = (individual[flip_indx]+1)%2
        return individual

    def sparsify_mutation(self, individual):
        """Change one 1 to 0."""
        indx = np.argwhere(individual==1)
        rand_num = np.random.rand(len(indx))
        flip_indx = (rand_num < 1.0/len(indx))
        individual[indx[flip_indx]] = 0
        return individual
        
    def create_new_generation(self):
        """Create a new generation."""
        from random import choice
        new_generation = []
        srt_indx = np.argsort(self.fitness)[::-1]

        assert self.fitness[srt_indx[0]] >= self.fitness[srt_indx[1]]

        # Pass the fittest to the next generation
        for i in range(self.elitism):
            new_generation.append(self.individuals[srt_indx[i]])

        cumulative_sum = np.cumsum(self.fitness)
        cumulative_sum /= cumulative_sum[-1]
        mutation_type = ["flip", "sparsify"]
        # Create new generation by mergin existing
        for i in range(self.elitism, self.pop_size):
            rand_num = np.random.rand()
            p1 = np.argmax(cumulative_sum>rand_num)
            p2 = p1
            while p2 == p1:
                rand_num = np.random.rand()
                p2 = np.argmax(cumulative_sum>rand_num)

            crossing_point = np.random.randint(low=0, high=self.num_genes)
            new_individual = self.individuals[p1].copy()
            new_individual[crossing_point:] = self.individuals[p2][crossing_point:]
            if np.random.rand() < self.mutation_prob:
                mut_type = choice(mutation_type)
                if mut_type == "flip":
                    new_individual = self.flip_mutation(new_individual)
                else:
                    new_individual = self.sparsify_mutation(new_individual)
            new_generation.append(new_individual)
        self.individuals = new_generation

    def population_diversity(self):
        """Check the diversity of the population."""
        std = np.std(self.individuals)
        return np.mean(std)

    def log(self, msg):
        """Log messages."""
        print(msg)

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

    def save_population(self):
        # Save population
        np.savetxt(self.fname, self.individuals, delimiter=",")
        print("Population written to {}".format(self.fname))

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
            self.log("Generation: {}. Best CV: {:.2f} meV/atom "
                     "Num ECI: {}. Pop. div: {:.2f}"
                     "".format(gen, cv, num_eci, diversity))
            self.create_new_generation()

            if abs(current_best - cv) > min_change:
                num_gen_without_change = 0
            else:
                num_gen_without_change += 1
            current_best = cv

            if gen%save_interval == 0:
                self.save_population()

            if num_gen_without_change >= gen_without_change:
                self.log("Reached {} generations, without sufficient improvement"
                         "".format(gen_without_change))
                break
            gen += 1
        self.save_population()
        

