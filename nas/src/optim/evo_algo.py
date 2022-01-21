import copy
import random
from tqdm import tqdm
import numpy as np

__all__ = ['EvolutionFinder']


class ArchManager:
    def __init__(self, blocks):
        self.blocks = blocks
        self.block_offset = [0]
        for b in self.blocks:
            self.block_offset.append(self.block_offset[-1] + b)


    def random_sample(self):
        sample = []
        for block in self.blocks:
            one_hot = [0 for _ in range(block)]
            argmax = np.random.choice(block)
            one_hot[argmax] = 1
            sample += one_hot

        return sample


    def random_resample(self, sample, i):
        assert i >= 0 and i < len(self.blocks)
        block = self.blocks[i]
        argmax = np.random.choice(block)
        for j in range(block):
            sample[self.block_offset[i] + j] = 1 if j == argmax else 0


    def mating(self, sample_1, sample_2):
        offspring = []
        parent_ind = np.random.choice(2, len(self.blocks))
        parents = (sample_1, sample_2)
        for i, p in zip(range(len(self.blocks)), parent_ind):
            for j in range(self.blocks[i]):
                offspring.append(parents[p][self.block_offset[i] + j])

        return offspring


class EvolutionFinder:
    valid_constraint_range = {
        'PyTorch_CPU': [0.02, 2.2],
    }

    def __init__(self, constraint_type, efficiency_constraint, efficiency_predictor, accuracy_predictor, **kwargs):
        self.constraint_type = constraint_type
        if not constraint_type in self.valid_constraint_range.keys():
            self.invite_reset_constraint_type()
        self.efficiency_constraint = efficiency_constraint
        if not (efficiency_constraint <= self.valid_constraint_range[constraint_type][1] and
                efficiency_constraint >= self.valid_constraint_range[constraint_type][0]):
            self.invite_reset_constraint()

        self.efficiency_predictor = efficiency_predictor
        self.accuracy_predictor = accuracy_predictor
        self.arch_manager = ArchManager(blocks=kwargs['blocks'])

        self.mutate_prob = kwargs.get('mutate_prob', 0.1)
        self.population_size = kwargs.get('population_size', 100)
        self.max_time_budget = kwargs.get('max_time_budget', 500)
        self.parent_ratio = kwargs.get('parent_ratio', 0.25)
        self.mutation_ratio = kwargs.get('mutation_ratio', 0.5)

    def invite_reset_constraint_type(self):
        print('Invalid constraint type! Please input one of:', list(self.valid_constraint_range.keys()))
        new_type = input()
        while new_type not in self.valid_constraint_range.keys():
            print('Invalid constraint type! Please input one of:', list(self.valid_constraint_range.keys()))
            new_type = input()
        self.constraint_type = new_type

    def invite_reset_constraint(self):
        print('Invalid constraint_value! Please input an integer in interval: [%d, %d]!' % (
            self.valid_constraint_range[self.constraint_type][0],
            self.valid_constraint_range[self.constraint_type][1])
              )

        new_cons = input()
        while (not new_cons.isdigit()) or (int(new_cons) > self.valid_constraint_range[self.constraint_type][1]) or \
                (int(new_cons) < self.valid_constraint_range[self.constraint_type][0]):
            print('Invalid constraint_value! Please input an integer in interval: [%d, %d]!' % (
                self.valid_constraint_range[self.constraint_type][0],
                self.valid_constraint_range[self.constraint_type][1])
                  )
            new_cons = input()
        new_cons = int(new_cons)
        self.efficiency_constraint = new_cons

    def set_efficiency_constraint(self, new_constraint):
        self.efficiency_constraint = new_constraint

    def random_sample(self):
        constraint = self.efficiency_constraint
        while True:
            sample = self.arch_manager.random_sample()
            efficiency = self.efficiency_predictor(sample)
            if efficiency <= constraint:
                return sample, efficiency


    def mutate_sample(self, sample):
        constraint = self.efficiency_constraint
        while True:
            new_sample = copy.deepcopy(sample)

            for i in range(len(self.arch_manager.blocks)):
                if random.random() < self.mutate_prob:
                    self.arch_manager.random_resample(new_sample, i)

            efficiency = self.efficiency_predictor(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency


    def crossover_sample(self, sample1, sample2):
        constraint = self.efficiency_constraint
        while True:
            new_sample = self.arch_manager.mating(sample1, sample2)

            efficiency = self.efficiency_predictor(new_sample)
            if efficiency <= constraint:
                return new_sample, efficiency


    def run_evolution_search(self, verbose=False):
        """Run a single roll-out of regularized evolution to a fixed time budget."""
        max_time_budget = self.max_time_budget
        population_size = self.population_size
        mutation_numbers = int(round(self.mutation_ratio * population_size))
        parents_size = int(round(self.parent_ratio * population_size))

        best_valids = [-np.inf]
        population = []  # (validation, sample, latency) tuples
        child_pool = []
        efficiency_pool = []
        if verbose:
            print('Generate random population...')
        for _ in range(population_size):
            sample, efficiency = self.random_sample()
            child_pool.append(sample)
            efficiency_pool.append(efficiency)

        accs = self.accuracy_predictor(child_pool)
        for i in range(population_size):
            population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))

        if verbose:
            print('Start Evolution...')
        # After the population is seeded, proceed with evolving the population.
        for iter in tqdm(range(max_time_budget), desc='Searching with %s constraint (%s)' %
                                                      (self.constraint_type, self.efficiency_constraint)):
            parents = sorted(population, key=lambda x: x[0])[::-1][:parents_size]
            acc = parents[0][0]
            if verbose:
                print('Iter: {} Acc: {}'.format(iter - 1, parents[0][0]))

            if acc > best_valids[-1]:
                best_valids.append(acc)
                best_sample = parents[0][1]
            else:
                best_valids.append(best_valids[-1])

            population = parents
            child_pool = []
            efficiency_pool = []

            for i in range(mutation_numbers):
                par_sample = population[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.mutate_sample(par_sample)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            for i in range(population_size - mutation_numbers):
                par_sample1 = population[np.random.randint(parents_size)][1]
                par_sample2 = population[np.random.randint(parents_size)][1]
                new_sample, efficiency = self.crossover_sample(par_sample1, par_sample2)
                child_pool.append(new_sample)
                efficiency_pool.append(efficiency)

            accs = self.accuracy_predictor(child_pool)
            for i in range(population_size):
                population.append((accs[i].item(), child_pool[i], efficiency_pool[i]))

        return best_sample