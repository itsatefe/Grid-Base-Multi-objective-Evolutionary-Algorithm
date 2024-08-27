import numpy as np
import random

class Recombination:
    def __init__(self, parents, parameters, crossover_probability=0.9):
        self.parents = parents
        self.crossover_probability = crossover_probability
        self.crossovers = {'sbx': self.sbx, 'pcx': self.pcx, 'spx': self.spx, 
                           'blx_alpha': self.blx_alpha, 'de_rand_1': self.de_rand_1}
        self.parameters = parameters

    # depending on eta, the childern can be like their parent
    # higher eta, higher similarities
    def sbx(self, **kwargs):
        eta = kwargs.get('eta', 20)
        parent1, parent2 = self.parents[0], self.parents[1]
        u = np.random.rand(len(parent1))
        beta = np.empty_like(u)
        beta[u <= 0.5] = (2 * u[u <= 0.5]) ** (1 / (eta + 1))
        beta[u > 0.5] = (2 * (1 - u[u > 0.5])) ** (-1 / (eta + 1))
        offspring1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
        offspring2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)
        offsprings = [offspring1, offspring2]
        return offsprings



    def pcx(self, **kwargs):
        sigma_s = kwargs.get('sigma_s', 0.1)
        sigma_eta = kwargs.get('sigma_eta', 0.1)
        num_parents = len(self.parents)
        p_idx = np.random.choice(num_parents)
        parent_p = self.parents[p_idx]
        centroid = np.mean(np.delete(self.parents, p_idx, axis=0), axis=0)
        d_p = centroid - parent_p
        offspring = np.copy(parent_p)

        # Compute the weighted direction vector term w_s * d^(p)
        # w_s is sampled from a Gaussian distribution with mean 0 and a certain standard deviation
        w_s = np.random.normal(0, sigma_s)
        offspring += w_s * d_p

        # Add the summation term to the offspring
        # For each parent i (except for parent p), compute w_n * D^(i) and add to offspring
        for i in range(num_parents):
            if i != p_idx:
                # Compute D^(i), the difference between parent i and the centroid of all parents excluding i
                centroid_i = np.mean(np.delete(self.parents, i, axis=0), axis=0)
                D_i = centroid_i - self.parents[i]

                # Compute the weighted D^(i) term w_n * D^(i)
                # w_n is sampled from a Gaussian distribution with mean 0 and a certain standard deviation
                w_n = np.random.normal(0, sigma_eta)
                offspring += w_n * D_i

        return [offspring]

    def spx(self, **kwargs):
        epsilon = kwargs.get('epsilon', 1.0)
        parents = self.parents[:2]
        m, n = parents.shape
        center = np.mean(parents, axis=0)
        expanded_simplex = (1 + epsilon) * (parents - center)
        offsprings = np.empty_like(parents)
        for i in range(m):
            random_weights = np.random.dirichlet(np.ones(m), size=1)
            offsprings[i] = center + np.dot(random_weights, expanded_simplex)
        return offsprings


    def blx_alpha(self, **kwargs):
        alpha = kwargs.get('alpha', 0.5)
        parent1, parent2 = self.parents[0], self.parents[1]
        d = np.abs(parent1 - parent2)
        min_vals = np.minimum(parent1, parent2) - alpha * d
        max_vals = np.maximum(parent1, parent2) + alpha * d
        offspring = min_vals + np.random.rand(len(parent1)) * (max_vals - min_vals)
        return [offspring]


    def de_rand_1(self, **kwargs):
        if len(self.parents) < 4:
            return self.parents
        cr = kwargs.get('cr', 1)
        f = kwargs.get('f', 0.5)
        target, a, b, c = random.sample(list(self.parents), 4)
        size = len(target)
        jrand = np.random.randint(size)
        offspring = np.copy(target)
        for j in range(size):
            if np.random.rand() < cr or j == jrand:
                offspring[j] = a[j] + f * (b[j] - c[j])
        return [offspring]
            
    
    def select_crossover_operator(self, operator_probabilities):
        operators, probabilities = zip(*operator_probabilities)
        selected_operator = random.choices(operators, weights=probabilities, k=1)[0]
        return selected_operator
    
    def execute_crossover(self, crossover_name):
        if np.random.rand() > self.crossover_probability:
            parent1, parent2 = self.parents[0], self.parents[1]
            return parent1, parent2
        crossover = self.crossovers[crossover_name]
        return crossover(**self.parameters[crossover_name])
