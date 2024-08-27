import numpy as np
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions


class DTLZEvaluator:
    def __init__(self, problem_name, n_var, M=3, n_partitions = 12):
        self.n_var = n_var
        self.M = M
        self.problem_name = problem_name
        self.n_partitions = n_partitions
        self.problem = self.get_dtlz_problem()
        
    def get_dtlz_problem(self):
        ref_dirs = get_reference_directions("das-dennis", self.M, n_partitions = self.n_partitions)
        return get_problem(self.problem_name)
    
    def evaluate(self, x):
        return self.problem.evaluate(x)

    def get_true_pareto(self):
        
        return self.problem.pareto_front()
    
    def get_bounds(self):
        return self.problem.xl, self.problem.xu
    
    def ideal_point(self):
        return self.problem.ideal_point()
    
    def nadir_point(self):
        return self.problem.nadir_point()
