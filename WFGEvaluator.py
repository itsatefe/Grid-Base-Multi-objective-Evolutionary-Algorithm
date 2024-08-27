import numpy as np
from pymoo.problems import get_problem

class WFGEvaluator:
    def __init__(self, problem_name, n_var, M):
        self.n_var = n_var
        self.M = M
        self.problem_name = problem_name
        self.problem = get_problem(problem_name, n_var=n_var, n_obj=M)
  
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
