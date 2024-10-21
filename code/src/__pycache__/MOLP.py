import pulp
import numpy as np
from copy import deepcopy, copy
import matplotlib.pyplot as plt


class Solution():
    '''

    This class defines a solution object for a multi-objective linear program.

    Attributes:
    - id: unique identifier
    - objectives: list of objective functions (Pulp variables)
    - variables: list of variables (Pulp variables)

    Methods:
    - objective_values(): returns the values of the objective functions (numpy array)
    - variable_values(): returns the values of the variables (numpy array)
    - objective_dict(): returns a dictionary with the objective values {objective_name: value}
    - variable_dict(): returns a dictionary with the variable values {variable_name: value}
    
    '''
    def __init__(self, objectives, variables):
        self.id = None # Unique identifier
        self.objectives = deepcopy(objectives)
        self.variables = deepcopy(variables)

    def objective_values(self):
        return np.array([obj.value() for obj in self.objectives])
    
    def variable_values(self):
        return np.array([var.value() for var in self.variables])
    
    def objective_dict(self):
        return {obj.name: obj.value() for obj in self.objectives}
    
    def variable_dict(self):
        return {var.name: var.value() for var in self.variables}
    
    def remove_variable(self, variable_to_remove):
        self.variables = [var for var in self.variables if var.name not in variable_to_remove]
    

class MOLP():
    '''
    
    This class defines a multi-objective linear program.

    Attributes:
    - id: unique identifier
    - prob: Pulp problem object
    - original_prob: original Pulp problem object
    - objectives: list of objective functions (Pulp variables)
    - variables: list of variables (Pulp variables)
    - individual_optima: list of individual optima(Solution objects)
        - this attribute should be computed by the method compute_all_individual_optima()

    Methods:
    - num_objectives(): returns the number of objectives
    - num_variables(): returns the number of variables
    - compute_individual_optima(objective): computes the individual optima for a given objective
    - compute_all_individual_optima(): computes the individual optima for all (returns a list of Solution objects)
    - payoff_matrix(): returns the payoff matrix (numpy array)
    - ideal_point(): returns the ideal point (numpy array)
    - nadir_point(): returns the nadir point (numpy array)

    '''
    def __init__(self, prob, objectives, variables):
        self.id = None # Unique identifier
        self.prob = prob
        self.original_prob = deepcopy(prob)
        self.objectives = objectives # List of objective functions (Pulp variables)
        self.variables = variables # List of variables (Pulp variables)
        self.individual_optima = [] # List of individual optima (Solution objects)

    def num_objectives(self):
        return len(self.objectives)
    
    def num_variables(self):
        return len(self.variables)

    def compute_individual_optima(self, objective):
        sub_problem = copy(self.prob)
        aggregated_objective = objective + 0.00000001 * sum([o for o in self.objectives if o != objective])
        sub_problem.setObjective(aggregated_objective)     
        sub_problem.solve(pulp.PULP_CBC_CMD(msg=0))
        solution = Solution(self.objectives, self.variables)
        return solution
    
    def compute_all_individual_optima(self):
        for objective in self.objectives:
            self.individual_optima.append(self.compute_individual_optima(objective))
        return self.individual_optima
    
    def payoff_matrix(self):
        # Compute individual optima if not already computed
        if self.individual_optima == []:
            self.compute_individual_optima()
        # Compute payoff table
        payoff_matrix = np.array([[opt.objective_values()[j] for j in range(self.num_objectives())] for opt in self.individual_optima])
        return payoff_matrix
    
    def ideal_point(self):
        return np.min(self.payoff_matrix(), axis=0)
    
    def nadir_point(self):
        return np.max(self.payoff_matrix(), axis=0)