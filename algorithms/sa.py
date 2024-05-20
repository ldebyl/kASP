"""
 _        _    ____  ____  
| | __   / \  / ___||  _ \ 
| |/ /  / _ \ \___ \| |_) |
|   <  / ___ \ ___) |  __/ 
|_|\_\/_/   \_\____/|_|    

Simulated Annealing Solver

Lee de Byl, May 2024
University of Western Australia

Provides a solver for the k-ASP problem using a simulated annealing algorithm.
"""

import itertools
import networkx as nx
import math
import random
import numpy as np
from algorithms.solver import kASPSolver
from graph.dynamic_aspl import PartiallyDynamicAllPairsShortestPaths

class SimulatedAnnealingSolver(kASPSolver):
    is_optimal = False
    method_name = 'Simulated Annealing Solver'

    def __init__(self,
                 maximum_iterations=5000,
                 initial_temperature=None,
                 minimum_temperature = 0.01,
                 annealing_schedule='adaptive',
                 acceptance_method='boltzmann',
                 acceptance_window=50,
                 target_acceptance_rate=0.5,
                 iar_window=100,
                 iar_acceptance_rate=0.8,
                 *args, **kwargs):
        super().__init__(*args, maximum_iterations, **kwargs)
        self.initial_temperature = initial_temperature
        self.annealing_schedule = annealing_schedule
        self.acceptance_window = acceptance_window
        self.target_acceptance_rate = target_acceptance_rate
        self.minimum_temperature = minimum_temperature
        self.acceptance_method = acceptance_method
        self.iar_window = iar_window
        self.iar_acceptance_rate = iar_acceptance_rate

    def algorithm(self, problem):
        def neighbours():
            "Returns edges from S that aren't in the current state"
            # Needs to be returned as a list to be able to use random.choice
            return list(set(S) - set(current_state))
        
        def calculate_iar():
            deltas = []
            # Determine the initial temperature
            current_state = oracle.evaluate_edges(random.sample(S, k))
            for i in range(self.iar_window):
                next_state = oracle.evaluate_edges(random.sample(S, k))
                deltas.append(abs(next_state - current_state))
            temperature = float(-np.mean(deltas) / np.log(self.iar_acceptance_rate))
            return temperature

        G, k, S = problem.G, problem.k, problem.S
        S = list(S)
        oracle = PartiallyDynamicAllPairsShortestPaths(G)
        

        # Set the starting temperature, either as provided
        # or dynamically calculated.
        temperature = self.initial_temperature or calculate_iar()

        # Calculate the cooling rate based on the starting and ending temperature
        cooling_rate = math.exp((math.log(self.minimum_temperature) - \
                                math.log(temperature)) / self.maximum_iterations)

        current_state = random.sample(S, k)
        current_aspl = oracle.evaluate_edges(current_state)
        best_aspl = current_aspl
        best_state = None
        
        # For adaptive cooling
        accepted_moves = 0
        objective_values = []

        # Perform the simulated annelaing algorithm
        for i in self.iterator():
            proposed_state = current_state.copy()
            # Determine which edge shall be replaced
            index = random.randint(0, k - 1)
            # Replace it with a random edge from S that isn't in the current state
            proposed_state[index] = random.choice(neighbours())
            # Evaluate the new state
            proposed_aspl = oracle.evaluate_edges(proposed_state)
            # Calculate the delta: greater than zero indicates a worse state
            delta = proposed_aspl - current_aspl
            # Calculate the acceptance probability using the Boltzmann distribution
            acceptance_probability = self.acceptance_probability(delta, temperature)

            # If the new state is better, accept it
            if proposed_aspl < current_aspl or math.log(random.random()) < acceptance_probability:
                current_aspl = proposed_aspl
                current_state = proposed_state
            # If the propsed state is the best so far, store it
            if proposed_aspl < best_aspl:
                best_aspl = proposed_aspl
                best_state = proposed_state

            self.trace(aspl=current_aspl,
                        temperature=temperature,
                        log_acceptance_probability=acceptance_probability,
                        delta=delta,
                        solutions_explored=i+1)
            
            match self.annealing_schedule:
                case 'exponential':
                    temperature *= cooling_rate

                # https://arxiv.org/pdf/2002.06124.pdf
                case 'adaptive':
                    
                    objective_values.append(proposed_aspl)

                    # Adaptive cooling based on the acceptance rate
                    if i % self.acceptance_window == 0:
                        acceptance_rate = accepted_moves / self.acceptance_window
                        if acceptance_rate > self.target_acceptance_rate:
                            # Decrease the temperature more slowly
                            temperature = temperature / (1 + (temperature * math.log(1 + 1e-4) / 3 * np.std(objective_values)))
                        else:
                            # Decrease the temperature more quickly
                            temperature = temperature / (1 + (temperature * math.log(1 + 1e-3) / 3 * np.std(objective_values)))
                        accepted_moves = 0
                        objective_values = []
                case _:
                    raise ValueError(f"Invalid annealing schedule '{self.annealing_schedule}'")
            
            if temperature < self.minimum_temperature:
                self._termination_reason = 'minimum_temperature'
                break
        return best_aspl, best_state

    @staticmethod
    def boltzmann_probability(delta, temperature):
        # The below may be wrong and be more sigmoid-esque
        return -delta / temperature 

    @staticmethod
    def metropolis_hastings_probability(delta, temperature):
        return -delta / temperature     #math.exp(-delta / temperature)
    
    def acceptance_probability(self, delta, temperature):
        match self.acceptance_method:
            case 'boltzmann':
                return self.boltzmann_probability(delta, temperature)
            case 'metropolis-hastings':
                return self.metropolis_hastings_probability(delta, temperature)
            case _:
                raise ValueError(f"Invalid acceptance method '{self.acceptance_method}'")

    def expected_iterations(self, problem=None):
        return self.maximum_iterations
