"""
Exhaustive solver for the kASP Problem.
"""
from itertools import combinations
import random
from collections import Counter
from algorithms.solver import kASPSolver
from graph.dynamic_aspl import PartiallyDynamicAllPairsShortestPaths

class MonteCarloSolver(kASPSolver):
    method_name = 'Monte Carlo Solver'

    def algorithm(self, problem):
        G, k, S = problem.G, problem.k, problem.S
        S = list(S)

        attempts = set()
        edge_count = Counter()
        oracle = PartiallyDynamicAllPairsShortestPaths(G)
        solution = None
        min_aspl = float('inf')

        for i in self.iterator():
            # Find a candidate that hasn't been tried before
            # Note that i should be limited to the number of possible solutions
            # This may not be an optimal approach for large number of iterations
            while True:
                candidate = frozenset(random.sample(S, k))
                if candidate not in attempts:
                    break
            
            G_prime_aspl = oracle.evaluate_edges(candidate)
            if G_prime_aspl < min_aspl:
                min_aspl = G_prime_aspl
                solution = candidate

            self.trace(aspl=G_prime_aspl, solutions_explored=i+1)
            #edge_count.update(candidate)

        # Number of solutions explored is i+1
        return min_aspl, solution
         

         