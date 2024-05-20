"""
Exhaustive solver for the kASP Problem.
"""
from itertools import combinations
from collections import Counter
from algorithms.solver import kASPSolver
from graph.dynamic_aspl import *

class ExhaustiveSolver(kASPSolver):
    method_name = 'Exhaustive Solver'
    maximum_problem_size = 1e7
    
    def algorithm(self, problem):
        G, k, S = problem.G, problem.k, problem.S
        S = list(S)

        edge_count = Counter()
        candidates = combinations(S, k)
        oracle = PartiallyDynamicAllPairsShortestPaths(G)
        solution = None
        min_aspl = float('inf')

        for i, s in zip(self.iterator(), candidates):
            G_prime_aspl = oracle.evaluate_edges(s)
            if G_prime_aspl < min_aspl:
                min_aspl = G_prime_aspl
                solution = s

            self.trace(aspl=G_prime_aspl, solutions_explored=i+1)
            #edge_count.update(s)

        # Number of solutions explored is i+1
        return min_aspl, solution
         

         