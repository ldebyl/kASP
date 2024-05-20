import itertools
import networkx as nx
import math
import random
from algorithms.solver import kASPSolver
from graph.dynamic_aspl import PartiallyDynamicAllPairsShortestPaths

class GreedySolver(kASPSolver):
    method_name = 'Greedy Solver'

    def algorithm(self, problem):
        G, k, S = problem.G, problem.k, problem.S
        S = list(S)
        solution = set()
        oracle = PartiallyDynamicAllPairsShortestPaths(G)

        for i in self.iterator():
            best = float('inf')
            for s in S:
                G_prime_aspl = oracle.evaluate_edges([s])
                if G_prime_aspl < best:
                    best = G_prime_aspl
                    s_best = s

            solution.add(s_best)
            oracle.insert_edges([s_best])
        self.trace(aspl=best, solutions_explored=1)
        
        # The greedy solver always explores one complete solution and every edge k times.
        return best, solution

    def expected_iterations(self, problem=None):
        return problem.k
