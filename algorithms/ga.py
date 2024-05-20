"""
Genetic Algorithm Solver for the k-ASP Problem

Lee de Byl, University of Western Australia, May 2024

Provides an implementation of a Genetic Algorithm Solver for the k-ASP problem.
Implements both Roulette and Ranking Selection, along with Uniform Crossover with Replacement.
"""


import networkx as nx
import random
import bitarray
import bitarray.util
import itertools
import numpy as np
import logging
from graph.dynamic_aspl import PartiallyDynamicAllPairsShortestPaths
from algorithms.solver import kASPSolver
from collections import Counter

class GeneticSolver(kASPSolver):
    """
    A class that represents a genetic algorithm solver for the k-ASP problem.

    Attributes:
    - population_size (int): The size of the population in the genetic algorithm.
    - creature_mutation_rate (float): The mutation rate for each creature in the population.
    - gene_mutation_rate (float): The mutation rate for each gene in a creature.
    - selection_method (str): The selection method used in the genetic algorithm.
    """

    # Define the default method name. Solutions will be stored in the trace
    # with this method name, unless overridden.
    method_name = 'Genetic Solver'

    def __init__(self, population_size=200,
                 creature_mutation_rate=0.1,
                 gene_mutation_rate=0.1,
                 selection_method='ranking',
                 trace_probabilities=False, *args, **kwargs):
        """
        Initializes a GeneticSolver object.

        Args:
        - population_size (int): The size of the population in the genetic algorithm.
        - creature_mutation_rate (float): The mutation rate for each creature in the population.
        - gene_mutation_rate (float): The mutation rate for each gene in a creature.
        - selection_method (str): The selection method used in the genetic algorithm.
        - *args, **kwargs: Additional arguments and keyword arguments, passed to the superclass.

        """
        super().__init__(*args, **kwargs)
        self.population_size = population_size
        self.gene_mutation_rate = gene_mutation_rate
        self.creature_mutation_rate = creature_mutation_rate
        self.selection_method = selection_method
        self.trace_probabilities = trace_probabilities
        self.probability_trace = []

    def algorithm(self, problem):
        """Genetic Algorithm Solver"""

        def spawn_population():
            """
            Spawns a new population of organisms for use within
            a genetic algorithm. Each organism, or creature,
            is represented as a bitarray of edges.

            To-Do: Wrap in class to maintain state of various
            mappings - e.g. indexes in bitarrays to edges
            """
            nonlocal population
            population = [set(random.sample(S, k)) for _ in range(n)]
        
        def crossover():
            # Set Uniform Crossover with Replacement: In this variant of Set Uniform Crossover, items are randomly selected from t
            nonlocal population
            spawn = []
            for i in range(n):
                a, b = random.sample(population, 2)
                # Maintain common genes between a and b
                c = a & b
                d = (a | b) - c
                e = c | set(random.sample(list(d), k - len(c)))
                assert len(e) == k, "Crossover failed to produce a valid offspring"
                spawn.append(e)
            population = spawn

        def mutate():
            # Mutate the population
            nonlocal population
            mutations = []

            # S = set(S)

            for i, g in enumerate(population):
                if np.random.rand() > self.creature_mutation_rate:
                    mutated = g
                else:
                    # Based on mutation probability, use a Binomial
                    # distribution to determine the number of mutations
                    # to apply to the creature
                    n_mutations = np.random.binomial(k, self.gene_mutation_rate)
                    n_keep = k - n_mutations
                    n_replace = k - n_keep
                    mutated = set(random.sample(list(g), n_keep))
                    mutated |= set(random.sample(list(set(S) - mutated), n_replace))
                    #assert len(mutated) == k, "Mutation failed to produce a valid offspring"
                mutations.append(mutated)
            population = mutations

        def evaluate():
            """
            Evaluate the ASPL of each creature in the population, returning the probability of selection for each.
            Updates the trace information.
            """
            nonlocal fitness

            fitness = np.empty(len(population), dtype=np.float32)

            for i, creature in enumerate(population):
                fitness[i] = oracle.evaluate_edges(creature)
                solutions_explored.add(frozenset(creature))
                # Convert the creature to an index representation
                # suitanle for serialisation
                edge_idxs = [S.index(e) for e in creature]
                edge_hits.update(edge_idxs)
                self.trace(aspl=fitness[i], solutions_explored=len(solutions_explored), edge_hits=edge_hits)
        
        def ranking_selection():
            """
            Perform ranking selection on the population based on their fitness scores.

            Args:
            - population (list): A list of individuals in the population.
            - fitness_scores (list): A list of fitness scores corresponding to the individuals in the population.

            Returns:
            - selected (list): A list of selected individuals.
            """
            nonlocal population
            # Convert fitness_scores to numpy array
            fitness_scores = np.array(fitness)
            # Sort individuals by their fitness scores, least-fit(highest ASPL) to most-fit (lowest ASPL)
            sorted_indices = np.argsort(fitness_scores)[::-1]
            sorted_population = [population[i] for i in sorted_indices]

            # Get the index of the most fit creature
            #champion_index = sorted_indices[-1]

            # Calculate cumulative probabilities
            total_rank = sum(range(1, len(population) + 1))
            cumulative_probabilities = np.cumsum(np.arange(1, len(population) + 1) / total_rank)

            # Perform selection
            selected_indices = np.searchsorted(cumulative_probabilities, np.random.random(len(population)))
            # If the most fit creature hasn't survived, replace the least fit.
            #if champion_index not in selected_indices:
            #    selected_indices[0] = champion_index

            # Update the global population to be those selected
            population = [sorted_population[i] for i in selected_indices]

        def roulette_selection():
            # Returns a selection from the population
            # based on roulette selection
            # Invert the fitness values (minimisation problem)
            nonlocal population
            p = 1 / fitness
            # Normalise the fitness values
            p = p / np.sum(p)

            selected = np.random.choice(population, size=n, replace=True, p=p)
            population = list(selected)

            if self.trace_probabilities:
                self.probability_trace.append(p)

        def find_champion():
            # Get the best solution
            nonlocal aspl_best, solution
            min_aspl = np.min(fitness)
            
            if min_aspl < aspl_best:
                aspl_best = min_aspl
                solution = list(population[np.argmin(fitness)])

        ############################
        # Initialise the Algorithm #
        ############################

        # Set the selection method
        match self.selection_method:
            case 'ranking':
                selection = ranking_selection
            case 'roulette':
                selection = roulette_selection
            case _:
                raise ValueError('Invalid selection method for genetic algorithm')

        # Get the graph G, candidate edges S, and the number of edges to be added K
        G, k, S = problem.G, problem.k, problem.S
        # Create a distance oracle for fast ASPL queries
        oracle = PartiallyDynamicAllPairsShortestPaths(G)

        aspl_best = np.inf
        solution = None
        solutions_explored = set()
        edge_hits = Counter()

        n = self.population_size
        # List of creatures, represented as EdgeSets
        population = None
        # Fitness stores the ASPL of each creature
        fitness = None
    
        #########################
        # Run the algorithm     #
        #########################
        spawn_population()
        for i in self.iterator():
            crossover()
            mutate()
            evaluate()
            find_champion()
            selection()
        return aspl_best, solution




