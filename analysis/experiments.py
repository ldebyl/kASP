"""
 _        _    ____  ____  
| | __   / \  / ___||  _ \ 
| |/ /  / _ \ \___ \| |_) |
|   <  / ___ \ ___) |  __/ 
|_|\_\/_/   \_\____/|_|    

Experiment Definitions

Lee de Byl
University of Western Australia
May 2024

Provides definitions of experiments to be run by the
experiment driver.
"""

from functools import partial, partial
from synthesis.graph_generator import \
    WattsStrogatzGraph, ErdosRenyiGraph, \
    BarabasiAlbertGraph, RandomRegularGraph, RandomGraphGenerator
from synthesis.randomisation import RandomChoice, UniformInteger
from synthesis.weight_generator import UniformWeighter
from algorithms import ExhaustiveSolver, GeneticSolver, SimulatedAnnealingSolver, GreedySolver, MonteCarloSolver

experiments = {
    'small_solveable': {
        'description': 'Random 50-150 nodes, random problem size 1e5 to 1e7, exhaustive solution.',
        'problems': 100,
        'N': UniformInteger(1e5, 1e7),
        'n': UniformInteger(50,100),
        'weighter': UniformWeighter(1,10000),
        'graph_generators': [
            partial(WattsStrogatzGraph, k=RandomChoice([2,5,10,20]), p=RandomChoice([0.1,0.5,0.9])),
            partial(ErdosRenyiGraph, p=RandomChoice([0.001, 0.3])),
            partial(BarabasiAlbertGraph, m=RandomChoice([1,10])),
            partial(RandomRegularGraph, d=RandomChoice([2,10])),
        ],
        'solvers': [
            ExhaustiveSolver(early_stopping=False),
            GreedySolver(early_stopping=False),
            SimulatedAnnealingSolver(early_stopping=False, maximum_iterations=20000, minimum_temperature=0.001, initial_temperature=None, annealing_schedule='exponential'),
            SimulatedAnnealingSolver(early_stopping=False, maximum_iterations=20000, minimum_temperature=0.001, initial_temperature=None, annealing_schedule='adaptive'),
            GeneticSolver(early_stopping=False, maximum_iterations=40, population_size=500, creature_mutation_rate=0.15, gene_mutation_rate=0.2, selection_method="ranking"),
            GeneticSolver(early_stopping=False, maximum_iterations=100, population_size=200, creature_mutation_rate=0.15, gene_mutation_rate=0.2, selection_method="ranking")
        ]
    },
    'medium_random_1': {
        'description': 'Medium-sized graphs with 50-500 nodes, no known solution, large solution space.',
        'problems': 100,
        'N': UniformInteger(1e10, 1e15),
        'n': UniformInteger(50,500),
        'weighter': UniformWeighter(1,10000),
        'graph_generators': [
            partial(WattsStrogatzGraph, k=RandomChoice([2,5,10,20,40]), p=RandomChoice([0.1,0.5,0.9])),
            partial(ErdosRenyiGraph, p=RandomChoice([0.001, 0.3, 0.7])),
            partial(BarabasiAlbertGraph, m=RandomChoice([1,10])),
            partial(RandomRegularGraph, d=RandomChoice([2,10, 20])),
        ],
        'solvers': [
            GreedySolver(early_stopping=False),
            SimulatedAnnealingSolver(early_stopping=True, maximum_iterations=50000, minimum_temperature=0.001, initial_temperature=None, annealing_schedule='exponential'),
            SimulatedAnnealingSolver(early_stopping=True, maximum_iterations=50000, minimum_temperature=0.001, initial_temperature=None, annealing_schedule='adaptive'),
            GeneticSolver(early_stopping=True, maximum_iterations=100, population_size=500, creature_mutation_rate=0.15, gene_mutation_rate=0.2, selection_method="ranking"),
            GeneticSolver(early_stopping=True, maximum_iterations=50, population_size=1000, creature_mutation_rate=0.15, gene_mutation_rate=0.2, selection_method="ranking")
        ]
    }
}
