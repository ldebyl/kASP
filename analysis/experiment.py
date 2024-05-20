"""
 _        _    ____  ____  
| | __   / \  / ___||  _ \ 
| |/ /  / _ \ \___ \| |_) |
|   <  / ___ \ ___) |  __/ 
|_|\_\/_/   \_\____/|_|    

Experiment Driver

Lee de Byl
University of Western Australia
May 2024

Provides a simple driver for running experiments on the k-ASP problem.
Experiments are defined in the experiments dictionary in experiments.py.
Optionally runs experiments across multiple processes.
"""
from . experiments import experiments
from graph import graph_db
from synthesis import RandomGraphGenerator
from synthesis.problem_generator import generate_problem
from synthesis.combinations import find_closest_nCr
from multiprocessing import Pool
import multiprocessing
import networkx as nx
import argparse

def run_experiment(experiment_name, experiment):
    experiment_description = experiment['description']
    N = experiment['N']
    n = experiment['n']
    weighter = experiment['weighter']
    graph_generators = [g(n=n, weighter=weighter) for g in experiment['graph_generators']]

    solvers = experiment['solvers']
    problems = experiment['problems']
    graph_generator = RandomGraphGenerator(generators=graph_generators)

    for i in range(problems):
        try:
            G = next(graph_generator)
            print(G)
            print(G.graph)
            n = nx.number_of_nodes(G)
            
            S_degree_max = min(len(list(nx.non_edges(G))), 200)
            if S_degree_max < 2:
                print("Skipping graph - no non-edges.")
                continue
            S_degree, k = find_closest_nCr(int(N), S_degree_max)

            problem = generate_problem(G, k, S_degree)
            problem.experiment_name = experiment_name
            problem.description = experiment_description
            #db.write_problem(problem)
            print(problem)

            for solver in solvers:
                try:
                    sln = solver.solve(problem)
                except Exception as e:
                    print(e)
                    raise

                with graph_db.open('data/graph.db') as db:
                    db.write_solution(sln)
                print(sln)
                print(f"Reason for termination: {sln.termination_reason}")
                print("\n")
            print("\n\n\n")
        except Exception as e:
            print(e)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run experiments.')
    # Add a boolean argument as to whether the experiment should be deleted?
    parser.add_argument('--delete', action='store_true', help='Delete the experiment before running.')
    parser.add_argument('--experiment', required=True, type=str, help='Name of the experiment to run.')
    parser.add_argument('--processes', type=int, default=multiprocessing.cpu_count(), help='Number of processes to run the experiment across.')
    return parser.parse_args()

if __name__ == '__main__':
    arguments = parse_arguments()
    if arguments.delete:
        with graph_db.open('data/graph.db') as db:
            db.delete_experiment(arguments.experiment)

    name = arguments.experiment
    experiment = experiments[name]
    processes = arguments.processes
    # Run the experiment across n processes
    with Pool(multiprocessing.cpu_count()) as p:
        # Supress the output of the experiment
        p.starmap(run_experiment, [(name, experiment)] * processes)





