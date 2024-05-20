"""
kASP Research Project

Provides a framework for implementing kASP solvers. The kASPSolver class
provides a common interface for implementing solvers, and includes
functionality for tracking the progress of the algorithm, logging,
and early stopping.

Author: Lee de Byl (lee@32kb.net)
Date: 2024-03-01
"""

from abc import abstractclassmethod
from typing import final
from graph.model import Solution
from collections import Counter
import time
import tqdm


class TimeoutException(Exception):
    pass

class LowerBoundException(Exception):
    pass

class ProblemTooBigException(Exception):
    pass

class kASPSolver:

    maximum_iterations = None
    maximum_time = None

    method_name = 'Abstract Base Solver'
    is_optimal = False
    # How freqeuently to update the progress bar? Setting this too low
    # will significantly affect execution times (with None, about 15x slower)
    update_iterations = None

    # Sets the maximum problem size inheriting solvers can operate on.
    maximum_problem_size = None

    # The following trace items won't be displayed. This can be overridden by
    # implementing classes
    suppress_updates = ['edge_hits']

    def __init__(self,
                 maximum_iterations:int=None,
                 maximum_time:float=None,
                 early_stopping:bool=True,
                 patience:int=20000,
                 min_delta:float=1e-4,
                 verbose:bool=True,
                 repititions:int=1,
                 method_name=None) -> None:
        self.verbose = bool(verbose)
        self.maximum_iterations = int(maximum_iterations) if maximum_iterations else kASPSolver.maximum_iterations
        self.maximum_time = maximum_time or kASPSolver.maximum_time
        self.patience = patience
        self.min_delta = float(min_delta)
        self.repititions = int(repititions)
        self.aspl_lower_bound = None
        self.method_name = method_name or self.__class__.method_name
        self.early_stopping = bool(early_stopping)
        self.initialise()

    def iterator(self):
        """
        Provides an iterator that can be used by imolemented solvers
        to control the total number of iterations and execution time.
        """
        # Create an iterator to yield the expected numebr of iterations
        r = range(self.expected_iterations(self._current_problem))

        # Create the progress bar
        self._pbar = tqdm.tqdm(r, desc=self.method_name, miniters=self.update_iterations, disable=not self.verbose)
        for i in self._pbar:
            # If we have reached the known lower bound, we can stop early
            if self._best_aspl and self.aspl_lower_bound and abs(self._best_aspl - self.aspl_lower_bound) < self.min_delta:
                self._termination_reason = 'lower_bound_reached'
                break
            # If the maximum time has been reached, stop early
            elif self.maximum_time and time.time() - self._start_time > self.maximum_time:
                self._termination_reason = 'time_limit_reached'
                break
            # If early stopping is enabled and no progress has been made, stop early
            elif self.early_stopping and self._iterations_since_improvement > self.patience:
                print(self.patience, self._iterations_since_improvement)
                self._termination_reason = 'early_stopping'
                break
            self._current_iteration = i
            yield i
        if self._termination_reason == 'unknown':
            self._termination_reason = 'iterations_exhausted'
        self._pbar.close()

    def solve_repeated(self, problem, repititions=None):
        """
        Given an instance of a kASP problems, solves it using the algorithm
        implemented in the subclass. Returns a Solution instance.

        The solve function handles timing, trace logging, and other bookkeeping.

        Parameters:
        - problem (Problem): The problem instance to be solved.
        - repetitions (int): The number of times to repeat the solving process.
            If unspecified, the instant default is used.
        """
        
        if self.maximum_problem_size and problem.N > self.maximum_problem_size:
            raise ProblemTooBigException()

        # If the problem has a known optimal solution, set the lower bound
        # for termination
        for solution in problem.solutions:
            if solution.is_optimal:
                self.aspl_lower_bound = solution.aspl
                print("Setting lower bound to ", self.aspl_lower_bound)
                break

        # Execute the algorithm
        repititions = repititions or self.repititions
        for repitition in range(repititions):
            # Prepare the instance for solving this problem.
            self.initialise()
            self._start_time = time.time()
            self._current_problem = problem
            (aspl, edges) = self.algorithm(problem)
            # Force aspl to be a float
            aspl = float(aspl)

            # Finalise time keeping
            end_time = time.time()
            # Convert the resultant set of edges to a Solution instance
            sln = Solution(problem, edges, aspl=aspl, method=self.method_name,
                           S_explored=self._solutions_explored,
                           termination_reason=self._termination_reason)
            sln.trace = self._current_trace
            sln.time = end_time - self._start_time
            sln.iterations = self._current_iteration
            sln.parameters = self.parameters
            # Validate that the solution has a lower ASPL than the original graph.
            # If not, the algorithm has failed to solve the problem or the set
            # of candidate edges may all be shortcut edges.
            if not aspl < problem.aspl:
                raise ValueError(f"ASPL of solution ({aspl}) is not less than original ASPL ({problem.aspl})")
            yield sln

        self.initialise()

    def solve(self, problem):
        """
        Solves the specific problem instance using the algorithm implemented.
        """
        # Use the repeated solver with a single repition.
        solutions = self.solve_repeated(problem, repititions=1)
        return next(solutions)

    def log_visits(self, edges):
        """
        Logs the edges visited during the search. This is used to track
        the search space and can be used to generate visualisations of the
        search space.
        """
        self._visited_edges.update(edges)

    def initialise(self):
        "Resets the current state of the solver."
        self._current_trace = []
        self._current_iteration = None
        self._start_time = None
        self._current_problem = None
        self._pbar = None
        self._best_aspl = None
        self._iterations_since_improvement = 0
        self._visited_edges = Counter()
        self._solutions_explored = 0
        self._termination_reason = 'unknown'
    
    def trace(self, aspl, solutions_explored, **kwargs):
        """
        Logs the current ASPL to the trace. This is used to track the progress
        of the algorithm and can be used to generate visualisations of the
        search space. Typically this will be called once per iteration from
        within the algorithm method.

        Parameters:
        - aspl (float): The current average shortest path length.
        - solutions_explored (int): The number of solutions explored so far.
        - **kwargs: Additional trace elements as keyword arguments. Implementation
            specific.
        """
        if not self._best_aspl or aspl < self._best_aspl:
            self._iterations_since_improvement = 0
            self._best_aspl = aspl
        else:
            self._iterations_since_improvement += 1

        self._solutions_explored = solutions_explored

        trace = dict(best_aspl=self._best_aspl, aspl=aspl, solutions_explored=solutions_explored)
        # Add additional trace elements
        trace.update(kwargs)
        self._current_trace.append(trace)
        
        display_data = {k: str(v) for k, v in trace.items() if k not in self.suppress_updates}
        # Ensure refresh=False for speed.
        self._pbar.set_postfix(display_data, refresh=False)

    @final
    def algorithm(self, problem):
        """
        This is the main algorithm that will be implemented by the subclasses.
        Should return the minimum achieved ASPL and edge set as a tuple.

        The interface has been specifically designed to minimise the degree
        of coupling between the kASPSolverClass and the implementation of
        algorithms. Subclasses can optionally make use of the functionality
        of the kASPSolver class, such as the time/iteration tracking and trace.
        """
        raise NotImplementedError
    
    @final
    def expected_iterations(self, problem=None):
        """
        Given a specific problem, returns the expected number of iterations.
        If a problem isn't specified, the current problem is used.

        Implementing subclasses can override this method, including
        optionally ignoring the maximum number of iterations.
        """
        problem = problem or self._current_problem
        result = min(problem.N, self.maximum_iterations or problem.N)
        return result

    @property
    def parameters(self):
        """
        Return the instance parameters.

        This method is useful for serialization.

        Returns:
            dict: A dictionary containing the instance parameters.
        """
        params = {key: value for key, value in vars(self).items() if not key.startswith('_')}
        return params

