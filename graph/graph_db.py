import io
import os
import sqlite3
import numpy as np
import pandas as pd
import networkx as nx
import zlib
from . model import Problem, Solution, EdgeSet
from . import edgeset
import json
from . serialization import to_json, to_edgelist, from_edgelist

DEFAULT_DATABASE = 'graph.db'

def open(filename=DEFAULT_DATABASE):
    """
    Opens a database of kASP Graphs.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Database {filename} does not exist")
    return GraphDB(filename)

def expand_json(df, columns=[]):
    """
    Given a pandas dataframe, expands any columns that contain JSON data into
    multiple columns. This is useful when you have a dataframe that contains
    nested JSON data.
    """
    def json_to_series(json_data):
        try:
            data = json.loads(json_data)
            return pd.Series(data)
        except (json.JSONDecodeError, TypeError):
            return pd.Series()

    for c in columns:
        expanded_df = df[c].apply(json_to_series)
        # Prefix the column names in expanded_df
        expanded_df = expanded_df.add_prefix(f'{c}_')
        df = pd.concat([df, expanded_df], axis=1)
    return df

class GraphDB:
    def __init__(self, filename=DEFAULT_DATABASE, timeout=30.0):
        self.db = sqlite3.connect(filename, timeout=timeout)

    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.close()


    def close(self):
        self.db.close()

    def execute(self, sql, *args):
        cursor = self.db.cursor()
        cursor.execute(sql, args)
        return cursor

    def read_experiment_names(self):
        "Returns a list of the distinc experiment names in the database"
        sql = "SELECT distinct EXPERIMENT_NAME from PROBLEMS"
        cursor = self.execute(sql)
        return [row[0] for row in cursor]
    
    def read_experiment(self, experiment_name):
        """
        Given the name of an experiment, returns all problems associated with that experiment.
        """
        sql = "SELECT problem_id FROM problems WHERE experiment_name = ?"
        cursor = self.execute(sql, experiment_name)
        return (self.read_problem(row[0]) for row in cursor)

    def read_graphs(self):
        "Reads all graphs from the database"
        sql = "SELECT graph_id FROM graphs"
        cursor = self.execute(sql)
        return (self.read_graph(row[0]) for row in cursor)
    
    def read_problems(self):
        "Reads all problems from the database"
        sql = "SELECT problem_id FROM problems"
        cursor = self.execute(sql)
        return (self.read_problem(row[0]) for row in cursor)
    
    def read_graph(self, id):
        "Reads a graph from the database"
        sql = "SELECT edges, aspl, n, m, name, class, generation_parameters, is_weighted, attributes FROM graphs WHERE graph_id = ?"
        cursor = self.execute(sql, id)
        row = cursor.fetchone()
        cursor.close()
        if row:
            edges, aspl, n, m, name, graph_class, generation_parameters, is_weighted, attributes = row

            # Edges are zlib compressed.
            if edges:
                edges = zlib.decompress(edges)
                G = from_edgelist(edges)                
            else:
                # Else, if no edges were stored, create an empty graph with n nodes and no edges
                G = nx.empty_graph(n)

            G.graph['id'] = id
            G.graph['aspl'] = aspl
            G.graph['name'] = name
            G.graph['class'] = graph_class
            G.graph['generation_parameters'] = json.loads(generation_parameters)
            G.graph['is_weighted'] = bool(is_weighted)
            G.graph['attributes'] = json.loads(attributes)
            return G
        return None

    def read_problem(self, id):
        "Reads a problem from the database"
        sql = """
            SELECT graph_id, k, S, generation_method, generation_parameters,
                experiment_name, description
            FROM problems WHERE problem_id = ?
            """
        cursor = self.execute(sql, id)
        row = cursor.fetchone()
        cursor.close()

        graph_id, k, S, generation_method, generation_parameters, experiment_name, description = row
        G = self.read_graph(graph_id)
        S = EdgeSet(json.loads(S))
        problem = Problem(G, S, k, id=id, experiment_name=experiment_name, description=description)
        problem.method = generation_method
        problem.method_parameters = json.loads(generation_parameters)

        # Load all solutions for this problem
        sql = """
            SELECT edges, aspl, is_optimal, method,
                solver_parameters, trace, iterations,
                s_explored, run_time, termination_reason
            FROM solutions
            WHERE problem_id = ?
            """
        cursor = self.execute(sql, id)
        for row in cursor:
            edges, aspl, is_optimal, method, solver_parameters, trace, iterations, s_explored, run_time, termination_reason = row
            is_optimal = bool(is_optimal)
            edges = EdgeSet(json.loads(edges))
            solution = Solution(problem, edges, aspl, method, is_optimal, termination_reason=termination_reason)

            # Traces are zlib compressed - decompress before decoding
            trace = zlib.decompress(trace)
            trace = trace.decode('utf-8')
            solution.trace = json.loads(trace)

            solution.parameters = json.loads(solver_parameters)
            solution.time = run_time
            problem.add_solution(solution)
        cursor.close()
        return problem
        

    def write_problem(self, problem):
        "Adds a problem to the database"
        if problem.id:
            return problem.id
        
        # Assemble all the attributes that comprise a "Problems" entry
        graph_id = self.write_graph(problem.G)
        k = problem.k
        S = to_json(problem.S)
        S_degree = len(problem.S)
        N = problem.N
        generation_parameters = to_json(problem.method_parameters)
        generation_method = problem.method
        experiment_name = problem.experiment_name
        description = problem.description

        sql = """
            INSERT INTO problems (graph_id, k, S, degree_S, N, generation_method, generation_parameters,
                experiment_name, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        c = self.execute(sql, graph_id, k, S, S_degree, N, generation_method, generation_parameters, experiment_name, description)
        problem.id = c.lastrowid
        c.close()

        for s in problem.solutions:
            self.write_solution(s)

        self.db.commit()
        return problem.id

    def write_graph(self, G, write_edges=True):
        "Adds a graph to the database"
        # Get relevant graph properties
        id = G.graph.get('id', None)

        if id:
            return id
        n = G.number_of_nodes()
        m = G.number_of_edges()
        attributes = G.graph
        aspl = G.graph.get('aspl', nx.average_shortest_path_length(G, 'weight'))
        if write_edges:
            edges = to_edgelist(G)
            edges = zlib.compress(edges)
        else:
            edges = None
        name = G.name
        graph_class = G.graph.get('class')
        params = to_json(G.graph.get('generation_parameters'))
        attributes = to_json(G.graph)
        is_weighted = nx.is_weighted(G)

        sql = """
            INSERT INTO graphs
            (graph_id, edges, aspl, n, m, name, class, generation_parameters, is_weighted, attributes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (graph_id) DO UPDATE SET
                aspl = aspl,
                attributes = attributes,
                generation_parameters = generation_parameters,
                class = class,
                name = name,
                is_weighted = is_weighted
        """

        cursor = self.execute(sql, id, edges, aspl, n, m, name, graph_class, params, is_weighted, attributes)
        if not G.graph.get('id'):
            id = cursor.lastrowid
            G.graph['id'] = id
            print("Updating Graph id", id)
        cursor.close()
        self.db.commit()
        
        return id

    def write_solution(self, solution):
        "Adds a solution to the database"
        if solution.id:
            return solution.id
        
        # Assemble the attributes to be written
        problem_id = self.write_problem(solution.problem)
        edges = to_json(solution.edges)
        method = solution.method
        aspl = solution.aspl
        is_optimal = solution.is_optimal

        # Convert the trace to json, then compress it wit zlib.
        trace = to_json(solution.trace)
        # Convert the trace to a bytes-like
        trace = trace.encode('utf-8')
        trace = zlib.compress(trace)

        solver_parameters = to_json(solution.parameters)
        termination_reason = str(solution.termination_reason)

        sql = """
            INSERT INTO solutions (problem_id, edges, aspl, method,
            run_time, iterations, solver_parameters, trace, S_explored, is_optimal, termination_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        cursor = self.execute(sql, problem_id, edges, aspl, 
                              method, solution.time, solution.iterations,
                              solver_parameters, trace,
                              solution.S_explored, solution.is_optimal, termination_reason)
        if cursor.lastrowid:
            solution.id = cursor.lastrowid
        cursor.close()
        self.db.commit()
        return solution.id

    def delete_experiment(self, experiment_name):
        "Deletes all problems and solutions associated with an experiment"
        sql = "DELETE FROM problems WHERE experiment_name = ?"
        cursor = self.execute(sql, experiment_name)
        cursor.close()
        self.db.commit()

    def as_dataframe(self):
        "Returns the database, including graphs, problems and solutions as a pandas dataframe"
        import pandas as pd
        sql = """
            select
                graphs.graph_id,
                problems.experiment_name,
                problems.description,
                graphs.aspl as graph_aspl,
                graphs.n,
                graphs.m,
                graphs.name,
                graphs.class as graph_class,
                graphs.generation_method,
                graphs.generation_parameters,
                problems.problem_id,
                problems.K,
                problems.degree_S,
                problems.generation_method,
                solutions.aspl as "solution_aspl",
                solutions.run_time,
                solutions.is_optimal,
                solutions.method as solution_method,
                solutions.solver_parameters,
                solutions.iterations,
                solutions.is_optimal,
                solutions.termination_reason
            from graphs
                left outer join problems on problems.graph_id = graphs.graph_id
                left outer join solutions on solutions.problem_id = problems.problem_id;
        """
        df = pd.read_sql(sql, self.db)
        df.columns = df.columns.str.lower()
        df = expand_json(df, columns=['generation_parameters', 'solver_parameters'])
        df.columns = df.columns.str.lower()
        return df
