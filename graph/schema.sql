/* 
 _        _    ____  ____  
| | __   / \  / ___||  _ \ 
| |/ /  / _ \ \___ \| |_) |
|   <  / ___ \ ___) |  __/ 
|_|\_\/_/   \_\____/|_|    

SQLite Schema for serialisation of experimental data.

Lee de Byl, May 2024
University of Western Australia

Schema version 1.1
*/


CREATE TABLE IF NOT EXISTS GRAPHS (
    GRAPH_ID INTEGER PRIMARY KEY,
    EDGES BLOB,             -- JSON Representation of edges and their attributes
    ASPL NUMERIC,           -- The Average Shortest Path Length of this graph
    N NUMERIC,              -- Number of Nodes, |V|
    M NUMERIC,              -- Number of Edges, |E|
    NAME TEXT,              -- The name of the graph
    CLASS TEXT,             -- The class of the graph
    ATTRIBUTES JSON,        -- The attributes of the graph 
    GENERATION_METHOD TEXT,
    GENERATION_PARAMETERS TEXT,
    IS_WEIGHTED BOOLEAN NOT NULL
);

-- PROBLEMS is a table to store the problems to be solved
CREATE TABLE IF NOT EXISTS PROBLEMS (
    PROBLEM_ID INTEGER PRIMARY KEY,
    GRAPH_ID INTEGER NOT NULL,
    EXPERIMENT_NAME TEXT,           -- Name of the experiement this problem was derived as part of
    DESCRIPTION TEXT,               -- Human readable description of the problem
    K INTEGER NOT NULL,             -- Value of k in the k-ASP Problem.
    S JSON,                         -- Edges
    DEGREE_S INTEGER,               -- The degree of the set S
    GENERATION_METHOD TEXT,         -- The method used to generate the problem
    GENERATION_PARAMETERS JSON,     -- The parameters used to generate the problem using GENERATION_METHOD
    N INTEGER,                       -- Search Space Cardinality,
    FOREIGN KEY(GRAPH_ID) REFERENCES GRAPHS(GRAPH_ID) ON DELETE CASCADE
);

-- SOLUTIONS is a table to store the solutions of the problems
CREATE TABLE IF NOT EXISTS SOLUTIONS (
    SOLUTION_ID INTEGER PRIMARY KEY,    
    PROBLEM_ID INTEGER NOT NULL,
    METHOD TEXT,
    EDGES JSON,
    ASPL NUMBER,
    SOLVER_PARAMETERS JSON,
    RUN_TIME NUMBER,
    TRACE BLOB,
    TERMINATION_REASON TEXT,
    ITERATIONS INTEGER,
    S_EXPLORED INTEGER, -- Number of unique explored solutions
    IS_OPTIMAL BOOLEAN,  -- Is the solution known to be optimal?
    FOREIGN KEY(PROBLEM_ID) REFERENCES PROBLEMS(PROBLEM_ID) ON DELETE CASCADE
);

-- SOLUTION_TRACE is a table to store the ASPL of each iteration of the algorithm
CREATE TABLE IF NOT EXISTS SOLUTION_TRACE (
    SOLUTION_ID INTEGER NOT NULL,
    ITERATION INTEGER NOT NULL,
    ASPL NUMERIC NOT NULL,
    FOREIGN KEY(SOLUTION_ID) REFERENCES SOLUTIONS(SOLUTION_ID) ON DELETE CASCADE
);