import networkx as nx
import graphblas_algorithms as ga
import graphblas as gb
import numpy as np
from collections import Counter

class AllPairsShortestPaths:

    VERIFY = False
    EPSILON = 1e-6

    def __init__(self, G):
        # Make networkx go brrr
        #self.G = ga.Graph.from_networkx(G, weight='weight')
        self.G = G.copy()
        # Create an adjacency matrix of weights
        # When working with graphblas in networkx, floyd_warhsall_numpy
        # returns a GraphBlas matrix, which is not compatible with numpy
        #self.shortest_paths = nx.floyd_warshall_numpy(G).to_dense()
        # If G is unweighted, self.shortest_paths will be an integer
        # matrix. Convert it to a float matrix to avoid issues with
        # adding weights to the matrix.
        #self.shortest_paths = self.shortest_paths.astype(np.float32)

        # Use graphblas to generate a numpy array of shortest paths
        #self.shortest_paths = ga.floyd_warshall(self.G).to_array()
        self.num_nodes = len(G)
        self.aspl = self.calculate_aspl()
    
    def calculate_aspl(self):
        # Create an adjacency matrix of weights
        # When working with graphblas in networkx, floyd_warhsall_numpy
        # returns a GraphBlas matrix, which is not compatible with numpy
        aspl = nx.average_shortest_path_length(self.G, 'weight')
        return aspl
    
    def insert_edges(self, edges):
        self.G.add_weighted_edges_from(edges)
        self.aspl = self.calculate_aspl()

    def evaluate_edges(self, edges):
        self.G.add_weighted_edges_from(edges)
        aspl = self.calculate_aspl()
        self.G.remove_edges_from(edges)
        return aspl
    
class AllPairsShortestPathsGB:

    VERIFY = False
    EPSILON = 1e-6

    def __init__(self, G, tracing=True):
        # Make networkx go brrr
        self.G = ga.Graph.from_networkx(G, weight='weight')
        # Create an adjacency matrix of weights
        # When working with graphblas in networkx, floyd_warhsall_numpy
        # returns a GraphBlas matrix, which is not compatible with numpy
        #self.shortest_paths = nx.floyd_warshall_numpy(G).to_dense()
        # If G is unweighted, self.shortest_paths will be an integer
        # matrix. Convert it to a float matrix to avoid issues with
        # adding weights to the matrix.
        #self.shortest_paths = self.shortest_paths.astype(np.float32)

        # Use graphblas to generate a numpy array of shortest paths
        #self.shortest_paths = ga.floyd_warshall(self.G).to_array()
        self.num_nodes = len(G)
        self.aspl = self.calculate_aspl()
        
        self.tracing = tracing
        self.edge_visit_count = Counter()
    
    def calculate_aspl(self):
        # Create an adjacency matrix of weights
        # When working with graphblas in networkx, floyd_warhsall_numpy
        # returns a GraphBlas matrix, which is not compatible with numpy
        #TODO: Edit the distance matrix directly (g.matrix) instead of using networkx
        #Create the distance matrix once
        shortest_paths = ga.floyd_warshall(self.G, is_weighted=True)
        # shortest_paths is a grabphblas matrix.
        # use a monoid to get the sum of all shortest paths
        # https://python-graphblas.readthedocs.io/en/latest/user_guide/operators.html?highlight=sum#monoids
        total = shortest_paths.reduce_scalar(gb.monoid.plus).value
        aspl = total / (self.num_nodes * (self.num_nodes - 1))
        return aspl
    
    def insert_edges(self, edges):
        self.G.add_weighted_edges_from(edges)
        self.aspl = self.calculate_aspl()

    def evaluate_edges(self, edges):
        previous_weights = []
        m = self.G.matrix
        for u,v,w in edges:
            previous_weights.append((u,v,m[u,v]))
            previous_weights.append((v,u,m[v,u]))
            m[u,v] = w
            m[v,u] = w
        aspl = self.calculate_aspl()
        # Restore the previous weights
        for u,v,w in previous_weights:
            m[u,v] = w
            m[v,u] = w
        return aspl

class PartiallyDynamicAllPairsShortestPaths:

    VERIFY = False
    EPSILON = 1e-6

    "http://forskning.diku.dk/PATH05/Pino.pdf"
    def __init__(self, G, tracing=True):
        # Create a node ID to index mapping. This is necessary because
        # the Floyd-Warshall algorithm requires that the nodes are
        # indexed from 0 to n-1, however there may be discontinuities
        # in Node IDs [Bug Fix]
        nodes = list(G.nodes())
        self.node_to_index = {node: index for index, node in enumerate(nodes)}

        # Make networkx go brrr
        #G = ga.Graph.from_networkx(G, weight='weight')
        # Create an adjacency matrix of weights
        # When working with graphblas in networkx, floyd_warhsall_numpy
        # returns a GraphBlas matrix, which is not compatible with numpy
        self.shortest_paths = nx.floyd_warshall_numpy(G)

        #self.shortest_paths = nx.floyd_warshall_numpy(G).to_dense()
        # If G is unweighted, self.shortest_paths will be an integer
        # matrix. Convert it to a float matrix to avoid issues with
        # adding weights to the matrix.
        #self.shortest_paths = self.shortest_paths.astype(np.float32)

        # Use graphblas to generate a numpy array of shortest paths
        #self.shortest_paths = ga.floyd_warshall(self.G).to_array()
        self.num_nodes = len(G)
        self.aspl = self.calculate_aspl(self.shortest_paths)
        
        self.tracing = tracing
        self.edge_visit_count = Counter()
        self.total_evaluations = 0

    def shortest_path_length(self, u, v):
        u = self.node_to_index[u]
        v = self.node_to_index[v]
        return self.shortest_paths[u][v]

    def calculate_shortest_paths(self, edges):
        # Original, well-tested implementation
        def sum_paths(shortest_paths, u, v, weight):
            u = self.node_to_index[u]
            v = self.node_to_index[v]
            sum_paths_uv = shortest_paths[u, :, np.newaxis] + shortest_paths[:, v]

            #TODO: Can this be moved outside the loop?
            sum_paths_uv += weight
            return np.minimum(shortest_paths, sum_paths_uv)
        
        shortest_paths = self.shortest_paths
 
        for u, v, weight in edges:
            shortest_paths = sum_paths(shortest_paths, u, v, weight)
            shortest_paths = sum_paths(shortest_paths, v, u, weight)
    
        return shortest_paths
    
    def calculate_aspl(self, shortest_paths):
        total_shortest_path_length = np.sum(shortest_paths)
        average_shortest_path_length = total_shortest_path_length / (self.num_nodes * (self.num_nodes - 1))
        return average_shortest_path_length

    def insert_edges(self, edges):
        self.shortest_paths = self.calculate_shortest_paths(edges)
        self.aspl = self.calculate_aspl(self.shortest_paths)
        self.verify(edges, self.aspl)

    def evaluate_edges(self, edges):
        shortest_paths = self.calculate_shortest_paths(edges)
        aspl = self.calculate_aspl(shortest_paths)
        self.verify(edges, aspl)
        self.total_evaluations += 1
        return aspl
    
    def verify(self, edges, aspl_test):
        if PartiallyDynamicAllPairsShortestPaths.VERIFY:
            G_prime = self.G.copy()
            G_prime.add_weighted_edges_from(edges)
            aspl_true = nx.average_shortest_path_length(G_prime, 'weight')
            assert abs(aspl_true - aspl_test) < PartiallyDynamicAllPairsShortestPaths.EPSILON, f'Verification failed: Oracle reported {aspl_test}, Floyd-Warshall reports {aspl_true}. Edges: {edges}'


class GraphBLASAllPairsShortestPaths:

    VERIFY = False
    EPSILON = 1e-6

    "http://forskning.diku.dk/PATH05/Pino.pdf"
    def __init__(self, G):
        self.G = ga.Graph.from_networkx(G)
        # Create an adjacency matrix of weights
        self.shortest_paths = nx.floyd_warshall_numpy(G)
        self.num_nodes = len(G)
        self.aspl = self.calculate_aspl(self.shortest_paths)
        
    def calculate_shortest_paths(self, edges):
        def sum_paths(shortest_paths, u, v, weight):
            sum_paths_uv = shortest_paths[u, :, np.newaxis] + shortest_paths[:, v]
            sum_paths_uv += weight
            return np.minimum(shortest_paths, sum_paths_uv)
        
        shortest_paths = self.shortest_paths
 
        for u, v, weight in edges:
            shortest_paths = sum_paths(shortest_paths, u, v, weight)
            shortest_paths = sum_paths(shortest_paths, v, u, weight)
        
        return shortest_paths
    
    def calculate_aspl(self, shortest_paths):
        total_shortest_path_length = np.sum(shortest_paths)
        average_shortest_path_length = total_shortest_path_length / (self.num_nodes * (self.num_nodes - 1))
        return average_shortest_path_length

    def insert_edges(self, edges):
        self.shortest_paths = self.calculate_shortest_paths(edges)
        self.aspl = self.calculate_aspl(self.shortest_paths)
        self.verify(edges, self.aspl)

    def evaluate_edges(self, edges):
        shortest_paths = self.calculate_shortest_paths(edges)
        aspl = self.calculate_aspl(shortest_paths)
        self.verify(edges, aspl)
        return aspl
    
    def verify(self, edges, aspl_test):
        if PartiallyDynamicAllPairsShortestPaths.VERIFY:
            G_prime = self.G.copy()
            G_prime.add_weighted_edges_from(edges)
            aspl_true = nx.average_shortest_path_length(G_prime, 'weight')
            assert abs(aspl_true - aspl_test) < PartiallyDynamicAllPairsShortestPaths.EPSILON, f'Verification failed: Oracle reported {aspl_test}, Floyd-Warshall reports {aspl_true}. Edges: {edges}'

