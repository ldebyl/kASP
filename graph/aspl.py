"""
Helper functions and classes for working with graphs.
"""

import networkx as nx
import numpy as np
import graphblas_algorithms as gb

WEIGHT = 'weight'

def aspl(G, recalculate=False):
    """
    Returns the average weighted shortest path for graph G.
    If the aspl has already been calculated for this graph,
    returns that value.
    """
    if 'aspl' in G.graph and not recalculate:
        return G.graph['aspl']
    elif nx.is_weighted(G):
        #G = gb.Graph.from_networkx(G)
        aspl = nx.average_shortest_path_length(G, WEIGHT)
    else:
        # Setting the "weight" parameter to None (default) uses a different algorithm"
        #G = gb.Graph.from_networkx(G)
        aspl = nx.average_shortest_path_length(G)
    G.graph['aspl'] = aspl
    return aspl