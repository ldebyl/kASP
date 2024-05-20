import networkx as nx
import random
import itertools
import math
from scipy import stats
from graph import EdgeSet
from graph.model import Solution, Problem

def generate_problem(G, k, S_degree, weighter=None):
    """
    Generates a single problem instance without a solution.

    G: The graph upon which the problem is expressed. Should already
        be weighted, if so desired.
    k: The budget of edges that may be added. Does not impact
        the selection of edges in S
    S_degree: The number of edges in S. This is the number of
        candidate edges that can be added in the problem.
    """
    non_edges = list(nx.non_edges(G))
    assert k <= len(non_edges), f"Insufficient non-edges to formulate a problem with k={k}"
    
    # Note: Do not use random.choice here - it samples with replacement!
    S = random.sample(non_edges, k = S_degree)

    # If a weighter has been provided, weight the edges.
    # Note that we use shortcut edges only! Non-shortcut edges
    # will never improve ASPL, so no point including them.
    weighter = weighter or G.graph.get('weighter')
    if weighter:
        S = weighter.weight_shortcut_edges(G, S)

    p = Problem(G, S, k)
    return p


