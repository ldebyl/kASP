import networkx as nx
import random
import itertools
import math
from scipy import stats
from graph import EdgeSet
from graph.model import Solution, Problem



def generate_problem_with_solution(G, k, S_degree, weighter=None) -> Problem:
    if not weighter:
        if not 'weighter' in G.graph:
            raise ValueError('No weighter provided, and no weighter associated with graph')
        weighter = G.graph['weighter']

    non_edges = list(nx.non_edges(G))
    random.shuffle(non_edges)
    assert k <= len(non_edges), f"Insufficient non-edges to formulate a problem with k={k}"
    
    # Note: Do not use random.choice here - it samples with replacement!
    # random.sample returns in random selection order (i.e. shuffled)
    #S = random.sample(non_edges, k = S_degree)

    # Manipulate the weights of the edges in S
    # to form a problem. Break S into two sets
    # of size S_degree and k - S_degree.
    # TODO: This is probably buggy. We need to re-evaluate the ASPL
    # for each added edge and then choose weights.
    # Some results have indicated the algorithm is not
    # returning the truly optimal solution (Brute Force solver is finding
    # a better solution)
    G_prime = G.copy()

    inert = []
    shortcut = []

    for u,v in non_edges:
        if len(shortcut) == k and len(inert) == S_degree - k:
            break
        d_uv = nx.shortest_path_length(G_prime, u, v, weight='weight')
        if d_uv > weighter.a and len(shortcut) < k:
            # Generate a weight less than d_uv
            w = weighter.random_weight(b=d_uv)
            assert w < d_uv, f"Generated shortcut edge for existing d_uv of {d_uv} with weight {w}"
            shortcut.append((u,v,w))
            G_prime.add_edge(u, v, weight=w)
        elif d_uv < weighter.b and len(inert) < S_degree - k:
            # Generate a weight greater than d_uv
            w = weighter.random_weight(a=d_uv)
            assert w >= d_uv, f"Generated inert edge for existing d_uv of {d_uv} with weight {w}"
            inert.append((u,v,w))
            #G_prime.add_edge(u, v, weight=w)

    S = shortcut + inert
    p = Problem(G, S, k)

    # Determine the ASPL of the solution
    aspl = nx.average_shortest_path_length(G_prime, 'weight')
    sln = Solution(p, shortcut, method="Weighted Edge Addition", aspl=aspl, known_optimal=True)
    return p






    