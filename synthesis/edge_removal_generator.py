"""
Problem Generator
Implementation of the proposed solution by Ward for the construction
of graphs with known solutions.

Note that this doesn't work with unweighted graphs: there will
always be an edge between two neighbours that will be contained
within the set of edges in shortest paths (the shortest path
between two neighbours will always be the edge connecting them,
leaving no edges that can be removed).

Using a weighted graph should alleviate this problem.
"""
import networkx as nx
import random
import itertools
from graph import EdgeSet
from graph.model import Solution, Problem

#def edgeset(edges):
#    return frozenset([tuple(sorted(e)) for e in edges])

def paths_to_edges(S):
    """
    Given a dictionary of paths between source and
    destination vertices, returns the set of all edges
    within those paths. This will typically be of the format {S1: [D1, D2, D3, ...], S2: [D1, D2, D3, ...] ...}
    """
    S_edges = EdgeSet()
    # S may be a generator, convert to a dict. WTAF Am I thinking here? To-Do: Review
    S = dict(S)
    for source, destinations in S.items():
        for path in destinations.values():
            for edge in itertools.pairwise(path):
                S_edges.add(edge)
    return S_edges

def generate_problem(G, k, d = None):
    # Let g be a connected graph, k be the budget of edges to add,
    # d is the total number of edges to remove to form graph G'.
    """
    Take a connected graph G=(V, E). Let apl(G) be the average path length in G.
    Pick a subset of edges from E and call it D.
    The subset D should be selected at random such that G'=(V, E \ D) is still connected.
    We can create a new problem instance G'=(V, E') with D being the set of edges
    we are allowed to choose from. I also assume we have a budget k where 1<=k<=|D|.

    d may 
    """
    def add_weights(E, W):
        """
        For a collection of edges in (u,v) format, and a mapping W
        of (u,v) => w, returns E in the format (u,v,w)
        """
        return [(u,v, W[frozenset((u,v))]) for u,v in E]
    
    assert not d or k <= d, "Budget of edges to add must be less than or equal to the number of edges deleted."
    G_prime = G.copy()

    # Record edge weights for inclusion with the returned solution
    #W = {frozenset((u,v)): d for u,v,d in G_prime.edges(data='weight')}

    # Get all edge weights. These will be used to return edges
    # including weights in our solution.
                                                
    #print(f"Initial ASPL of G is {G_aspl}")

    # Find shortest paths for all vertex pairs.
    # We shall remove no more than k edges that
    # appear in this set.
    paths = nx.all_pairs_bellman_ford_path(G_prime, 'weight')

    # Take the shortest paths between all pairs u,v
    # and create a set of all edges contained within
    # those paths. Take the intersection with E which
    S = paths_to_edges(paths)
    assert len(S) < len(G_prime.edges), "All edges are part of a shortest path."

    #print(f"G has {len(G_prime.edges)} edges. {len(S)} edges are part of a shortest path")


    # Remove known bridges from S_edges
    # S = S - edgeset(nx.bridges(G_prime))
    D = EdgeSet()

    for i in range(k):
        E_bridges = EdgeSet(nx.bridges(G_prime))
        E_removable = S - E_bridges - D
        if not E_removable:
            raise ValueError("Insufficient candidate edges for removal")
        u,v,w = random.choice(sorted(E_removable))
        G_prime.remove_edge(u,v)
        D.add((u,v,w))

        
    #print(f"G_prime has {len(G_prime.edges)} edges. {len(S)} are part of a shortest path")
    # We wish to remove 0 <= |S_edges| <= k edges
    # Choose how many edges from S_edges we wish
    # to remove from G to form G'               

    #D = set(random.sample(list(S), k))
    #print(f"Starting with {k} edges: {D}")
    #print(f"|S_edges|: {len(S)}")
    #G_prime.remove_edges_from(list(D))

    # Let D be the set of candidate edges for addition.
    # It should be selected such that G' = (V, E \ D) 
    # remains connected. This can be determined from the 
    # set of edges minus the bridges.
    # Exclude edges already selected in D_s
    while d is None or len(D) < d:

        E_removable = EdgeSet(G_prime.edges) - EdgeSet(nx.bridges(G_prime)) - S

        if not E_removable:
            if d is not None:
                # There are no additional edges to remove.
                raise ValueError("Insufficient edges to remove that meet criteria")
            else:
                # There are no additional edges to remove, but this is okay.
                assert d is None or len(D) >= d, "Could not find d edges to remove."
                break

        # Choose an edge to remove and add it to D
        (u,v,w) = random.choice(sorted(E_removable))
        G_prime.remove_edge(u,v)
        D.add((u,v,w))
    
    # The solution consists of those edges which are part of a shortest
    # path and have been deleted.
    sln = S & D

    #assert set(sln).issubset(set(D)), "Solution contains edges not in D"
    #assert set(D).isdisjoint(set(G_prime.edges)), "Set D contains edges still in G_prime"
    #assert D.vertices.issubset(set(G_prime.nodes)), "Candidate Edges in solution refer to vertices that don't exist."
    #assert sln.vertices.issubset(set(G_prime.nodes)), "Solution Edges reference nodes not in G_prime."

    problem = Problem(G_prime, D, k)
    solution = Solution(problem, sln, optimal=True, method="KnownGeneration")
    problem.add_solution(solution)

    return problem





    

