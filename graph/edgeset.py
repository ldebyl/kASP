"""
Helper functions and classes for working with graphs.
"""

import networkx as nx
import numpy as np

def conformtype(fn):
    def wrapper(self, other):
        if not isinstance(other, type(self)):
            other = type(self)(other)
        return EdgeSet(fn(self, other))
    return wrapper

class EdgeSet(list):
    """
    An EdgeSet represents a set of edges where direction is ignored.

    An EdgeSet behaves as as an iterable. But currently isn't indexable.

    Note: This should _not_ inherit from dict as it will break JSON serialisation.
    """
    def __init__(self, edges=None):
        "Initialises the EdgeSet with a list of edges"
        # Edges is a mapping of (u,v) -> w
        self.__edges = {}
        for e in edges or {}:
            u,v,w = self.normalise(e)

            # Insert the edge into the dict __edges
            # Use setdefault to get the weight of an existing edge (u,v)
            # We can check for duplicte edges between (u,v) with different weights
            # This is particularly useful when creating a new edgeset as part of
            # a set operation
            w2 = self.__edges.setdefault((u,v), w)

            if w2 != w:
                raise ValueError(f"Duplicate edge with different weights: {u,v,w} and {u,v,w2}")
        self.__hash = hash(iter(self))

    @conformtype
    def __and__(self, other):
        "Set Interesection. Edges will only be included if w is equal"
        return set(self) & set(other)

    @conformtype
    def __sub__(self, other):
        return set(self) - set(other)

    @conformtype
    def __or__(self, other):
        " Set Union. If there are duplicate (u,v) edges mapping to different w, an error will be raise"
        return set(self) | set(other)

    def __hash__(self):
        return self.__hash

    def __eq__(self, other, /):
        return self.__edges == EdgeSet(other).__edges

    def __ne__(self, other, /):
        not self == other

    def __iter__(self):
        # To-Do: This should return ((u,v),w)?
        return ((u,v,w) for (u,v), w in self.__edges.items())

    def __len__(self):
        return len(self.__edges)

    def __str__(self):
        return str(list(self))
    
    def __repr__(self):
        return f"<EdgeSet with {len(self)} edges: {self.__edges}>"

    def __contains__(self, edge):
        return self.index(edge) != -1

    def __getitem__(self, index):
        match index:
            case (u,v):
                return self.__edges.get((u,v))
            case (idx):
                return list(self)[idx]

    def symmetric_difference(self, other):
        return EdgeSet(set(self).symmetric_difference(set(other)))

    def index(self, edge):
        u,v,w = self.normalise(edge)
        index = list(self.__edges).index((u,v))
        index = -1 if w and w != self.__edges[(u,v)] else index
        return index

    @staticmethod
    def normalise(edge):
        "Numerically Orders edges expressed as a couple or triplet, leaving the weight last"     
        match edge:
            case (u,v):
                return (u,v,None) if u <= v else (v,u)
            case (u,v,w):
                return (u,v,w) if u<= v else (v,u,w)
            case _:
                raise ValueError("Bad edge format - should be (u,v) or (u,v,w).")

    def add(self, edge):
        "Adds an edge in either u,v,w or u,v format"
        (u,v,w) = EdgeSet.normalise(edge)
        self.__edges[u,v] = w

    @property
    def vertices(self):
        "Returns a set of the vertices in the EdgeSet"
        return set(self.__edges)
    
    @property
    def sum_of_weights(self):
        return sum(self.__edges.values())

