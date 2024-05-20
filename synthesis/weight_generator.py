"""
Generates randomised graphs of different classes.
"""

import networkx as nx
import numpy as np
import scipy.stats as stats
import random
from synthesis.randomisation import *

#TODO: This decorator isn't being used. Modify the code to use it
def restrictdomain(fn):
    def wrapper(self, *args, a=None, b=None, **kwargs):
        a = self.a if a is None else max(a, self.a)
        b = self.b if b is None else min(b, self.b)
        result = fn(self, *args, a, b , **kwargs)
        assert a <= result <= b, f"Weight {result} is outside the domain [{a},{b}]"
        return result
    return wrapper

class Weighter:
    """
    Abstract Base Class that defines a weighter that assigns weights to graphs
    or EdgeSets.
    """

    enabled = True
    name = "Abstract Base Weighter"

    def __init__(self, a=0, b=1000):
        assert a >= 0, "a must be greater than or equal to 0"
        assert b is None or b >= a, "b must be greater than or equal to a"
        self.a, self.b = float(a), float(b)

    def __repr__(self):
        return f"{self.name} with parameters {self.parameters}"

    def domain(self, a=None, b=None):
        a = self.a if a is None else max(a, self.a)
        b = self.b if b is None else min(b, self.b)
        return a,b

    def toJSON(self):
        return self.parameters

    def weight_graph(self, G, a=None, b=None):
        "Given a graph, assigns weights to its edges"
        for _,_, d in G.edges(data=True):
            d['weight'] = self.random_weight(a, b)

        # Set metadata on the graph
        G.graph['weighter_class'] = self.name
        G.graph['weighter'] = self
        return G

    def weight_edges(self, edges, a=None, b=None):
        """
        Given an iterable of edges in (u,v) format, returns an iterable
        of edges in (u,v,w) format. Useful for weighting candidate
        edges not yet added to a graph.
        """
        return [(u,v, self.random_weight(a,b)) for u,v in edges]

    def weight_edge(self, u, v, a=None, b=None):
        return u, v, self.random_weight(a,b)

    def weight_shortcut_edges(self, G, edges):
        """
        Given an interable of edges in (u,v) format, returns an iterable
        of edges in (u,v,w) format where the weight w will be selected
        to ensure that the edge (u,v) is not a shortcut in the graph G.
        """
        # TO-DO: Update these functions to support truncated functions
        # TODO: THERE IS AN ISSUE WHERE THE MINIMUM WEIGHT FOR AN 
        # INERT EDGE IS OUTSIDE THE BOUND OF THE TRUNCATED
        # DISTRIBUTION. THIS IS A PROBLEM.
        G_prime = G.copy()
        for u,v in edges:
            shortest_path_uv = nx.shortest_path_length(G_prime, u, v, weight='weight')
            w = self.random_weight(b = shortest_path_uv)
            assert w < shortest_path_uv, "Invalid shortcut edge."
            G_prime.add_edge(u, v, weight=w)
            yield (u,v,w)

    @property
    def parameters(self):
        return dict(a=self.a, b=self.b)

class ConstantWeighter(Weighter):
    name = "Constant Weighter"

    def __init__(self, w=1, a=1, b=1):
        super().__init__(a,b)
        self.w = float(w)
        assert self.a <= self.w <= self.b, "w must be within the domain [a,b]"

    def toJSON(self):
        json = super().toJSON()
        return json.update(dict(w=self.w))

    def random_weight(self, a=None, b=None):
        return self.w

class UniformWeighter(Weighter):

    name = "Uniform Weighter"

    def __init__(self, a=None, b=None):
        assert a is None or a > 0, "a must be greater than zero"
        self.a = float(a or Uniform(min=0, max=1000))
        self.b = float(b or Uniform(min=self.a, max=2000))
        assert self.a < self.b, "a must be less than b"

    def random_weight(self, a=None, b=None):
        a,b = self.domain(a,b)
        return np.random.uniform(a, b)
    
class GaussianWeighter(Weighter):
    # Class defaults
    min_mean = 10
    max_mean = 500
    min_std = 0.5
    max_std = 10

    enabled = True

    name = "Gaussian Weighter"
    #TODO: The mean should lie somewhere between a and b
    def __init__(self, mu=None, sigma=None, a=0, b=1000):
        super().__init__(a,b)
        self.mu = float(mu or Uniform(self.min_mean, self.max_mean))
        self.sigma = float(sigma or Uniform(self.min_std, self.max_std))

    def toJSON(self):
        #TODO: Implement completely
        json = super().toJSON()
        return json
    
    def random_weight(self, a=None, b=None):
        "Generates a random weight from the distribution"
        a,b = self.domain(a,b)
        X = stats.truncnorm(
            (a - self.mu) / self.sigma, (b - self.mu) / self.sigma, loc=self.mu,
            scale=self.sigma)
        return X.rvs(size=1)[0]

class MixtureWeighter(Weighter):
    name = "Mixture Weighter"
    enabled = True

    def __init__(self, weighters=None, a=0, b=1000):
        super().__init__(a,b)
        self.weighters = weighters or [GaussianWeighter(a=a,b=b), GaussianWeighter(a=a,b=b)]

    @property
    def parameters(self):
        #For each of the weighters in the mixture, get their parameters and
        #merge them toegther into a single dictionary
        return {f"{parameter}_{i}": value for i,w in enumerate(self.weighters) 
                for parameter,value in w.parameters.items()}

    def random_weight(self, a=None, b=None):
        a,b = self.domain(a,b)
        weighter = random.choice(self.weighters)
        return weighter.random_weight(a, b)

class MetaWeighter(Weighter):
    """
    A MetaWeighter is a weighter that selects a random weighter from a list of
    weighters and uses it to assign weights to a graph or a set of edges.

    It can be used where a Weighter Sublass is used, but behaves as a randomly
    selected Weighter subclass at instatiation.
    """
    # The MetaWeighter shouldn't be included in the set of weighters to be used in a MetaWeighter
    enabled = False
    def __new__(cls, weighters=None):
        # In retrospect, this wasn't the right pattern to use...
        weighters = weighters or [c for c in Weighter.__subclasses__() if c.enabled]
        weighter = random.choice(weighters)
        # __init__ will not be implicitly called if the classes are different.
        # It must be called explicitly
        weighter = weighter.__new__(weighter)
        weighter.__init__()
        return weighter

  
        


    


