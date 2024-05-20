"""
 _        _    ____  ____  
| | __   / \  / ___||  _ \ 
| |/ /  / _ \ \___ \| |_) |
|   <  / ___ \ ___) |  __/ 
|_|\_\/_/   \_\____/|_|    

Graph Generator Module

Lee de Byl
University of Western Australia
May 2024

Provides classes for the generation of random graphs
"""

from abc import abstractmethod
import networkx as nx
import numpy as np
import scipy.stats as stats
import random
import synthesis.randomisation as randomisation
from synthesis.weight_generator import GaussianWeighter, Weighter
from synthesis.randomisation import RandomIterator, UniformInteger
from networkx.exception import NetworkXPointlessConcept

def get_all_subclasses(cls):
    all_subclasses = set()

    for subclass in cls.__subclasses__():
        all_subclasses.add(subclass)
        all_subclasses.update(get_all_subclasses(subclass))

    return list(all_subclasses)

class GraphGenerationException(Exception):
    pass

def retry(fn):
    """
    Preovides a decorator that can be used to re-attempt
    graph generation in the event that a graph is not
    valid. This is useful for graph generators that are
    not guranteed to produce useful graphs.
    """
    def wrapper(self, *args, **kwargs):
        result = None
        for i in range(self.max_attempts):
            print (f"Graph Generation Attempt {i+1} of {self.max_attempts}")
            try:
                return fn(self, *args, **kwargs)
            except GraphGenerationException as e:
                continue
        raise GraphGenerationException
    return wrapper


class GraphGenerator:
    """
    Abstract Base Class that defines a graph generator that parametrically
    generates new instances of graphs.
    """
    enabled = True
    graph_function = None

    def __init__(self, n,
                 weighter=None,
                 max_attempts=40,
                 minimum_nodes=3,
                 force_connected=True,
                 remove_isolates=True,
                 allow_subgraph=True,
                 subgraph_degree_tolerance_factor=0.1):
        self.n = n
        self.max_attempts = max_attempts
        self.force_connected = force_connected
        self.minimum_nodes = minimum_nodes
        self.remove_isolates = remove_isolates
        self.weighter = weighter
        self.allow_subgraph = allow_subgraph
        self.subgraph_degree_tolerance_factor=subgraph_degree_tolerance_factor
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Generates a single instance of the graph class with the specified paramenters.
        The returned graph will have the appropriate attributes set upon it.
        Actual graph generation is performed by the __generate() method.
        """

        # Generate the graph with the specified parameters
        G = None
        for i in range(self.max_attempts):
            try:
                parameters = self.generate_parameters()
                G = self.generator(**parameters)

                # Remove isolates
                if self.remove_isolates:
                    G.remove_nodes_from(list(nx.isolates(G)))
                
                # If the graph isn't connected, determine what the largest
                # subgraph is and form the graph from that.
                if not nx.is_connected(G):
                    # Determine the lower acceptable bound of nodes
                    # in the subgraph
                    n_lower = max(nx.number_of_nodes(G) * (1 - self.subgraph_degree_tolerance_factor), self.minimum_nodes)
                    nodes = max(nx.connected_components(G), key=len)
                    
                    G = nx.subgraph(G, nodes)
                    if nx.number_of_nodes(G) < n_lower:
                        raise GraphGenerationException("""
                            Attempted to form valid graph from connected subgraph, but the
                            resultant graph was too small with {nx.number_of_nodes(G)} nodes.
                            """)
                    
                if not self.is_valid(G):
                    raise GraphGenerationException("Generated graph is not valid.")
                
            except (GraphGenerationException, NetworkXPointlessConcept):
                if i + 1 >= self.max_attempts:
                    print("Warning: Maximum number of attempts exceeded to generate {self.__name__}")

        # Set the generation parameters as metadata on the
        # generated graph.
        G.graph['generation_parameters'] = parameters
        G.graph['class'] = self.graph_class
        G.graph['density'] = nx.density(G)
        G.name = self.name

        w = self.weighter() if isinstance(self.weighter, type) else self.weighter
        if w:
            # Weight the graph. This also sets metadata on graph G
            # about how the graph is weighted
            G = w.weight_graph(G)
        return G

    @property
    def name(self):
        return type(self).__name__

    @property
    def graph_class(self):
        return type(self).__name__

    def is_valid(self, G):
        """
        Returns true if graph G is valid according to the constraints of the graph class.
        """
        return G.number_of_nodes() >= self.minimum_nodes and \
            (nx.is_connected(G) or not self.force_connected) and \
            len(list(nx.non_edges(G))) > 0

    @abstractmethod
    def generate_parameters(self):
        "Returns concrete parameters for a single instance generation"
        return dict()



class ErdosRenyiGraph(GraphGenerator):
    enabled = True

    @staticmethod
    def generator(n, p):
        return nx.erdos_renyi_graph(n, p)

    def __init__(self, *args, p=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p or randomisation.Uniform()

    def generate_parameters(self):
        n = int(self.n)
        p = float(self.p)
        return dict(n=n, p=p)
        
class BarabasiAlbertGraph(GraphGenerator):
    enabled = True

    def __init__(self, *args, m=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.m = m or UniformInteger(1, 10)
        
    @staticmethod
    def generator(n, m):
        return nx.barabasi_albert_graph(n, m)

    def generate_parameters(self):
        n = int(self.n)
        # M should satisfy the constraint 1 <= m < n
        while ((m := int(self.m)) >= n): pass
        return dict(n=n, m=m)

class RandomRegularGraph(GraphGenerator):
    # Should this class of graph be included in the set
    # of randomly generated graphs?

    enabled = True

    def __init__(self, *args, d=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.d = d or randomisation.UniformInteger(2, 50)

    @staticmethod
    def generator(n, d):
        return nx.random_regular_graph(d, n)

    def generate_parameters(self):
        # n * d must be even
        while True:
            n = int(self.n)
            d = int(self.d)
            if n * d % 2 == 0 and d < n:
                break
        return dict(n=n, d=d)


class WattsStrogatzGraph(GraphGenerator):
    enabled = True

    def __init__(self, *args, k=None, p=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k or randomisation.UniformInteger(2,20)
        self.p = p or randomisation.Uniform(0, 1)

    @staticmethod
    def generator(n, k, p):
        return nx.connected_watts_strogatz_graph(n, k, p)

    def generate_parameters(self):
        n = int(self.n)
        while ((k := int(self.k)) >= n): pass
        p = float(self.p)
        return dict(n=n, k=k, p=p)

class NewmanWattsStrogatzGraph(WattsStrogatzGraph):
    enabled = True

    @staticmethod
    def generator(n, k, p):
        return nx.newman_watts_strogatz_graph(n, k, p)
    
class RandomGraphGenerator:
    """
    GraphGenerator generates random graphs of the specified classes with the specified weighting
    functions.

    n: The desried number of nodes that should be included in each graph. Note that this value
    is a suggestion to the individual graph generators and may be overridden, for example,
    if an even number of nodes is required for a particular graph class. This may be a 
    generator that returns integers.

    generators: Random graph generator objects that should be used to generate new
    graph instances. If this is None, all the enabled graph generators in the module are used.

    weighters: Instantiated weighter objects that should be used to assign weights to a 
    generated graph. If this is None, all Weighter classes defined in the module are used
    with default paramters.
    """

    def __init__(self, n=None, weighter = None, generators=None):
        self.generators = generators or [c(n=n, weighter=weighter) for c in get_all_subclasses(GraphGenerator) if c.enabled]
        self.weighter = weighter
        self.n = n

    def __iter__(self):
        return self

    def __next__(self):
        # Get the number of nodes. Note the cast to int will trigger the __int__ magic method
        g = random.choice(self.generators)
        return next(g)


