"""
 _        _    ____  ____  
| | __   / \  / ___||  _ \ 
| |/ /  / _ \ \___ \| |_) |
|   <  / ___ \ ___) |  __/ 
|_|\_\/_/   \_\____/|_|    

Serialisation utilities for the k-ASP problem.

Lee de Byl
University of Western Australia
May 2024

Provides utilities for serialising and deserialising objects to and from JSON.
"""

import json
import numpy as np
import io
import networkx as nx
from . import EdgeSet
from synthesis.weight_generator import Weighter

class JSONEncoder(json.JSONEncoder):
    "Custom Numpy to JSON Encoder - the JSON Module won't encode numpy objects"
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, EdgeSet):
            return list(obj)
        if isinstance(obj, Weighter):
            return obj.toJSON()
        return super(JSONEncoder, self).default(obj)
    
def to_json(obj):
    "Converts an object to a JSON string"
    return json.dumps(obj, cls=JSONEncoder)

def to_edgelist(G):
    # Convert the graph to a string using nx.to_edgelist and writing
    # to a buffer
    buffer = io.BytesIO()
    if nx.is_weighted(G):
        nx.write_weighted_edgelist(G, buffer)
    else:
        nx.write_edgelist(G, buffer)
    buffer.seek(0)
    return buffer.read()

def from_edgelist(edgelist):
    # Convert the edgelist to a bytes-like object
    #edgelist = edgelist.encode('utf-8')
    buffer = io.BytesIO(edgelist)
    #G = nx.parse_edgelist(edgelist_file, nodetype=int, data=(('weight', float),))
    G = nx.read_weighted_edgelist(buffer)
    return G

