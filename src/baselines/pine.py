import networkx as nx
import numpy as np


def get_pine_base_nodes(G):
    num_nodes = G.number_of_nodes()

    edges = np.array(
        [(u, v) for u, v in G.edges() if u != v],
        dtype=np.int64
    )
    
    self_loops = np.arange(num_nodes, dtype=np.int64)
    
    edges = np.vstack((
        edges,
        np.column_stack((self_loops, self_loops))
    ))

    src, dst = edges.T
    indeg = np.bincount(dst, minlength=num_nodes)
    contrib = 1.0 / indeg[dst]
    pine_vals = np.bincount(src, weights=contrib, minlength=num_nodes)
    pine_vals[pine_vals == 0] = 1.0

    pine_dict = {k: v for k, v in dict(enumerate(pine_vals)).items() if k in G}
    return pine_dict
