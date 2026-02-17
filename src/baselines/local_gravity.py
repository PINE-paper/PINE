import networkx as nx
import numpy as np


def get_local_gravity_centrality_nodes(G, R=2, use_out_degree=True):
    """
    Compute Local Gravity Model centrality for directed graphs efficiently.
    Only considers nodes within truncation radius R.
    """
    # Degree as mass
    deg = dict(G.out_degree() if use_out_degree else G.in_degree())
    
    scores = {}
    for node in G.nodes():
        S = 0.0
        # BFS up to depth R
        lengths = nx.single_source_shortest_path_length(G, node, cutoff=R)
        for target, dist in lengths.items():
            if node == target or dist == 0:
                continue
            S += deg[node] * deg[target] / (dist * dist)
        scores[node] = S
    return scores

