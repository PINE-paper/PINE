import numpy as np


def laplacian_centrality_directed(G, mode='out'):
    """
    Compute Laplacian Centrality (LC) for directed graph G.
    
    Parameters:
    - G: nx.DiGraph
    - mode: 'out' for out-degree, 'in' for in-degree, 'total' for in+out-degree
    
    Returns:
    - LC: dict mapping node -> Laplacian Centrality
    """
    if mode == 'in':
        deg = dict(G.in_degree())
        neighbors_func = G.predecessors
    elif mode == 'out':
        deg = dict(G.out_degree())
        neighbors_func = G.successors
    elif mode == 'total':
        deg = {node: G.in_degree(node) + G.out_degree(node) for node in G.nodes()}
        neighbors_func = G.neighbors
    else:
        raise ValueError("mode must be 'in', 'out', or 'total'")
    
    LC = {}
    for node in G.nodes():
        d_i = deg[node]
        neighbor_sum = sum(deg[neighbor] for neighbor in neighbors_func(node))
        LC[node] = d_i * d_i + d_i + 2 * neighbor_sum
    return LC


def get_ilc_nodes(G, mode='out'):
    """
    Compute Improved Laplacian Centrality (ILC) for directed graph G.
    
    Parameters:
    - G: nx.DiGraph
    - mode: 'out' for out-degree, 'in' for in-degree, 'total' for in+out-degree
    
    Returns:
    - ILC: dict mapping node -> Improved Laplacian Centrality
    """
    LC = laplacian_centrality_directed(G, mode)
    
    if mode == 'in':
        neighbors_func = G.predecessors
    elif mode == 'out':
        neighbors_func = G.successors
    else:
        neighbors_func = G.neighbors
    
    ILC = {}
    for node in G.nodes():
        lc_i = LC[node]
        neighbor_lc_sum = sum(LC[neighbor] for neighbor in neighbors_func(node))
        ILC[node] = lc_i * lc_i + lc_i + 2 * neighbor_lc_sum
    return ILC
