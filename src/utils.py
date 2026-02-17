import numpy as np
import torch

def topology_weights(G):
    """"
    Calculate topology weights for edges
    G: directed weighted networkx graph
    return: dict with edges as keys and topology weights as values
    """
    tw = dict()
    for u in G:
        in_edges = G.in_edges(u)
        indegree = len(in_edges)
        for (v, _) in in_edges:
            tw[(v, u)] = 1 / indegree
    return tw


def attribute_weights(G):
    """"
    Calculate attribute weights for edges and normalize them
    G: directed weighted networkx graph
    return: dict with edges as keys and attribute weights as values
    """
    aw = dict()
    for u in G:
        in_edges = G.in_edges(u)
        total = 0
        for (v, _) in in_edges:
            total += np.exp(G[v][u]['weight'])
        for (v, _) in in_edges:
            aw[(v, u)] = np.exp(G[v][u]['weight']) / total
    return aw


def cos_sim(emb1, emb2):
    if np.linalg.norm(emb1) * np.linalg.norm(emb2) > 0:
        return np.dot(emb1, emb2)/(np.linalg.norm(emb1) * np.linalg.norm(emb2))
    else:
        return 0
    

def sort_nodes(imp_dict):
    imp_sorted = {k: v for k, v in sorted(imp_dict.items(), key=lambda item: item[1], reverse=True)}
    imp_nodes = list(imp_sorted.keys())
    return imp_nodes
