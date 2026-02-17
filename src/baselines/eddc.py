import networkx as nx
import numpy as np


def node_entropy(G, v):
    neighbors = list(G.neighbors(v))
    if len(neighbors) == 0:
        return 0.0

    degs = np.array([G.degree(u) for u in neighbors], dtype=float)
    total = degs.sum()
    if total == 0:
        return 0.0

    p = degs / total
    p_nonzero = p[p > 0]
    return -np.sum(p_nonzero * np.log2(p_nonzero))


def get_eddc_nodes(G):

    N = G.number_of_nodes()
    if N == 0:
        return {}

    entropy = {v: node_entropy(G, v) for v in G.nodes()}

    distances = {v: dict(nx.single_source_shortest_path_length(G, v)) for v in G.nodes()}

    eddc_scores = {}

    for v in G.nodes():
        deg_v = G.degree(v)
        if deg_v == 0:
            eddc_scores[v] = 0.0
            continue

        ev = entropy[v]

        sum_val = 0.0
        for u in G.nodes():
            if u == v:
                continue

            d = distances[v].get(u, None)
            if d is None or d <= 0:
                continue

            eu = entropy[u]
            sum_val += np.sqrt(ev + eu) / d

        eddc_scores[v] = (deg_v / N) * sum_val

    return eddc_scores
