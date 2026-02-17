import networkx as nx


def get_semilocal_centrality_nodes(G, mode="out"):
    """
    Compute semi-local centrality for a directed graph.

    Args:
        G (networkx.DiGraph): Directed graph.
        mode (str): "out" for influence-based (default),
                    "in" for prestige-based.

    Returns:
        dict: node -> semi-local centrality score
    """

    if mode == "out":
        neighbors = G.successors
    elif mode == "in":
        neighbors = G.predecessors
    else:
        raise ValueError("mode must be 'out' or 'in'")

    # Step 1: Compute N(w) = number of 1st + 2nd order neighbors
    N = {}
    for w in G.nodes():
        first = set(neighbors(w))

        second = set()
        for u in first:
            second.update(neighbors(u))

        # remove duplicates and self
        second.discard(w)
        second -= first

        N[w] = len(first) + len(second)

    # Step 2: Compute Q(u) = sum of N(w) over neighbors of u
    Q = {}
    for u in G.nodes():
        Q[u] = sum(N[w] for w in neighbors(u))

    # Step 3: Semi-local centrality
    C_SL = {}
    for v in G.nodes():
        C_SL[v] = sum(Q[u] for u in neighbors(v))

    return C_SL
