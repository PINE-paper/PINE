

def compute_h_index(node, graph):
    """
    Compute the h-index of a node in an undirected graph.
    h-index: maximum h such that node has at least h neighbors
    with degree >= h.
    """
    # take the degrees of neighbors
    neighbor_degrees = [graph.degree(neigh) for neigh in graph.neighbors(node)]
    neighbor_degrees.sort(reverse=True)

    # find the largest h such that there are >= h neighbors with deg >= h
    h_index = 0
    for i, deg in enumerate(neighbor_degrees, start=1):
        if deg >= i:
            h_index = i
        else:
            break
    return h_index


def get_local_h_index_nodes(graph):
    """
    Compute the local h-index (LH-index) for all nodes in an undirected graph.
    Returns a dictionary: node -> LH-index value.
    """
    # First compute h-index for each node
    h_index_map = {node: compute_h_index(node, graph) for node in graph.nodes()}

    # Then sum up h-index of node + neighbors
    lh_index_map = {}
    for node in graph.nodes():
        neighbor_sum = sum(h_index_map[n] for n in graph.neighbors(node))
        lh_index_map[node] = h_index_map[node] + neighbor_sum

    return lh_index_map

