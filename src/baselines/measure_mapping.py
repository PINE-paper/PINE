from src.baselines.node_degree import get_degree_nodes
from src.baselines.out_degree import get_outdegree_nodes
from src.baselines.weighted_out_degree import get_weighted_outdegree_nodes
from src.baselines.relative_out_degree import get_relative_outdegree_nodes
from src.baselines.pagerank import get_pagerank_nodes
from src.baselines.katz import get_katz_nodes
from src.baselines.closeness_centrality import get_closeness_nodes
from src.baselines.betweenness_centrality import get_betweenness_nodes
from src.baselines.voterank import get_voterank_nodes
from src.baselines.improved_laplacian_centrality import get_ilc_nodes
from src.baselines.semilocal_centrality import get_semilocal_centrality_nodes
from src.baselines.h_index import get_local_h_index_nodes
from src.baselines.local_gravity import get_local_gravity_centrality_nodes
from src.baselines.eddc import get_eddc_nodes
from src.baselines.enrenew import get_enrenew_nodes
from src.baselines.pine
import get_pine_base_nodes


measure_funcs = {
    'degree': get_degree_nodes,
    'out_degree': get_outdegree_nodes,
    'weighted': get_weighted_outdegree_nodes,
    'relative': get_relative_outdegree_nodes,
    'pagerank': get_pagerank_nodes,
    'katz': get_katz_nodes,
    'closeness': get_closeness_nodes,
    'betweenness': get_betweenness_nodes,
    'voterank': get_voterank_nodes,
    'enrenew': get_enrenew_nodes,
    'ilc': get_ilc_nodes,
    'semilocal': get_semilocal_centrality_nodes,
    'h_index': get_local_h_index_nodes,
    'gravity': get_local_gravity_centrality_nodes,
    'eddc': get_eddc_nodes,
    'pine': get_pine_base_nodes,
}
