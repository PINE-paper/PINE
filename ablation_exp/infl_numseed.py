import networkx as nx
import json
import os
import torch
import numpy as np
from src.data_preparation.prepare_graph import get_graph
from src.utils import sort_nodes
from src.simulation.utils import get_simulation_result
from src.data_preparation.graph_construction import construct_graph
from src.baselines.measure_mapping import measure_funcs
from tqdm import tqdm
import time
import csv
import argparse

    
    
def influence_node_ratio(dataset_name, propagation_model_name, ratios, measure_names, res_folder, file_name, num_runs=1000):
    
    feat, graph_edges = get_graph(dataset_name)
    G, tw, aw = construct_graph(feat, graph_edges)
    
    os.makedirs(res_folder, exist_ok=True)
    res_path = os.path.join(res_folder, f'{file_name}_{dataset_name}_{propagation_model_name}.json')
    
    infl_res = {}
    infl_res['ratio'] = ratios
    for measure_name in measure_names:
        infl_res[measure_name] = []
        imp_dict = measure_funcs[measure_name](G)
        imp_nodes = sort_nodes(imp_dict)
        for r in ratios:
            num_starts = [np.round(feat.shape[0]*r).astype(np.int64)]
            if propagation_model_name == 'LT+':
                sim_lt = get_simulation_result(G, imp_nodes, tw, aw, 'LT', num_starts, num_runs)
                sim_lt_mean, sim_lt_std = sim_lt[0][0], sim_lt[1][0]
                infl_res[measure_name].append([sim_lt_mean, sim_lt_std])
            elif propagation_model_name == 'IC+':
                sim_ic = get_simulation_result(G, imp_nodes, tw, aw, 'IC', num_starts, num_runs)
                sim_ic_mean, sim_ic_std = sim_ic[0][0], sim_ic[1][0]
                infl_res[measure_name].append([sim_ic_mean, sim_ic_std])
            elif propagation_model_name == 'SIR':
                sim_sir = get_simulation_result(G, imp_nodes, tw, aw, 'SIR', num_starts, num_runs)
                sim_sir_mean, sim_sir_std = sim_sir[0][0], sim_sir[1][0]
                infl_res[measure_name].append([sim_sir_mean, sim_sir_std])
            else:
                print('Unknown propagation model')

    with open(res_path, 'w') as f:
        json.dump(infl_res, f)
    return



if __name__ == "__main__":
    dataset_name = 'citeseer'
    propagation_model_name = 'LT+'
    ratios = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    measure_names = ['out_degree', 'ilc', 'h_index', 'pagerank', 'voterank', 'enrenew', 'betweenness', 'eddc', 'pine_base']
    res_folder = 'ablation_results'
    file_name = 'influence_on_ratio'
    influence_node_ratio(dataset_name, propagation_model_name, ratios, measure_names, res_folder, file_name)
