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


def compare_time_acc(datasets, measure_names, res_folder, file_name, num_launches):
    
    for dataset_name in datasets:
        feat, graph_edges = get_graph(dataset_name)

        G, tw, aw = construct_graph(feat, graph_edges)

        os.makedirs(res_folder, exist_ok=True)
        res_path = os.path.join(res_folder, f'{file_name}.csv')
        columns = ['measure_name', 'time_mean', 'time_std']
        with open(res_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            
        for measure_name in measure_names:
            row2write = [measure_name]
            running_times = []
            for _ in range(num_launches):
                start_time = time.time()
                _ = measure_funcs[measure_name](G)
                end_time = time.time()
                delta_time = end_time - start_time
                running_times.append(delta_time)

            row2write = row2write + [np.mean(running_times), np.std(running_times)]
            
            with open(res_path, 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(row2write)
    return


if __name__ == "__main__":
    datasets = ['wiki-cs']
    measure_names = ['pagerank', 'voterank', 'enrenew', 'betweenness', 'eddc', 'pine_base']
    res_folder = 'ablation_results'
    file_name = 'time_accuracy'
    num_launches = 10
    compare_time_acc(datasets, measure_names, res_folder, file_name, num_launches)
