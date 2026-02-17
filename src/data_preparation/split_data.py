import numpy as np
import random
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import torch


def transductive_edge_split(graph_edges, num_nodes, num_val, num_test, disjoint_train_ratio):
    transform = RandomLinkSplit(num_val=num_val, 
                                num_test=num_test,
                                is_undirected=False, 
                                add_negative_train_samples=True,
                                neg_sampling_ratio=1.0,
                                key = "edge_label", # supervision label
                                disjoint_train_ratio=disjoint_train_ratio, # disjoint mode if > 0
                                )    
    graph_data = Data(edge_index=torch.tensor(graph_edges), num_nodes=num_nodes)
    train_edges, val_edges, test_edges = transform(graph_data)
    return train_edges, val_edges, test_edges
