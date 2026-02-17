# PINE🌲: Priority-based Important Node Exploration

PINE is a novel graph measure that achieves state-of-the-art performance in unsupervised node ranking and offers 10x better scalability compared to prior measures. PINE’s design is inspired by Graph Attention Network (GAT), thus providing unprecedented evidence that GATs inherently encode a powerful node centrality. 

<img src="./pictures/pine_scheme1.png" width="40%">

The result of PINE work is an identified set of important nodes:

<img src="./pictures/graph_citeseer1.png" width="60%">

# 🚀 Launch PINE


### Step 1. Set environment
To start, create conda environment with the name 'pine_env' with proper dependencies by running: 
```
conda env create --file=environment.yml
```
Then, activate it:
```
conda activate pine_env
```


### Step 2. Get data

We provide the studied datasets in [https://drive.google.com/drive/folders/10gtjmNtiOMKBSQ906t3lUIx4OXF7mQkD?usp=share_link](https://drive.google.com/drive/folders/10gtjmNtiOMKBSQ906t3lUIx4OXF7mQkD?usp=share_link). Seven attributed homogeneous networks are under consideration: Cora, CiteSeer, PubMed, Wiki-CS, HEP-TH, ogbn-Arxiv, and DBLP.
To download them into folder `data`, run:
```
bash bin/get_data.sh
```
ogbn-Arxiv dataset is available from [Open Graph Benchmark](https://ogb.stanford.edu/docs/nodeprop/). 

The datasets, like Cora, CiteSeer, PubMed, Wiki-CS, and ogbn-Arxiv come with already prepared embeddings for text attributes in nodes. For the DBLP dataset, we use graph with node embeddings prepared in [TAG-benchmark](https://github.com/sktsherlock/TAG-Benchmark) (roberta_base_512_cls model). For HEP-TH dataset, we utilize [PhysBERT](https://huggingface.co/thellert/physbert_cased) model to infer embeddings (check [python script](src/data_preparation/get_embeds_hepth.py) for that, transformers library is needed). 


### Step 3. Run methods
Here, we compare PINE with a set of traditional centrality measures. 

**Problem setup**. A goal of each method is to associate every node in a graph with an importance score. The larger assigned score the more important the node is considered to be in the network. 

**Evaluation of results**. 
To compare methods' performance, we adopt a simulation-based procedure. Each method associates importance scores with nodes. Then, top-K nodes are taken as seed nodes for the start of information diffusion process. At the end, an influence spread over the network is evaluated. We assume that the method is better than others if it identifies nodes, which lead to the greatest spread of information. 

**Propagation models**.
As we consider attributed networks, it is important to simulate information diffusion taking into account node attributes. To do this, each edge of the graph is associated with topology and attribute weights. Then, these weights are used in such propagation models as **Linear Threshold (LT+)** and **Independent Cascade (IC+)**. Plus sign in their names indicates their attribute-awareness. In addition, we utilize a classical **SIR** propagation model, but it relies only on the graph structure. The implementations of propagation models are given in a folder `src/simulation`: [**LT+**](src/simulation/LT_plus.py), [**IC+**](src/simulation/IC_plus.py), and [**SIR**](src/simulation/SIR.py).

**Launch script**

```
python src/run_methods.py \
--dataset_names 'cora' 'citeseer' 'pubmed' 'wiki-cs' 'hepth' 'ogbn-arxiv' 'dblp' \
--measure_names 'pine' 'out-degree' 'pagerank' 'voterank' 'betweenness' 'enrenew' \
--propagation_model_names 'LT+' 'IC+' 'SIR' \
--res_folder './simulation_results' \
--device 'cuda:0' \
--node_ratio 0.1 \
--num_runs 1000
```

* `dataset_names` is names of networks, on which different measures are compared.
* `measure_names` is names of measures for node importance estimation.
* `propagation_model_names` pointa out which propagation models to use for a simulation of information dissemination in a network.
* `device` is used for PINE training. Other graph measures are calculated on cpu by default.
* `node_ratio` is a part of the nodes from which information dissemination process starts. 0.1 means that 10\% of nodes with the greatest importance scores are initialized as active.
* `num_runs` is a number of Monte-Carlo simulation runs.
* `res_folder` will contain the results that include optimized hyperparametrs of GAT, which is trained within PINE, and csv files with influence spread values for the selected measures under the specified propagation models.


# Case of heterogeneous network
We explore a heterogeneous network called FB15K, which is a subset from a FreeBase. The data preprocessing step follows the precedure outlined in [repo](https://github.com/yankai-chen/EASING) on the basis of the [work](https://arxiv.org/pdf/2503.20697). 

To download a heterogeneous network FB15K into a folder `heterogeneous_data`, run:
```
bash bin/get_heterogeneous_data.sh
```

The structure of data files in the folder `FB15K` is the following:
* `fb15k_rel.pk` - file with graph data and structural features of nodes
* `fb_lang.pk` - file with semantic features of nodes
* `idx_1000` - folder that contains data splits with node IDs and their labels (ground truth markup of node importance scores)

In a heterogeneous case, PINE considers the subgraph of the most popular edge type. Refer to the [**notebook**](notebooks/PINE_heterogeneous.ipynb) for computing node importance scores with PINE and othe graph measures on FB15K.

