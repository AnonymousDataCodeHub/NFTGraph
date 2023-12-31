import argparse
import os
import warnings
import networkx as nx
import torch
import numpy as np
import random
import pandas as pd
from utils import *

warnings.filterwarnings("ignore")

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--datasets', type=str, default=None)
args = parser.parse_args()


datasets = []
if args.datasets is not None:
    datasets = args.datasets.split('-')
    print('Evaluated Datasets: ', datasets)

prefix = '/data/sx/NFTGraph'

file_id = None
for dataset_name in datasets:
    data = Dataset(dataset_name,prefix=prefix+'/datasets/dgl_graph/')
    graph = data.graph.to_networkx()
    mG = graph
    diG = nx.DiGraph(graph)
    uG = nx.Graph(graph)

    results = {}
    # 节点连边数
    nums_nodes = mG.number_of_nodes()
    nums_edges = mG.number_of_edges()
    print(dataset_name," nums_nodes:",nums_nodes)
    print(dataset_name," nums_edges:",nums_edges)

    # 密度
    density = nx.density(mG)
    print(dataset_name," density:",density)

    # 平均度
    avg_degree = np.mean(mG.degree())
    print(dataset_name," avg_degree:",avg_degree)

    # PageRank中心性
    t = nx.pagerank(mG)
    pagerank_centrality = np.mean(list(t.values()))
    print(dataset_name," pagerank_centrality:",pagerank_centrality)

    
    # assortativity
    assortativity = nx.degree_assortativity_coefficient(mG)
    print(dataset_name," assortativity:",assortativity)

    # transitivity
    transitivity = nx.transitivity(uG)
    print(dataset_name," transitivity:",transitivity)

    # Clustering Coeff
    average_clustering = nx.average_clustering(diG)
    print(dataset_name," average_clustering:",average_clustering)

    # Triangles
    triangles = np.sum(list(nx.triangles(uG).values()))
    print(dataset_name," triangles:",triangles)

    results[dataset_name+'-'+'nums_nodes'] = nums_nodes
    results[dataset_name+'-'+'nums_edges'] = nums_edges
    results[dataset_name+'-'+'density'] = density
    results[dataset_name+'-'+'avg_degree'] = avg_degree
    results[dataset_name+'-'+'pagerank_centrality'] = pagerank_centrality
    results[dataset_name+'-'+'assortativity'] = assortativity
    results[dataset_name+'-'+'transitivity'] = transitivity
    results[dataset_name+'-'+'average_clustering'] = average_clustering
    results[dataset_name+'-'+'triangles'] = triangles

    results = pd.DataFrame(results, index=['metric'])
    file_id = save_results(results, file_id)
    file_id += 1
    
    print(results)