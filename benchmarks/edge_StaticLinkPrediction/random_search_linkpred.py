import argparse
import math
import time
import dgl
# from torch_geometric.nn import GCNConv, SAGEConv, GATConv
import numpy as np
# import numpy_indexed as npi
import scipy.sparse as ssp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.utils.data import DataLoader
from dgl.sampling import random_walk
# from torch_cluster import random_walk
import os.path as osp
# from gen_model import gen_model
from loss import calculate_loss
from utils import *
import random
import os
import sys
from sklearn.metrics import roc_auc_score
import datetime
from tqdm import tqdm 
import subprocess
import argparse
import time
import pandas
from copy import deepcopy
from utils import *
import warnings
import torch
from torch_geometric.data import Data
from torch_geometric.nn import Node2Vec

import shutil
import yaml
warnings.filterwarnings("ignore")
seed_list = list(range(0, 9))

        
parser = argparse.ArgumentParser(description='OGBL-NFTGraph (GNN)')
parser.add_argument('--device', type=int, default=2)
parser.add_argument('--dataset', type=str, default='ogbl-nftgraph')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--model', type=str, default='matfac')
parser.add_argument('--clip-grad-norm', type=float, default=1)


parser.add_argument('--no-node-feat', action='store_true')
parser.add_argument('--batchsize', type=int, default=6400)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--trials', type=int, default=100)  # Number of random search
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--metric', type=str, default='rocauc')
parser.add_argument("--val_percent", type=float, default=0.3)

args = parser.parse_args()

prefix = '/data/sx/NFTGraph'
res_dir = os.path.join(prefix,f'linkpred/results',f'{args.dataset.split("-")[1]}-{args.model}',)
if os.path.exists(res_dir):
    shutil.rmtree(res_dir)

os.makedirs(res_dir)

def print_log(*x, sep="\n", end="\n", mode="a"):
    print(*x, sep=sep, end=end)
    with open(log_file, mode=mode) as f:
        print(*x, sep=sep, end=end, file=f)

log_file = os.path.join(res_dir, "log.txt")
cmd_input = "python " + " ".join(sys.argv) + "\n"

print_log(f"Results will be saved in {res_dir}")
print_log(f"{cmd_input}")
print_log(f"Command line input is saved.")

prefix = '/data/sx/NFTGraph'
dataset_name = args.dataset.split("-")[1]

meta_dict = torch.load(prefix+f'/datasets/ogb_graph/submission_ogbl_{dataset_name}/meta_dict.pt')
meta_dict['dir_path'] = prefix+f'/datasets/ogb_graph/submission_ogbl_{dataset_name}/{dataset_name}'

dataset = DglLinkPropPredDataset(name = dataset_name, root = meta_dict['dir_path'] ,meta_dict=meta_dict)

graph = dataset[0]
split_edge = dataset.get_edge_split()

evaluator = Evaluator(name=args.dataset)

device = (f"cuda:{args.device}" if args.device != -1 and torch.cuda.is_available() else "cpu")
device = torch.device(device)

idx = torch.randperm(int(split_edge['valid']['edge'].size(0)*args.val_percent))
idx = idx[:int(split_edge['valid']['edge'].size(0)*args.val_percent)]
split_edge['eval_train'] = {'edge': split_edge['valid']['edge'][idx],'edge_neg': split_edge['valid']['edge_neg'][idx]}

graph = graph.remove_self_loop().add_self_loop()
graph = graph.to(device)

feat = graph.ndata['feat'].float()
edge_feat = None
in_feats = feat.shape[1]

columns = [dataset_name+'-'+args.metric]

results = pd.DataFrame(columns=columns)
best_model_configs = {}
file_id = None

models = args.model.split('-')
print_log('Evaluated Baselines: ', models)

for model_name in models:
    model_result = {'name': model_name}
    best_model_configs[model_name] = {}
    print_log('============Dataset {} Model {}=============='.format(dataset_name, model_name))
    
    time_cost = 0
    best_val_score = 0
    best_finalval, best_test = 0,0
    train_config = {
        'device': device,
        'epochs': args.epochs,  #traing epoch for models
        'batch_size': args.batchsize,
        'patience': 30,
        'metric': 'rocauc',
        'inductive': False
    }
    
    train_config['batch_size'] = args.batchsize
    for t in range(args.trials):
        torch.cuda.empty_cache()
        print_log("Dataset {}, Model {}, Trial {}, Time Cost {:.2f}".format(dataset_name, model_name, t, time_cost))
        if time_cost > 86400:  # 86400 Stop after 1 day
            break
        set_seed(args.seed+t)
        train_config['seed'] = args.seed+t
        st = time.time()
        
        tmp = graph.ndata['feat']
        tmp = (tmp - torch.mean(tmp,dim=0))/ torch.std(tmp,dim=0)
        graph.ndata['feat'] = tmp
    
        train_graph = dgl.graph((split_edge['train']['edge'][:,0],split_edge['train']['edge'][:,1]),num_nodes=graph.num_nodes()).to(device)
        train_graph.ndata['feat'] = graph.ndata['feat']
        train_graph = train_graph.remove_self_loop().add_self_loop()
        
        model_config = sample_param(model_name, dataset_name, t)
        model_config['graph'] = train_graph
        model, predictor = gen_model(model_name, model_config, in_feats, train_config['device'])
        parameters = list(model.parameters()) + list(predictor.parameters())

        num_param = count_parameters(model) + count_parameters(predictor)
        print_log(f"Total number of parameters is {num_param}")

        patience_knt = 0
        for e in range(train_config['epochs']):
            if model_name == 'node2vec':
                loss = train_node2vec(model, predictor, feat, edge_feat, train_graph, split_edge, model_config, train_config['batch_size'])
            else:
                optimizer = torch.optim.Adam(parameters,lr=model_config['lr'])
                loss = train(model, predictor, feat, edge_feat, train_graph, split_edge, optimizer, train_config['batch_size'],model_config['clip_grad_norm'])
            print_log('Loss: {:.4f}'.format(loss))
            
            val_score = test(evaluator, model, 'valid', predictor, feat, edge_feat, train_graph, split_edge, train_config['batch_size'])
            if val_score[train_config['metric']] > best_val_score:
                # print_log("****current best score****")
                best_val_score = val_score[train_config['metric']]
                best_model_config = deepcopy(model_config)
                final_val_score = test(evaluator, model, 'final_val', predictor, feat, edge_feat, train_graph, split_edge, train_config['batch_size'])
                
                train_graph.add_edges(split_edge['valid']['edge'][:,0].to(device),split_edge['valid']['edge'][:,1].to(device))
                test_score = test(evaluator, model, 'test', predictor, feat, edge_feat, train_graph, split_edge, train_config['batch_size'])
                best_finalval, best_test  = final_val_score[train_config['metric']] ,test_score[train_config['metric']]
                best_model_config['epoch'] = e
            else:
                patience_knt += 1
                if patience_knt > train_config['patience']:
                    break
        ed = time.time()
        time_cost += ed - st
        print_log("Current Val Best:{:.4f}; Final Val:{:.4f}; Test:{:.4f}".format(
            best_val_score, best_finalval, best_test))
        del model

    print_log("best_model_config:", best_model_config)
    del best_model_config['graph']
    config_file = os.path.join(res_dir, "config.yaml")
    config = best_model_config
    with open(config_file,'w') as file:
        file.write(yaml.dump(config))

    # best_model_configs[model_name][dataset_name] = deepcopy(best_model_config)
    model_result[dataset_name+'-finalval_score'] = best_finalval
    model_result[dataset_name+'-test_score'] = best_test
    model_result[dataset_name+'-Time'] = time_cost/(t+1)
    
    model_result = pd.DataFrame(model_result, index=[0])
    results = pd.concat([results, model_result])
    file_id = save_results(results, file_id)
