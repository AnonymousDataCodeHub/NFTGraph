import random

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
# import numpy_indexed as npi
import dgl
from tqdm import tqdm
import random
import sys

import numpy as np
import torch
from dgl.sampling import global_uniform_negative_sampling
from scipy.sparse.csgraph import shortest_path
import pandas as pd
import shutil, os
import os.path as osp
import torch
import numpy as np
from dgl.data.utils import load_graphs, save_graphs, Subset
import dgl
from ogb.utils.url import decide_download, download_url, extract_zip
from ogb.io.read_graph_dgl import read_graph_dgl, read_heterograph_dgl
from ogb.utils.torch_util import replace_numpy_with_torchtensor


from sklearn.metrics import roc_auc_score
import pandas as pd
import os
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
from models import *


def compute_pred(h, predictor, edges, batch_size):
    preds = []
    for perm in DataLoader(range(edges.size(0)), batch_size):
        edge = edges[perm].t()
        preds += [predictor(h[edge[0]], h[edge[1]]).sigmoid().squeeze().cpu().view(-1)]
    pred = torch.cat(preds, dim=0)
    return pred


def train_node2vec(model, predictor, feat, edge_feat, graph, split_edge, model_config, batch_size):
    loader = model.loader(batch_size)
    if model.use_sparse:
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=model_config['lr'])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['lr'])
    model.train()
    total_loss = 0
    for pos_traces, neg_traces in tqdm(loader,ncols=70):
        pos_traces, neg_traces = pos_traces.to(feat.device), neg_traces.to(feat.device)
        optimizer.zero_grad()
        loss = model.loss(pos_traces, neg_traces)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def train(model, predictor, feat, edge_feat, graph, split_edge, optimizer, batch_size,clip_grad_norm):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(feat.device)
    edge_weight_margin = None
    neg_train_edge = split_edge['train']['edge_neg'].to(feat.device)

    total_loss = total_examples = 0
    for perm in tqdm(DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True),ncols=70):
        optimizer.zero_grad()

        h = model(graph, feat, edge_feat)
        
        edge = pos_train_edge[perm]
        neg_edge = neg_train_edge[perm]

        pos_out = predictor(h[edge[:, 0]], h[edge[:, 1]])
        # pos_loss = -torch.log(pos_out + 1e-15).mean()

        neg_out = predictor(h[neg_edge[:,0]], h[neg_edge[:,1]])
        # neg_loss = -torch.log(1 - neg_out + 1e-15).mean()
        # loss = pos_loss + neg_loss
        weight_margin = edge_weight_margin[perm].to(feat.device) if edge_weight_margin is not None else None

        loss = calculate_loss(pos_out, neg_out, 0 , margin=weight_margin, loss_func_name='CE')
        # cross_out = predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,0].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,1].view(-1, args.n_neg)]) + \
        #             predictor(h[edge[:,1].view(-1, 1)], h[neg_edge[:,0].view(-1, args.n_neg)])
        # cross_loss = -torch.log(1 - cross_out.sigmoid() + 1e-15).sum()
        # loss = loss + 0.1 * cross_loss
        loss.backward()

        if clip_grad_norm > -1:
            # if 'feat' not in graph.ndata:
            #     torch.nn.utils.clip_grad_norm_(feat, args.clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad_norm)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(evaluator, model, mode, predictor, feat, edge_feat, graph,  split_edge, batch_size):
    model.eval()
    predictor.eval()

    if mode == 'valid':
        pos_valid_edge = split_edge['eval_train']['edge'].to(feat.device)
        neg_valid_edge = split_edge['eval_train']['edge_neg'].to(feat.device)
        tofeed_pos_edge = pos_valid_edge
        tofeed_neg_edge = neg_valid_edge
    elif mode == 'final_val':
        pos_finalval_edge = split_edge['valid']['edge'].to(feat.device)
        neg_finalval_edge = split_edge['valid']['edge_neg'].to(feat.device)
        tofeed_pos_edge = pos_finalval_edge
        tofeed_neg_edge = neg_finalval_edge
    elif mode == 'test':
        pos_test_edge = split_edge['test']['edge'].to(feat.device)
        neg_test_edge = split_edge['test']['edge_neg'].to(feat.device)
        tofeed_pos_edge = pos_test_edge
        tofeed_neg_edge = neg_test_edge

    h = model(graph, feat, edge_feat)

    pos_y_pred = compute_pred(h, predictor, tofeed_pos_edge, batch_size)
    neg_y_pred = compute_pred(h, predictor, tofeed_neg_edge, batch_size)

    # results = evaluate_mrr(evaluator, pos_y_pred, neg_y_pred)
    
    results = {}
    y_true = torch.cat([torch.ones(pos_y_pred.size(0)),torch.zeros(neg_y_pred.size(0))],dim=0)
    y_pred = torch.cat([pos_y_pred,neg_y_pred],dim=0)
    results['rocauc'] = roc_auc_score(y_true, y_pred)

    return results


def evaluate_hits(y_pred_pos, y_pred_neg, hits_K):
    results = {}
    hits_K = map(
        lambda x: (int(x.split("@")[1]) if isinstance(x, str) else x), hits_K
    )
    for K in hits_K:
        evaluator.K = K
        hits = evaluator.eval(
            {
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
            }
        )[f"hits@{K}"]

        results[f"hits@{K}"] = hits

    return results


def evaluate_mrr(evaluator,y_pred_pos, y_pred_neg):
    y_pred_neg = y_pred_neg.view(y_pred_pos.shape[0], -1)
    results = {}
    mrr = (
        evaluator.eval(
            {
                "y_pred_pos": y_pred_pos,
                "y_pred_neg": y_pred_neg,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    results["mrr"] = mrr

    return results

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
def gen_model(model_name, model_config, in_feats,device):

    if model_name == 'gcn':
        model = GCN(in_feats, model_config['n_hidden'],
                    model_config['n_hidden'], model_config['n_layers'],
                    model_config['dropout'], model_config['input_drop'],
                    bn=model_config['bn'], residual=model_config['residual']).to(device)
    if model_name == 'gat':
        model = GAT(in_feats, model_config['n_hidden'],
                    model_config['n_hidden'], model_config['n_layers'],
                    model_config['n_heads'],
                    model_config['dropout'], model_config['input_drop'], model_config['attn_drop'],
                    bn=model_config['bn'], residual=model_config['residual']).to(device)
    if model_name == 'sage':
        model = SAGE(in_feats, model_config['n_hidden'],
                     model_config['n_hidden'], model_config['n_layers'],
                     model_config['dropout'], model_config['input_drop'],
                     bn=model_config['bn'], residual=model_config['residual']).to(device)
    if model_name == 'agdn':
        model = AGDN(in_feats, model_config['n_hidden'],
                     model_config['n_hidden'], model_config['n_layers'],
                     model_config['n_heads'], model_config['K'],
                     model_config['dropout'], model_config['input_drop'], 
                     model_config['attn_drop'], model_config['edge_drop'], model_config['diffusion_drop'],
                     bn=model_config['bn'], output_bn=model_config['output_bn'],
                     transition_matrix=model_config['transition_matrix'],
                     no_dst_attn=model_config['no_dst_attn'],
                     hop_norm=model_config['hop_norm'],
                     weight_style=model_config['weight_style'],
                     pos_emb=not model_config['no_pos_emb'],
                     share_weights=not model_config['no_share_weights'],
                     residual=model_config['residual'],
                     pre_act=model_config['pre_act']).to(device)
    if model_name == 'memagdn':
        model = MemAGDN(in_feats, model_config['n_hidden'],
                     model_config['n_hidden'], model_config['n_layers'],
                     model_config['n_heads'], model_config['K'],
                     model_config['dropout'], model_config['input_drop'], model_config['attn_drop']).to(device)
    if model_name == 'mlp':
        model = MLP(in_feats, model_config['n_hidden'], model_config['n_hidden'],
                                model_config['n_layers'], model_config['dropout']).to(device)
    if model_name == 'node2vec':
        model = Node2vec(model_config['graph'],model_config['n_hidden'],model_config['walk_length'],
                         model_config['p'],model_config['q'],model_config['num_walks']).to(device)
    if model_name == 'matfac':
        model = MF(model_config['graph'].num_nodes(),model_config['n_hidden']).to(device)
    # n_heads = model_config['n_heads if model_name in ['gat', 'agdn'] else 1
    if model_config['predictor'] == 'MLP':
        predictor = LinkPredictor(model_config['n_hidden'], model_config['n_hidden'], 1,
                                model_config['n_layers'], model_config['dropout']).to(device)
    if model_config['predictor'] == 'DOT':
        predictor = DotPredictor().to(device)

    if model_config['predictor'] == 'COS':
        predictor = CosPredictor().to(device)
    return model, predictor



class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.runs = runs
        self.results = {
            "valid": [[] for _ in range(runs)],
            "test": [[] for _ in range(runs)],
        }

    def add_result(self, run, result, split="valid"):
        assert run >= 0 and run < len(self.results["valid"])
        assert split in ["valid", "test"]
        self.results[split][run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        # if run is not None:
        #     result = torch.tensor(self.results["valid"][run])
        #     print(f"Run {run + 1:02d}:", file=f)
        #     print(f"Highest Valid: {result.max():.4f}", file=f)
        #     print(f"Highest Eval Point(Eval_epoch): {result.argmax().item()+1}", file=f)
        #     if not self.info.no_test:
        #         print(
        #             f'   Final Test Point[1](True Epoch): {self.results["test"][run][0][0]}',
        #             f'   Final Valid: {self.results["test"][run][0][1]}',
        #             f'   Final Test: {self.results["test"][run][0][2]}',
        #             sep="\n",
        #             file=f,
        #         )
        if (run+1) % 3 == 0 or ((run+1) == self.runs):
            # print(self.results["test"])
            best_result = torch.tensor(
                [test_res[0] for test_res in self.results["test"][:run+1]]
            )

            print(f"Runs 1-{run+1}:", file=f)
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.4f} ± {r.std():.4f}", file=f)
            if not self.info.no_test:
                r = best_result[:, 2]
                print(f"   Final Test: {r.mean():.4f} ± {r.std():.4f}", file=f)


def set_seed(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def print_log(*x, sep="\n", end="\n", mode="a"):
    print(*x, sep=sep, end=end)
    with open(log_file, mode=mode) as f:
        print(*x, sep=sep, end=end, file=f)
        
        
def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id


def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0.0,'clip_grad_norm':1}
    # if t == 0:
    #     return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v,)
    # Avoid OOM in Random Search
    # if model in ['gat']:
    #     model_config['n_hidden'] = 16
    #     model_config['n_heads'] = 2
    #     model_config['predictor'] = 'DOT'
    if model in ['agdn']:
        model_config['n_hidden'] = 16
        model_config['n_heads'] = 2
        if model_config['weight_style'] == 'lstm':
            model_config['weight_style'] = 'sum'
    if model in ['mlp']:
        model_config['predictor'] = 'MLP'
    if model in ['node2vec']:
        model_config['predictor'] = 'DOT'
    return model_config

param_space = {}
activations = {
    'ReLU': torch.nn.functional.relu,
    'LeakyReLU': torch.nn.functional.leaky_relu,
    'Tanh': torch.nn.functional.tanh}

model_dict = ['agdn','gat','gcn','sage','memagdn','node2vec','mlp','matfac']

for model in model_dict:
    param_space[model] = {
    'n_hidden': [16],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'input_drop': [0.0, 0.1, 0.2, 0.3],
    'bn': [True,False],
    'n_layers': [2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':[1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
    'residual': [True,False],
    'attn_drop':[0.0, 0.1, 0.2, 0.3],
    'n_heads': [2],
    'edge_drop':[0.0, 0.1, 0.2, 0.3],
    'diffusion_drop':[0.0, 0.1, 0.2, 0.3],
    'output_bn': [True,False],
    'transition_matrix': ['gat'],
    'no_dst_attn': [True,False],
    'hop_norm': [True,False],
    'weight_style': ['HC', 'HA', 'HA+HC', 'sum', 'max_pool', 'mean_pool', 'lstm'],
    'no_pos_emb':[True],
    'no_share_weights':[True],
    'pre_act': [False],
    'K':[3],
    'predictor':['MLP','DOT'],
    'walk_length':[5],
    'p':[0.25,1.0],
    'q':[0.25,1.0],
    'num_walks':[10]
    }
    