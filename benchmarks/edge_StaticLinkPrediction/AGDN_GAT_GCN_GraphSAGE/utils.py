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

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def count_parameters(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def adjust_lr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

def to_undirected(graph):
    print(f'Previous edge number: {graph.number_of_edges()}')
    graph = dgl.add_reverse_edges(graph, copy_ndata=True, copy_edata=True)
    keys = list(graph.edata.keys())
    for k in keys:
        if k != 'weight':
            graph.edata.pop(k)
        else:
            graph.edata[k] = graph.edata[k].float()
    graph = dgl.to_simple(graph, copy_ndata=True, copy_edata=True, aggregator='sum')
    print(f'After adding reversed edges: {graph.number_of_edges()}')
    return graph

def filter_edge(split, nodes):
    mask = npi.in_(split['edge'][:,0], nodes) & npi.in_(split['edge'][:,1], nodes)
    print(len(mask), mask.sum())
    split['edge'] = split['edge'][mask]
    split['year'] = split['year'][mask]
    split['weight'] = split['weight'][mask]
    if 'edge_neg' in split.keys():
        mask = npi.in_(split['edge_neg'][:,0], nodes) & npi.in_(split['edge_neg'][:,1], nodes)
        split['edge_neg'] = split['edge_neg'][mask]
    return split


def precompute_adjs(A):
    '''
    0:cn neighbor
    1:aa
    2:ra
    '''
    w = 1 / A.sum(axis=0)
    w[np.isinf(w)] = 0
    w1 = A.sum(axis=0) / A.sum(axis=0)
    temp = np.log(A.sum(axis=0))
    temp = 1 / temp
    temp[np.isinf(temp)] = 0
    D_log = A.multiply(temp).tocsr()
    D = A.multiply(w).tocsr()
    D_common = A.multiply(w1).tocsr()
    return (A, D, D_log, D_common)


def RA_AA_CN(adjs, edge):
    A, D, D_log, D_common = adjs
    ra = []
    cn = []
    aa = []

    src, dst = edge
    # if len(src) < 200000:
    #     ra = np.array(np.sum(A[src].multiply(D[dst]), 1))
    #     aa = np.array(np.sum(A[src].multiply(D_log[dst]), 1))
    #     cn = np.array(np.sum(A[src].multiply(D_common[dst]), 1))
    # else:
    batch_size = 1000000
    ra, aa, cn = [], [], []
    for idx in tqdm(DataLoader(np.arange(src.size(0)), batch_size=batch_size, shuffle=False, drop_last=False)):
        ra.append(np.array(np.sum(A[src[idx]].multiply(D[dst[idx]]), 1)))
        aa.append(np.array(np.sum(A[src[idx]].multiply(D_log[dst[idx]]), 1)))
        cn.append(np.array(np.sum(A[src[idx]].multiply(D_common[dst[idx]]), 1)))
    ra = np.concatenate(ra, axis=0)
    aa = np.concatenate(aa, axis=0)
    cn = np.concatenate(cn, axis=0)

        # break
    scores = np.concatenate([ra, aa, cn], axis=1)
    return torch.FloatTensor(scores)


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