import os
import os.path as osp
import sys

from numpy.random import seed
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parents[2]))

import torch
import numpy as np
import pandas as pd
import scipy
import argparse
import networkx as nx
import torch_geometric.transforms as T
from tgb.linkproppred.dataset import LinkPropPredDataset


dataset = LinkNFTGraph(root='./data', 
                          name=args.dataset,
                          splitting_strategy=args.splitting_strategy,
                          number_of_workers = args.number_of_workers,
                          val_ratio = args.train_val_test[1],
                          test_ratio = args.train_val_test[2],
                          seed=args.seed,
                          )


