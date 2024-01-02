from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.linkproppred.negative_generator import NegativeEdgeGenerator
from torch.utils.data import Dataset, DataLoader
import torch
from torch_geometric.data import Dataset, TemporalData

import numpy as np
import numpy as np
import random
import pandas as pd

from torch_geometric.loader import TemporalDataLoader
import os.path as osp

class Data:

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        
def get_link_pred_data_TRANS_TGB(dataset_name: str,prefix: str):
    """
    generate data for link prediction task (NOTE: transductive dynamic link prediction)
    load the data with the help of TGB and generate required format for DyGLib
    :param dataset_name: str, dataset name
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, (Data object)
    """
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 10  # a specific setting for consistency among baselines
 
    # data loading
    dataset = PyGLinkPropPredDataset(name=dataset_name, root=prefix+'/datasets/tgb_graph/')
    data = dataset.get_TemporalData()
    

    # get split masks
    train_mask = dataset.train_mask
    val_mask = dataset.val_mask
    test_mask = dataset.test_mask
    # get split data
    train_data = data[train_mask]
    val_data = data[val_mask]
    test_data = data[test_mask]

    # Load data and train val test split
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    edge_raw_features =  data.msg.numpy()
    node_raw_features = np.zeros((data.dst.size(0), NODE_FEAT_DIM))


    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], "Unaligned feature dimensions after feature padding!"

    src_node_ids = (data.src.numpy()+1).astype(np.longlong)
    dst_node_ids = (data.dst.numpy()+1).astype(np.longlong)
    node_interact_times = data.t.numpy().astype(np.float64)
    edge_ids = np.array([i for i in range(1, len(data.src)+1)]).astype(np.longlong)
    labels = data.y.numpy()

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, 
                    node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels)

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask])

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask])

    print("INFO: The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("INFO: The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("INFO: The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("INFO: The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset

dataset_name = 'tgbl-nftgraph'
prefix = '/data/sx/NFTGraph'
meta_dict = {}
dataset = PyGLinkPropPredDataset(name=dataset_name, root=prefix+'/datasets/tgb_graph/')
data = dataset.get_TemporalData()

# get data for training, validation and testing
node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, dataset = \
    get_link_pred_data_TRANS_TGB(dataset_name=dataset_name,prefix=prefix)


train_graph = TemporalData(
    src=train_data.src_node_ids,
    dst=train_data.dst_node_ids,
    t=train_data.node_interact_times,
    msg=None,
    y=train_data.labels,
)
        
ng_gen = NegativeEdgeGenerator(dataset_name='tgbl-nftgraph',first_dst_id=0,last_dst_id=1161846,strategy='rnd',historical_data=train_graph,num_neg_e=20)

ng_gen.generate_negative_samples(data, 'val', partial_path=prefix+'/datasets/tgb_graph/tgbl_nftgraph')
ng_gen.generate_negative_samples(data, 'test', partial_path=prefix+'/datasets/tgb_graph/tgbl_nftgraph')