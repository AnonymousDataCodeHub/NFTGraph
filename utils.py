import random
from models.detector import *
from dgl.data.utils import load_graphs
import os
from pygod.metrics import eval_precision_at_k,eval_recall_at_k,eval_roc_auc,eval_average_precision
from pygod.models import DOMINANT,AnomalyDAE,DONE,AdONE,GAAN,GCNAE,CONAD,MLPAE
import torch.nn.functional

class Dataset:
    def __init__(self, name='tfinance', prefix='/datasets'):
        graph = load_graphs(prefix + name)[0][0]
        self.name = name
        self.graph = graph

    def split(self, semi_supervised=True, trial_id=0):
        if semi_supervised:
            trial_id += 10
        self.graph.ndata['train_mask'] = self.graph.ndata['train_masks'][:,trial_id]
        self.graph.ndata['val_mask'] = self.graph.ndata['val_masks'][:,trial_id]
        self.graph.ndata['test_mask'] = self.graph.ndata['test_masks'][:,trial_id]
        print(self.graph.ndata['train_mask'].sum(), self.graph.ndata['val_mask'].sum(), self.graph.ndata['test_mask'].sum())


# supervised models
model_detector_dict = {
    # Classic Methods
    'MLP': BaseGNNDetector,
    'KNN': KNNDetector,
    'SVM': SVMDetector,
    'RF': RFDetector,

    # Standard GNNs
    'GCN': BaseGNNDetector,
    'SGC': BaseGNNDetector,
    'GIN': BaseGNNDetector,
    'GraphSAGE': BaseGNNDetector,
    'GAT': BaseGNNDetector,
    'GT': BaseGNNDetector,

    # Specialized GNNs
    'GAS': GASDetector,
    'BernNet': BaseGNNDetector,
    'AMNet': BaseGNNDetector,
    'GHRN': GHRNDetector,
    'GATSep': BaseGNNDetector,
    'PCGNN': PCGNNDetector,
}

# unsupervised models
model_dict = {
    'DONE': DONE,
    'CONAD': CONAD,
    'AnomalyDAE': AnomalyDAE,
    'AdONE': AdONE,
    'GCNAE': GCNAE,
    'MLPAE': MLPAE,
    'DOMINANT':DOMINANT,
    'GAAN':GAAN
}


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
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)
    # Avoid OOM in Random Search
    if model in ['GAT', 'GATSep', 'GT']:
        model_config['h_feats'] = 16
        model_config['num_heads'] = 2
    if dataset == 'tsocial':
        model_config['h_feats'] = 16
    return model_config


param_space = {}

param_space['MLP'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GCN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['SGC'] = {
    'h_feats': [16, 32, 64],
    'k': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GIN'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['sum', 'max', 'mean'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GraphSAGE'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3],
    'agg': ['mean', 'gcn', 'pool'],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['BernNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'mlp_layers': [1, 2],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3, 4, 5],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['AMNet'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'orders': [2, 3],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GAS'] = {
    'h_feats': [16, 32, 64],
    'num_layers': [1, 2, 3, 4],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'k': range(3, 51),
    'dist': ['euclidean', 'cosine'],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['GHRN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': 10 ** np.linspace(-2, -1, 1000),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'mlp_layers': [1, 2],
}

param_space['RF'] = {
    'n_estimators': list(range(10, 201)),
    'criterion': ['gini', 'entropy'],
    'max_samples': list(np.linspace(0.1, 1, 1000)),
}

param_space['SVM'] = {
    'weights': ['uniform', 'distance'],
    'C': list(10 ** np.linspace(-1, 1, 1000))
}

param_space['KNN'] = {
    'k': list(range(3, 51)),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}


param_space['GAT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GATSep'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['GT'] = {
    'h_feats': [16, 32],
    'num_heads': [1, 2, 4, 8],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['PCGNN'] = {
    'h_feats': [16, 32, 64],
    'del_ratio': np.linspace(0.01, 0.8, 1000),
    'add_ratio': np.linspace(0.01, 0.8, 1000),
    'dist': ['euclidean', 'cosine'],
    # 'k': list(range(3, 10)),
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
}

param_space['NA'] = {
    'n_estimators': list(range(10, 201)),
    'eta': 0.5 * 10 ** np.linspace(-1, 0, 1000),
    'lambda': [0, 1, 10],
    'subsample': [0.5, 0.75, 1],
    'k': list(range(0, 51)),
}

param_space['PNA'] = {
    'h_feats': [16, 32, 64],
    'drop_rate': [0, 0.1, 0.2, 0.3],
    'num_layers': [1, 2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh'],
}


activations = {
    'ReLU': torch.nn.functional.relu,
    'LeakyReLU': torch.nn.functional.leaky_relu,
    'Tanh': torch.nn.functional.tanh}

param_space['DONE'] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}

param_space['AdONE'] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':[1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}

param_space['AnomalyDAE'] = {
    'embed_dim': [16, 32, 64],
    'out_dim': [4,8,16],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'weight_decay': [1e-5,1e-4,1e-3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}


param_space['CONAD'] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay': [1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}

param_space['DOMINANT'] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay': [1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}

param_space['GAAN'] = {
    'noise_dim': [8, 16, 32, 64],
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'generator_layers': [2, 3, 4],
    'encoder_layers': [2, 3, 4],
    'weight_decay': [1e-5,1e-4,1e-3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}

param_space['GCNAE'] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':[1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}


param_space['MLPAE'] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay': [1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
}


best_param = {}

def set_best_param(model, dataset,t):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    
    if not model in model_detector_dict.keys():
        for k, v in best_param[model].items():
            model_config[k] = v

    return model_config



best_param['GCNAE'] = {
    'hid_dim': 16,
    'dropout': 0.3,
    'num_layers': 2,
    'lr': 0.022,
    'weight_decay': 1e-5,
    'act': activations['ReLU'],
}


best_param['MLPAE'] = {
    'hid_dim': 32,
    'dropout': 0.3,
    'num_layers': 3,
    'lr': 0.0017,
    'weight_decay': 1e-5,
    'act': activations['LeakyReLU'],
}


best_param['GAAN'] = {
    'noise_dim': 32,
    'hid_dim': 64,
    'dropout': 0,
    'generator_layers': 3,
    'encoder_layers': 2,
    'weight_decay': 1e-5,
    'lr': 0.0847,
    'act': activations['Tanh'],
}


best_param['AnomalyDAE'] = {
    'embed_dim': 16,
    'out_dim': 8,
    'dropout': 0,
    'num_layers': 2,
    'weight_decay': 1e-3,
    'lr': 0.03660695147596903,
    'act': activations['ReLU'],
}


best_param['CONAD'] = {
    'hid_dim': 64,
    'dropout': 0.3,
    'num_layers': 4,
    'weight_decay': 1e-5,
    'lr': 5e-3,
    'act': activations['ReLU'],
}


best_param['DOMINANT'] = {
    'hid_dim': 64,
    'dropout': 0.3,
    'num_layers': 4,
    'lr': 5e-3,
    'weight_decay': 1e-5,
    'act': activations['ReLU']
    }

best_param['DONE'] = {
    'hid_dim': 32,
    'dropout': 0.2,
    'num_layers': 2,
    'lr': 0.03941,
    'weight_decay':1e-4,
    'act': activations['Tanh'],
}

best_param['AdONE'] = {
    'hid_dim': 32,
    'dropout': 0.2,
    'num_layers': 2,
    'lr': 0.03941,
    'weight_decay':1e-4,
    'act': activations['Tanh'],
}