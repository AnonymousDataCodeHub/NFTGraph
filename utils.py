import random
from models.detector import *
from dgl.data.utils import load_graphs
import os
from pygod.metric import eval_precision_at_k,eval_recall_at_k,eval_roc_auc,eval_average_precision
from pygod.detector import *
import torch.nn.functional
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
from pyod.models.cof import COF
from pyod.models.hbos import HBOS
from pyod.models.sod import SOD
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
# from pyod.models.deep_svdd import DeepSVDD
from pyod.models.loda import LODA
from pyod.models.iforest import IForest


class Dataset:
    def __init__(self, name='tfinance', prefix='datasets/'):
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

def toeval(labels, probs):
    score = {}
    with torch.no_grad():
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        if torch.is_tensor(probs):
            probs = probs.cpu().numpy()
        score['AUROC'] = roc_auc_score(labels, probs)
        score['AUPRC'] = average_precision_score(labels, probs)
        labels = np.array(labels)
        k = labels.sum()
    score['RecK'] = sum(labels[probs.argsort()[-k:]]) / sum(labels)
    return score
    
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
    # 'GCNAE': GCNAE,
    # 'MLPAE': MLPAE,
    'DOMINANT':DOMINANT,
    'GAAN':GAAN,
    'SCAN':SCAN,
    'Radar':Radar,
    'ANOMALOUS':ANOMALOUS,
    'ONE':ONE,
    'OCGNN':OCGNN,
    'CoLA':CoLA,
    'GUIDE':GUIDE,
}


# PYOD models
model_dict_od = {
    'PCA': PCA,
    'OCSVM': OCSVM,
    'LOF': LOF,
    'CBLOF': CBLOF,
    'COF':COF,
    'HBOS':HBOS,
    'SOD':SOD,
    'COPOD':COPOD,
    'ECOD':ECOD,
    'ONE':ONE,
    'LODA':LODA,
    'IForest':IForest,
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


for unsupervised_model in model_dict.keys():
    param_space[unsupervised_model] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':[1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
    'gamma':[0.01,0.1,0.5,1],
    'embed_dim': [16, 32, 64],
    'noise_dim': [8, 16, 32, 64],
    'hid_a': [16, 32, 64],
    'hid_s': [16, 32, 64],
    'alpha':[0.01,0.1,0.5,1],
    'theta':[0.1,1,10],
    'eta':[0.1,1,10],
    'weight':[0.01,0.1,0.5,1],
    'w1':[0.1,0.2,0.5,0.7,1],
    'w2':[0.1,0.2,0.5,0.7,1],
    'w3':[0.1,0.2,0.5,0.7,1],
    'w4':[0.1,0.2,0.5,0.7,1],
    'w5':[0.1,0.2,0.5,0.7,1],
    'graphlet_size':[4,10,50],
    'beta':[0.01,0.1,0.5,1],
    'eps':[0.1,0.5,0.7],
    'mu':[2,10,20],
    }
    
    

for unsupervised_od_model in model_dict_od.keys():
    param_space[unsupervised_od_model] = {
    'hid_dim': [16, 32, 64],
    'dropout': [0.0, 0.1, 0.2, 0.3],
    'num_layers': [2, 3, 4],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'weight_decay':[1e-5,1e-4,1e-3],
    'act': [activations['ReLU'], activations['LeakyReLU'], activations['Tanh']],
    'kernel':['linear', 'poly', 'rbf', 'sigmoid'],
    'nu':[0.2,0.5,0.7,1],
    'degree':[2,3],
    'gamma':'auto',
    'coef0':[0.0,0.5,1],
    'tol':[1e-3,1e-2],
    'algorithm' : ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'n_neighbors':[10,20,30],
    'leaf_size':[20,30,50],
    'p':[1,2,3],
    'n_clusters':[5,8,10,20],
    'alpha':[0.6,0.8,0.9],
    # 'beta':[2,5,10],
    'n_bins':[5,10,20],
    'alpha':[0.01,0.1,0.5,1],
    'tol':[1e-3,1e-2,0.5],
    'hid_a': [16, 32, 64],
    'hid_s': [16, 32, 64],
    'gamma':[0.01,0.1,0.5,1],
    'n_random_cuts':[10,100],
    'n_estimators':[20,50,100],
    'max_samples':['auto',50,100],    
    }

best_param = {}

def set_best_param(model, dataset,t):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if model in model_dict_od.keys(): # using default hyperparameters
        return model_config
    if dataset.endswith('_remove'):
        for k, v in best_param_remove[model].items():
            model_config[k] = v
    else:
        for k, v in best_param[model].items():
            model_config[k] = v        

    return model_config



best_param['MLP'] = {'model': 'MLP', 'lr': 0.0019692202554791715, 'drop_rate': 0.2, 'h_feats': 16, 'num_layers': 1, 'activation': 'LeakyReLU', 'in_feats': 8}
best_param['KNN'] = {'model': 'KNN', 'lr': 0.01, 'drop_rate': 0, 'k': 40, 'weights': 'uniform', 'p': 2, 'in_feats': 8}
best_param['SVM'] = {'model': 'SVM', 'lr': 0.01, 'drop_rate': 0, 'weights': 'distance', 'C': 2.6880010215376062, 'in_feats': 8}
best_param['RF'] = {'model': 'RF', 'lr': 0.01, 'drop_rate': 0, 'n_estimators': 129, 'criterion': 'entropy', 'max_samples': 0.7144144144144144, 'in_feats': 8}
best_param['GCN'] = {'model': 'GCN', 'lr': 0.004174655289253135, 'drop_rate': 0, 'h_feats': 32, 'num_layers': 1, 'activation': 'ReLU', 'in_feats': 8}
best_param['SGC'] = {'model': 'SGC', 'lr': 0.031586354082678174, 'drop_rate': 0.2, 'h_feats': 16, 'k': 2, 'mlp_layers': 1, 'in_feats': 8}
best_param['GIN'] = {'model': 'GIN', 'lr': 0.028018665564591955, 'drop_rate': 0.2, 'h_feats': 32, 'num_layers': 3, 'agg': 'max', 'activation': 'Tanh', 'in_feats': 8}
best_param['GraphSAGE'] = {'model': 'GraphSAGE', 'lr': 0.0036354699612933176, 'drop_rate': 0.2, 'h_feats': 32, 'num_layers': 1, 'agg': 'pool', 'activation': 'ReLU', 'in_feats': 8}
best_param['GAT'] = {'model': 'GAT', 'lr': 0.03115254223555488, 'drop_rate': 0.2, 'h_feats': 16, 'num_heads': 2, 'num_layers': 3, 'in_feats': 8}
best_param['GT'] = {'model': 'GT', 'lr': 0.08431909292866255, 'drop_rate': 0, 'h_feats': 16, 'num_heads': 2, 'num_layers': 2, 'in_feats': 8}
best_param['GAS'] = {'model': 'GAS', 'lr': 0.08994022174092044, 'drop_rate': 0, 'h_feats': 32, 'num_layers': 1, 'k': 14, 'dist': 'cosine', 'activation': 'Tanh', 'in_feats': 8, 'mlp_layers': 0}
best_param['BernNet'] = {'model': 'BernNet', 'lr': 0.037288213071828336, 'drop_rate': 0.1, 'h_feats': 16, 'mlp_layers': 2, 'orders': 5, 'activation': 'Tanh', 'in_feats': 8}
best_param['AMNet'] = {'model': 'AMNet', 'lr': 0.05620173848083188, 'drop_rate': 0, 'h_feats': 64, 'num_layers': 2, 'orders': 2, 'activation': 'LeakyReLU', 'in_feats': 8}
best_param['GHRN'] = {'model': 'GHRN', 'lr': 0.06604193962330306, 'drop_rate': 0, 'h_feats': 16, 'del_ratio': 0.05939826693920356, 'num_layers': 2, 'mlp_layers': 2, 'in_feats': 8}
best_param['GATSep'] = {'model': 'GATSep', 'lr': 0.04916903577628029, 'drop_rate': 0, 'h_feats': 16, 'num_heads': 2, 'num_layers': 1, 'in_feats': 8}
best_param['PCGNN'] = {'model': 'PCGNN', 'lr': 0.023625084654779464, 'drop_rate': 0, 'h_feats': 32, 'del_ratio': 0.09935935935935936, 'add_ratio': 0.06377377377377377, 'dist': 'cosine', 'num_layers': 2, 'in_feats': 8}


best_param_remove = {}
best_param_remove['MLP'] = {'model': 'MLP', 'lr': 0.0046630349297427315, 'drop_rate': 0.3, 'h_feats': 16, 'num_layers': 4, 'activation': 'ReLU', 'in_feats': 8}
best_param_remove['KNN'] = {'model': 'KNN', 'lr': 0.01, 'drop_rate': 0, 'k': 49, 'weights': 'distance', 'p': 2, 'in_feats': 8}
best_param_remove['SVM'] = {'model': 'SVM', 'lr': 0.01, 'drop_rate': 0, 'weights': 'distance', 'C': 1.596626022101426, 'in_feats': 8}
best_param_remove['RF'] = {'model': 'RF', 'lr': 0.01, 'drop_rate': 0, 'n_estimators': 158, 'criterion': 'entropy', 'max_samples': 0.13693693693693693, 'in_feats': 8}
best_param_remove['GCN'] = {'model': 'GCN', 'lr': 0.0019692202554791715, 'drop_rate': 0.2, 'h_feats': 16, 'num_layers': 1, 'activation': 'LeakyReLU', 'in_feats': 8}
best_param_remove['SGC'] = {'model': 'SGC', 'lr': 0.04051423171114647, 'drop_rate': 0.2, 'h_feats': 16, 'k': 1, 'mlp_layers': 1, 'in_feats': 8}
best_param_remove['GIN'] = {'model': 'GIN', 'lr': 0.005137013543351339, 'drop_rate': 0.1, 'h_feats': 16, 'num_layers': 1, 'agg': 'mean', 'activation': 'LeakyReLU', 'in_feats': 8}
best_param_remove['GraphSAGE'] = {'model': 'GraphSAGE', 'lr': 0.003503842245290676, 'drop_rate': 0, 'h_feats': 16, 'num_layers': 2, 'agg': 'mean', 'activation': 'Tanh', 'in_feats': 8}
best_param_remove['GAT'] = {'model': 'GAT', 'lr': 0.05031548945038054, 'drop_rate': 0.3, 'h_feats': 16, 'num_heads': 2, 'num_layers': 3, 'in_feats': 8}
best_param_remove['GT'] = {'model': 'GT', 'lr': 0.012738113231864785, 'drop_rate': 0.1, 'h_feats': 16, 'num_heads': 2, 'num_layers': 3, 'in_feats': 8}
best_param_remove['GAS'] = {'model': 'GAS', 'lr': 0.01783410220710008, 'drop_rate': 0, 'h_feats': 64, 'num_layers': 1, 'k': 32, 'dist': 'cosine', 'activation': 'LeakyReLU', 'in_feats': 8, 'mlp_layers': 0}
best_param_remove['BernNet'] = {'model': 'BernNet', 'lr': 0.014831025143361045, 'drop_rate': 0.1, 'h_feats': 64, 'mlp_layers': 2, 'orders': 2, 'activation': 'Tanh', 'in_feats': 8}
best_param_remove['AMNet'] = {'model': 'AMNet', 'lr': 0.09418331534647952, 'drop_rate': 0.1, 'h_feats': 32, 'num_layers': 1, 'orders': 2, 'activation': 'ReLU', 'in_feats': 8}
best_param_remove['GHRN'] = {'model': 'GHRN', 'lr': 0.07479522515621821, 'drop_rate': 0.2, 'h_feats': 16, 'del_ratio': 0.013936192742241428, 'num_layers': 3, 'mlp_layers': 2, 'in_feats': 8}
best_param_remove['GATSep'] = {'model': 'GATSep', 'lr': 0.04567301270168744, 'drop_rate': 0.2, 'h_feats': 16, 'num_heads': 2, 'num_layers': 3, 'in_feats': 8}
best_param_remove['PCGNN'] = {'model': 'PCGNN', 'lr': 0.0017711210643450886, 'drop_rate': 0.1, 'h_feats': 16, 'del_ratio': 0.04874874874874875, 'add_ratio': 0.15787787787787788, 'dist': 'cosine', 'num_layers': 2, 'in_feats': 8}

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
