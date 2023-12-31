import torch
from torch_geometric.data import Data
from pygod.metrics import eval_roc_auc
import numpy as np
import random
import os
import argparse
import time
from utils import *
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

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
parser.add_argument('--trials', type=int, default=5)
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--inductive', type=int, default=0)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)

args = parser.parse_args()

prefix = '/data/sx/NFTGraph'

datasets = []
if args.datasets is not None:
    datasets = args.datasets.split('-')
    print('Evaluated Datasets: ', datasets)


if args.models is not None:    
    if args.models == 'allsuper':
        models = ['MLP', 'KNN', 'SVM', 'RF', 'GCN', \
                'SGC', 'GIN', 'GraphSAGE', 'GAT', 'GT', 'PNA', 'BGNN', 'GAS', \
                'BernNet', 'AMNet', 'GHRN', 'GATSep', 'PCGNN',\
                'RFGraph']
    elif args.models == 'allunsuper':
        models = ['GCNAE','MLPAE','GAAN', 'DONE','AnomalyDAE','AdONE','CONAD','DOMINANT']
    else:
        models = args.models.split('-')
    print('Evaluated Baselines: ', models)


file_id = None
results = pd.DataFrame()
for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        time_cost = 0
        train_config = {
            'device': args.device,
            'epochs': args.epochs,
            'epoch': args.epochs,
            'patience': 50,
            'metric': 'AUROC',
            'inductive': bool(args.inductive)
        }
        if dataset_name.endswith('_remove'):
            dataset_ = dataset_name.split('_')[0]
            g = Dataset(dataset_,prefix=prefix+'/datasets/')
            graph_tmp = g.graph
            maxdegree = (graph_tmp.in_degrees() + graph_tmp.out_degrees()).argsort()[-1]
            print("maxdegree",maxdegree)
            graph_final = dgl.remove_nodes(graph_tmp,maxdegree.item())
        else:
            g = Dataset(dataset_name,prefix=prefix+'/datasets/')
            graph_final = g.graph            

        model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}

        auc_list, pre_list, rec_list = [], [], []
        for t in range(args.trials):
            torch.cuda.empty_cache()
            print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
            seed = seed_list[t]
            set_seed(seed)
            train_config['seed'] = seed
            g.split(args.semi_supervised, t)
            st = time.time()
            if model in model_detector_dict.keys():
                detector = model_detector_dict[model](train_config, model_config, g)
                print(detector.model)
                test_score = detector.train()
                auc_list.append(test_score['AUROC']), pre_list.append(test_score['AUPRC']), rec_list.append(test_score['RecK'])

            else: #unsupervised
                # To utilize the `pygod` package, graph of `pyg` format is employed.
                c = torch.stack([graph_final.edges()[0], graph_final.edges()[1]], dim=1).t().contiguous()
                data = Data(x=graph_final.ndata['feature'].to(torch.float32),edge_index=c,y=graph_final.ndata['label'])
                print(f"Data: {data}\n\n\n")
                
                if model == 'GCNAE':
                    batchsize = 0
                elif model == 'CONAD' or model == 'DOMINANT':
                    batchsize = 128
                else:
                    batchsize = 512
                train_config['gpu'] = args.device
                train_config['batch_size'] = batchsize
                model_config = set_best_param(model, dataset_name, t)

                detector = model_dict[model](**model_config,**train_config)
                detector.fit(data)
                outlier_scores = detector.decision_scores_
                y = list(data.y.numpy())
                outlier_scores = list(outlier_scores)
                auc_score = eval_roc_auc(y, outlier_scores)
                map = average_precision_score(y, outlier_scores)
                k = sum(y)
                reck = sum(np.array(y)[np.array(outlier_scores).argsort()[-k:]]) / sum(y)
                auc_list.append(auc_score),pre_list.append(map),rec_list.append(reck)

            ed = time.time()
            time_cost += ed - st
            del detector
        
        del data, g
        model_result[dataset_name+'-AUROC mean'] = np.mean(auc_list)
        model_result[dataset_name+'-AUROC std'] = np.std(auc_list)
        model_result[dataset_name+'-AUPRC mean'] = np.mean(pre_list)
        model_result[dataset_name+'-AUPRC std'] = np.std(pre_list)
        model_result[dataset_name+'-RecK mean'] = np.mean(rec_list)
        model_result[dataset_name+'-RecK std'] = np.std(rec_list)
        model_result[dataset_name+'-Time'] = time_cost/args.trials
    model_result = pd.DataFrame(model_result, index=[0])
    results = pd.concat([results, model_result])
    file_id = save_results(results, file_id)
    print(results)
