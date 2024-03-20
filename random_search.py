import argparse
import time
import pandas
from copy import deepcopy
from utils import *
import warnings
import torch
from torch_geometric.data import Data
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
parser.add_argument('--trials', type=int, default=100)  # Number of random search
parser.add_argument('--semi_supervised', type=int, default=0)
parser.add_argument('--models', type=str, default='allunsuper') # SemiGNN
parser.add_argument('--datasets', type=str, default='tinynftgraph')
parser.add_argument('--device', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100) # Number of training epochs
parser.add_argument('--pyod', type=int, default=0)

args = parser.parse_args()

prefix = '/data/sx/NFTGraph'

datasets = []
if args.datasets is not None:
    datasets = args.datasets.split('-')
    print('Evaluated Datasets: ', datasets)

columns = ['name']
for dataset_name in datasets:
    for metric in ['AUROC', 'AUPRC', 'RecK', 'Time']:
        columns.append(dataset_name+'-'+metric)


if args.models is not None:    
    if args.models == 'allsuper':
        models = ['MLP', 'KNN', 'SVM', 'RF', 'GCN', \
                'SGC', 'GIN', 'GraphSAGE', 'GAT', 'GT', 'GAS', \
                'BernNet', 'AMNet', 'GHRN', 'GATSep', 'PCGNN']
    elif args.models == 'allunsuper':
        models = ['ANOMALOUS','ONE', 'OCGNN', 'CoLA',\
            'DONE', 'AnomalyDAE', 'AdONE', 'CONAD' 'DOMINANT'] #'GAAN', 'SCAN', 'Radar', 'GUIDE'
    elif args.models == 'allunsuperod':
        models = ['OCSVM', 'LOF', 'CBLOF', 'COF', 'HBOS',\
            'SOD', 'COPOD', 'ECOD','LODA','IForest']
    else:
        models = args.models.split('-')
    print('Evaluated Baselines: ', models)


results = pd.DataFrame(columns=columns)
best_model_configs = {}
file_id = None

for model in models:
    model_result = {'name': model}
    best_model_configs[model] = {}
    for dataset_name in datasets:
        print('============Dataset {} Model {}=============='.format(dataset_name, model))
        auc_list, pre_list, rec_list = [], [], []
        set_seed()
        time_cost = 0
        best_val_score = 0
        best_troc, best_tprc, best_treck = 0,0,0
        train_config = {
            'device': args.device,
            'epochs': args.epochs,  #traing epoch for supervised models
            'epoch': args.epochs, #training epoch for unsupervised models
            'patience': 20,
            'metric': 'AUROC',
            'inductive': False
        }

        if dataset_name.endswith('_remove'):    # removing the node with the highest degree
            dataset_ = dataset_name.split('_')[0]
            g = Dataset(dataset_,prefix=prefix+'/datasets/')
            graph_tmp = g.graph
            maxdegree = (graph_tmp.in_degrees() + graph_tmp.out_degrees()).argsort()[-1]
            # print("maxdegree",maxdegree)
            graph_final = dgl.remove_nodes(graph_tmp,maxdegree.item())
            g.graph = graph_final
        else:
            g = Dataset(dataset_name,prefix=prefix+'/datasets/')
            graph_final = g.graph            

        g.split(args.semi_supervised, 0)
        
        
        # To utilize the `pygod` package, graph of `pyg` format is employed.
        c = torch.stack([graph_final.edges()[0], graph_final.edges()[1]], dim=1).t().contiguous()
        c = c.long()
        data = Data(x=graph_final.ndata['feature'].to(torch.float32),edge_index=c,y=graph_final.ndata['label'],train_mask=graph_final.ndata['train_mask'],val_mask=graph_final.ndata['val_mask'],test_mask=graph_final.ndata['test_mask'])
        data.x = (data.x - torch.mean(data.x,dim=0))/ torch.std(data.x,dim=0)
        print(f"\nData: {data}\n\n")
        
        tmp = graph_final.ndata['feature']
        tmp = (tmp - torch.mean(tmp,dim=0))/ torch.std(tmp,dim=0)
        graph_final.ndata['feature'] = tmp        
        
        for t in range(args.trials):
            print("Dataset {}, Model {}, Trial {}, Time Cost {:.2f}".format(dataset_name, model, t, time_cost))
            if time_cost > 7200:  # 86400 Stop after 1 day
                break
            train_config['seed'] = seed_list[t]
            # g.split(args.semi_supervised, t)
            st = time.time()
            if model in model_detector_dict.keys():

                model_config = sample_param(model, dataset_name, t)
                detector = model_detector_dict[model](train_config, model_config, g)
                print("model_config: ", model_config)
                test_score = detector.train()
                if detector.best_score > best_val_score:
                    print("****current best score****")
                    best_val_score = detector.best_score
                    best_model_config = deepcopy(model_config)
                    best_troc, best_tprc, best_treck = test_score['AUROC'], test_score['AUPRC'], test_score['RecK']
                model_config['withoutDummyNode'] = args.withoutDummyNode
                model_config['withoutLinkPred'] = args.withoutLinkPred

                if args.pyod:   
                    detector = model_dict_od[model]()
                    detector.fit(data.x[data.train_mask])
                    outlier_scores = detector.decision_function(data.x.numpy())
                    outlier_scores = torch.from_numpy(outlier_scores)
                else:
                    detector = model_dict[model](**model_config,**train_config)
                    detector.fit(data)
                    outlier_scores = detector.decision_score_
                labels = data.y
                train_labels, val_labels, test_labels = labels[data.train_mask], labels[data.val_mask], labels[data.test_mask]
                val_score = toeval(val_labels, outlier_scores[data.val_mask])
                if val_score[train_config['metric']] > best_val_score:
                    print("****current best score****")
                    best_val_score = val_score[train_config['metric']]
                    best_model_config = deepcopy(model_config)
                    test_score = toeval(test_labels, outlier_scores[data.test_mask])
                    best_troc, best_tprc, best_treck = test_score['AUROC'], test_score['AUPRC'], test_score['RecK']
            elif model == 'SNGNN':
                graph_final = graph_final.long()
                g.graph = graph_final
                if not model_config['withoutDummyNode']:
                    feat = g.graph.ndata['feature'][g.graph.ndata['train_mask']==1]
                    meanfeat = feat[g.graph.ndata['label'][g.graph.ndata['train_mask']==1]==1].mean(axis=0).to(torch.float32).unsqueeze(0)
                    zeromasks = torch.tensor([0]*20,dtype=torch.uint8).unsqueeze(0)
                    onemasks = torch.tensor([1]*20,dtype=torch.uint8).unsqueeze(0)
                    g.graph = dgl.add_nodes(g.graph,1,{'feature':meanfeat,'test_masks':zeromasks,'val_masks':zeromasks,'train_masks':onemasks, \
                        'train_mask':torch.tensor([1],dtype=torch.uint8),'val_mask':torch.tensor([0],dtype=torch.uint8),'test_mask':torch.tensor([0],dtype=torch.uint8), \
                        'label':torch.tensor([1],dtype=torch.int64)})

                    indices  = torch.nonzero(g.graph.ndata['label'][g.graph.ndata['train_mask']==1] == 1).squeeze()
                    g.graph.add_edges(torch.tensor([g.graph.num_nodes()-1]*len(indices)),indices)        
                    g.graph = g.graph.remove_self_loop().add_self_loop()        
                detector = SNGNNDetector(train_config, model_config, g)
                test_score = detector.train()
                print(test_score)
                if detector.best_score > best_val_score:
                    print("****current best score****")
                    best_val_score = detector.best_score
                    best_model_config = deepcopy(model_config)
                    best_troc, best_tprc, best_treck = test_score['AUROC'], test_score['AUPRC'], test_score['RecK']
            else: #unsupervised
                if model == 'GCNAE':
                    batchsize = 0
                elif model == 'CONAD' or model == 'DOMINANT':
                    batchsize = 128
                else:
                    batchsize = 512
                train_config['gpu'] = args.device
                train_config['batch_size'] = batchsize
                model_config = sample_param(model, dataset_name, t)   
                model_config['verbose'] = 1      
                
                if args.pyod:   
                    detector = model_dict_od[model]()
                    detector.fit(data.x[data.train_mask])
                    outlier_scores = detector.decision_function(data.x.numpy())
                    outlier_scores = torch.from_numpy(outlier_scores)
                    # outlier_scores = detector.decision_function(data.x,return_pred=False,return_score=True)
                else:
                    detector = model_dict[model](**model_config,**train_config)
                    detector.fit(data)
                    outlier_scores = detector.decision_score_
                    # outlier_scores = detector.decision_score_predict(data,return_pred=False,return_score=True)

                labels = data.y
                train_labels, val_labels, test_labels = labels[data.train_mask], labels[data.val_mask], labels[data.test_mask]
                val_score = toeval(val_labels, outlier_scores[data.val_mask])
                if val_score[train_config['metric']] > best_val_score:
                    print("****current best score****")
                    best_val_score = val_score[train_config['metric']]
                    best_model_config = deepcopy(model_config)
                    test_score = toeval(test_labels, outlier_scores[data.test_mask])
                    best_troc, best_tprc, best_treck = test_score['AUROC'], test_score['AUPRC'], test_score['RecK']

            ed = time.time()
            time_cost += ed - st
            print("Current Val Best:{:.4f}; Test AUC:{:.4f}, PRC:{:.4f}, RECK:{:.4f}".format(
                best_val_score, best_troc, best_tprc, best_treck))

        print("best_model_config:", best_model_config)
        best_model_configs[model][dataset_name] = deepcopy(best_model_config)
        model_result[dataset_name+'-AUROC'] = best_troc
        model_result[dataset_name+'-AUPRC'] = best_tprc
        model_result[dataset_name+'-RecK'] = best_treck
        model_result[dataset_name+'-Time'] = time_cost/(t+1)
    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id)
