# NFTGraph
This is the officaial implementation of the paper:

> Understanding the Influence of Extreme High-degree Nodes on Graph Anomaly Detection. (Under Review)


General document tree is as follows:
```
NFTGraph/
├── README.md
├── raw_data
├── datasets
├── preprocess.ipynb
├── statistic.py
├── models
├── anomaly_detection.py
├── random_search.py
├── utils.py
└── benchmarks
```

### !!! Due to the large files of `raw_data` and `datasets`, please download them from [google drive](https://drive.google.com/drive/folders/12YG7k3PM1mGP76g5qbm4-UrXWogpD8Gm?usp=drive_link) and replace the directories to make sure the doc trees are:
```
/raw_data/
├── crawler
│   ├── crawler_selenium.py
│   ├── readme.md
│   └── token_category.csv
├── create_edges.ipynb
├── create_nodes.ipynb
├── create_tinygraph.ipynb
├── edges.csv
├── nodes.csv
├── readme.md
├── suspicious_label
│   ├── ground_truth_anomaly_nodes.txt
│   ├── nftgraph_suspicious_labels.txt
│   └── nftgraph_tiny_suspicious_labels.txt
├── tinyedges.csv
├── tinynodes.csv
└── transactions.csv
```
```
/datasets/
├── dgl_graph
│   ├── nftgraph.bin
│   └── tinynftgraph.bin
├── ogb_graph
│   ├── make_master_file.py
│   ├── submission_ogbl_nftgraph
│   │   ├── meta_dict.pt
│   │   └── nftgraph.zip
│   ├── submission_ogbl_tinynftgraph
│   │   ├── meta_dict.pt
│   │   └── tinynftgraph.zip
│   ├── submission_ogbn_nftgraph
│   │   ├── meta_dict.pt
│   │   └── nftgraph.zip
│   └── submission_ogbn_tinynftgraph
│       ├── meta_dict.pt
│       └── tinynftgraph.zip
├── process.ipynb
├── process-tiny.ipynb
├── pyg_graph
│   ├── nftgraph.bin
│   └── tinynftgraph.bin
└── tgb_graph
    ├── tgbl_nftgraph
    │   ├── generate_tgbl-nftgraph.py
    │   ├── ml_tgbl-nftgraph_edge.pkl
    │   ├── ml_tgbl-nftgraph.pkl
    │   └── tgbl-nftgraph_edgelist.csv
    └── tgbl_tinynftgraph
        ├── generate_tgbl-tinynftgraph.py
        ├── ml_tgbl-tinynftgraph_edge.pkl
        ├── ml_tgbl-tinynftgraph.pkl
        ├── tgbl-tinynftgraph_edgelist.csv
        ├── tgbl-tinynftgraph_test_ns.pkl
        └── tgbl-tinynftgraph_val_ns.pkl
```
### !!! Please start by running `preprocess.ipynb` before running `anomaly_detection.py` or `random_search.py`.

## Pre: Background Knowledge of blockchain and NFT
At the end.


## I. Dataset
A new graph dataset, ERC1155-Graph (short for NFTGraph) is proposed in this paper, and mainstream graph learning libraries such as `ogb`, `pyg`, and `dgl` are provided. We introduce them in the following.

### A. raw_data
This folder encompasses all the raw data for both NFTGraph and NFTGraph-Tiny.
```
raw_data/
├── create_edges.ipynb
├── create_nodes.ipynb
├── create_tinygraph.ipynb
├── edges.csv
├── nodes.csv
├── readme.md
├── suspicious_label
│   ├── ground_truth_anomaly_nodes.txt
│   ├── nftgraph_suspicious_labels.txt
│   └── nftgraph_tiny_suspicious_labels.txt
├── tinyedges.csv
├── tinynodes.csv
└── transactions.csv
```

<!-- The `crawel` folder contains the code for downloading all transcations from the TokenView website. -->

----
The `suspicious_label` folder includes the labels for both NFTGraph and NFTGraph-Tiny. Specically, `ground_truth_anomaly_nodes.txt` file contains the account addresses of **ground-truth fraudulent nodes** published by previous researchers encompassing Ponzi schemes and phishing scams. Data sources of them are in the following:\
[https://www.kaggle.com/datasets/arkantaha/ethereum-phishing-scams-dataset-ethpsd](https://www.kaggle.com/datasets/arkantaha/ethereum-phishing-scams-dataset-ethpsd).\
[https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset](https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset).\
[https://ibase.site/scamedb/](https://ibase.site/scamedb/).\
[https://xblock.pro/#/dataset/13](https://xblock.pro/#/dataset/13).\
[https://xblock.pro/#/article/95](https://xblock.pro/#/article/95).\
[https://xblock.pro/#/article/98](https://xblock.pro/#/article/98).\
We collect an approximate total of 6,000 ground-truth fraudulent account addresses, and label nodes that exhibit interactions with the fraudulent nodes exceeding a count of three instances as **suspicious nodes**. These labels of account adresses can be found in `nftgraph_suspicious_labels.txt` and `nftgraph_tiny_suspicious_labels.txt`, respectively.

---

The `transactions.csv` file undergoes processing based on the initial outcomes derived from web crawling. Given that a single transaction hash, denoted as `txhash`, may encompass multiple internal operations such as transfers, trades, mints, and burns, we ascertain the count of actions within a transaction, denoted as `times`. Subsequently, we evenly distribute the values of `value` and `transactionFee` among each action, i.e., `value` ÷ `times` and `transactionFee` ÷ `times`. The computation of the `transferedAmount` involves applying the function `int(log(t))+1` (zero remains invariant as zero) and is then normalized to a maximum value of 100, considering that certain `transferedAmount` values can be very large.

Download link of the `transactions.csv` file: [google drive](https://drive.google.com/file/d/1wZmG6famm-sx5hutub_Y0EN5Yaa64S8O/view?usp=sharing).

Preview of the `transactions.csv` file:
| txn_id | source | target | tokenid | timestamp | transferedAmount | value | transactionFee | from | to | token | txhash | edgelabel |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|
|0 | 674618 | 501095 | 1170882 | 20220730055230 | 1 | 78.52 | 2.23 | 0x9463ea1dadf279e174e1075b49b8b7a13d1e7293 | 0x6e388502b891ca05eb52525338172f261c31b7d3 | 0xd07dc4262bcdbf85190c01c996b4c06a461d2430 | 0xb55b5b44aa556916ab6c8b38c40649c06c6363be5f0034cac678fd44e5f9b420 | 11 |
|1 | 0 | 984132 | 1170882 | 20220730055230 | 14 | 0.0 | 0.98 | 0x0000000000000000000000000000000000000000 | 0xd8b75eb7bd778ac0b3f5ffad69bcc2e25bccac95 | 0xd07dc4262bcdbf85190c01c996b4c06a461d2430 | 0xa50ddcc6c3738761284a9e01427117781dd4810acc9140a3f6f6df6c6e00aeea | 12 |

Each row represents an action of a certain `transferedAmount` of `token` worth `value` between the `from` and `to` addresses at a specific `timestamp`, spending a transaction fee (`transactionFee`), as recorded in the transaction (`txhash`).

There are four values of `edgelabel`, where `10` represents the **Transfer** edge of *User-to-User*, `11` refers to the **Trade** edge of *User-to-User*, `12` is the **Mint** edge of *Null Address-to-User*, `13` means the **Burn** edge of *User-to-Null Address*.

---
The `nodes.csv` and `edges.csv` files are the node and edge data for the NFTGraph. These files are generated from the `transactions.csv` file using the `create_nodes.ipynb` and `create_edges.ipynb`. The `tinynodes.csv` and `tinyedges.csv` and `create_tinygraph.ipynb` are for NFTGraph-Tiny, creating by the `create_tinygraph.ipynb`.

Preview of the `nodes.csv` file:
| addr | OutCnt | OutAmount | OutValue | OutTransFee | InCnt | InAmount | InValue | InTransFee | label |
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|
0x9463ea1dadf279e174e1075b49b8b7a13d1e7293 | 30 | 30.0 | 7403.125 | 311.005 | 27 | 27.0 | 3326.335 | 701.33 | 0 |
0x0000000000000000000000000000000000000000 | 2650186 | 7217724.0 | 712324959.5924134 | 21609773.15612327 | 595830 | 2410795.0 | 20296148.73451125 | 4879685.167745083 | 0 |

The `label` attribute serves as a binary indicator, where a value of 0 denotes a *Normal* node, and a value of 1 signifies a *Suspicious* node.


Preview of the `edges.csv` file:

| from | to | timestamp | transferedAmount | value | transactionFee | TxnsCnt |
|:-|:-|:-|:-|:-|:-|:-|
| 0x9463ea1dadf279e174e1075b49b8b7a13d1e7293 | 0x6e388502b891ca05eb52525338172f261c31b7d3 | 20220730055230 | 1 | 78.52 | 2.23 | 1 |
0x0000000000000000000000000000000000000000 | 0xd8b75eb7bd778ac0b3f5ffad69bcc2e25bccac95 | 20220730055230 | 18 | 0.0 | 6.369999999999999 | 2 |

In the `transaction.csv` file, when there are multiple directed edges between any two nodes, they are merged into a single edge. The `TxnsCnt` column represents the count of these merged edges. The `timestamp` is set to the latest time among the merged edges, and other features are summed up to represent the merged edge's features.

---

### B. datasets
The `datasets` folder contains the codes and final outputs for mainstream graph learning libraries.

#### Usage
##### a. Reproduce 
Open and run `process.ipynb`, but change the prefix of your own location of ***NFTGraph*** to the appropriate path. 
```python
prefix = '/data/.../NFTGraph'
```
b. Directly load\
**dgl_graph (DGL)**:

```python
from dgl.data.utils import load_graphs

dataset_name = 'nftgraph'
graph = load_graphs(prefix + '/datasets/dgl_graph/' + dataset_name+'.bin')[0][0]
```

**pyg_graph (PyTorch Geometric)**: 

```python
import torch
from torch_geometric.data import Data

dataset_name = 'nftgraph'
data = torch.load(prefix+'/datasets/pyg_graph/' + dataset_name'.bin')
```

**ogb_graph (OGB)**:

An example for ogbl-nftgraph:

```python
import subprocess
import torch
from ogb.linkproppred import DglLinkPropPredDataset

prefix = '/data/.../NFTGraph'
dataset_name = 'nftgraph'

#unzip the file
filedir = prefix + f'/datasets/ogb_graph/submission_ogbl_{dataset_name}/{dataset_name}.zip'
dstdirs = prefix + f'/datasets/ogb_graph/submission_ogbl_{dataset_name}/{dataset_name}'

command = f'unzip {filedir} -d {dstdirs}'
process = subprocess.Popen(command,shell=True)

meta_dict = torch.load(prefix+f'/datasets/ogb_graph/submission_ogbl_{dataset_name}/meta_dict.pt')
meta_dict['dir_path'] = prefix+f'/datasets/ogb_graph/submission_ogbl_{dataset_name}/{dataset_name}'

dataset = DglLinkPropPredDataset(name = dataset_name, root = meta_dict['dir_path'] ,meta_dict=meta_dict)

graph = dataset[0]
split_edge = dataset.get_edge_split()
```

For a deeper understanding of the [Generation](https://docs.google.com/document/d/e/2PACX-1vS1hBTYLONRwAU9UxK42USTuRKrt_Yk4H0EhpLvJC_eOrGxbJUtrzDqlIStAFpnwZt2N28B3MuSxgqj/pub) and [Usage](https://ogb.stanford.edu/docs/home/) of OGB format datasets, we highly recommend exploring the official team documents and websites via the links.

**tgb_graph (TGB)**:

TGB is a mainstream library for **temporal** graphs similar to OGB. Refer to [TGB](https://tgb.complexdatalab.com/) for details.\
Due to github's limit on the size of uploaded files, please download from [google drive](https://drive.google.com/drive/folders/1bztR3VYDx6dOewN6c_57Dj60CjQD5-cF?usp=sharing) and ensure that the `tgb_graph` file tree is as follows:
```
tgb_graph/
├── tgbl_nftgraph
│   ├── generate_tgbl-nftgraph.py
│   ├── ml_tgbl-nftgraph_edge.pkl
│   ├── ml_tgbl-nftgraph.pkl
│   └── tgbl-nftgraph_edgelist.csv
└── tgbl_tinynftgraph
    ├── generate_tgbl-tinynftgraph.py
    ├── ml_tgbl-tinynftgraph_edge.pkl
    ├── ml_tgbl-tinynftgraph.pkl
    ├── tgbl-tinynftgraph_edgelist.csv
    ├── tgbl-tinynftgraph_test_ns.pkl
    └── tgbl-tinynftgraph_val_ns.pkl
```

i) Reproduce: Open and run `generate_tgbl-nftgraph.py`, but change the prefix of your own location of ***NFTGraph*** to the appropriate path. \
ii) Directly load:
```python
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset

dataset_name = 'tgbl-nftgraph'
prefix = '/data/.../NFTGraph'
dataset = PyGLinkPropPredDataset(name=dataset_name,root=prefix+'/datasets/tgb_graph/')
data = dataset.get_TemporalData()
```

---
### statistic.py
This code utilizes the `networkx` package to calculate various metrics on theg graph datasets.

## II. Benchmarking
### A. Anomaly Detection

#### a. Reproduce

First, change the prefix of your own location of ***NFTGraph*** to the appropriate path: \
`prefix = '/data/.../NFTGraph'`.

Benchmark GCNAE on the `nftgraph` and `tinynftgraph` datasets for training 100 epochs (10 trials).
```
python anomaly_detection.py --datasets nftgraph-tinynftgraph --models GCNAE -- trial 10 --epochs 100
```

Benchmark GCN and GraphSAGE models on the `nftgraph` and `tinynftgraph` datasets for training 100 epochs in the semi-supervised setting(10 trials).
```
python anomaly_detection.py --datasets nftgraph-tinynftgraph --models GCN-GraphSAGE -- trial 10 --epochs 100 --semi_supervised 1
```

Benchmark all supervised models on the `tinynftgraph` dataset for training 20 epochs (10 trials).
```
python anomaly_detection.py --datasets tinynftgraph --models allsuper -- trial 10 --epochs 20
```

Benchmark all unsupervised models on the `nftgraph` dataset for training 20 epochs (10 trials).
```
python anomaly_detection.py --datasets nftgraph --models allunsuper -- trial 10 --epochs 20
```

#### b. Random Search

Perform a random search of hyperparameters for the GCN model on the `nftgraph` dataset in the fully-supervised setting (100 trials).
```
python random_search.py --trial 100 --datasets nftgraph --models GCN
```

Our code is based on the code of **GADBench**, please refer to the [github](https://github.com/squareRoot3/GADBench/tree/master) and [paper](https://arxiv.org/abs/2306.12251) for some details such as semi-supervised setting. 

Hyperparameter search space:
![](/images/hyperparameter_search_space_for_all_supervised_models.png)
![](/images/hyperparameter_search_space_for_all_unsupervised_models.png)

Best hyperparameter:
![](/images/BestHyper_supervised.png)
![](/images/BestHyper_unsupervised.png)

### B. Link Prediction

#### a. edge_StaticLinkPrediction

Execute the command `python run_link_prediction.py` after entering each model folder. For example,

```python
cd /benchmarks/edge_StaticLinkPrediction/
python run_link_prediction.py --dataset ogbl-nftgraph --model gcn --trials 10 
```

Enter `/NFTGraph/benchmarks/edge_StaticLinkPrediction` for more details of benchmarks.

#### b. edge_TemporalLinkPrediction

Dynamic graph neural networks for temporal link prediction is accomplished by `TGB` package in a Python language, which can be found on https://github.com/fpour/TGB_Baselines/tree/main/models and https://github.com/shenyangHuang/TGB/tree/main.

Enter `/NFTGraph/benchmarks/edge_TemporalLinkPrediction` for more details of results of benchmarks.


## Background Knowledge of blockchain and NFT

### Blockchain
The blockchain is essentially a ledger, where each transaction is documented. Each transaction represents an action of a certain `transferedAmount` of `token` worth `value` between the `from` and `to` addresses at a specific `timestamp`, spending a transaction fee (`transactionFee`).

### Nodes and Edges on the Ethereum Blockchain
The Ethereum blockchain has two types of accounts: Externally Owned Accounts (EOAs) and Contract Accounts (CAs). EOAs are operated by external users who interact with smart contracts or send transactions to others. On the other hand, CAs are controlled by smart contracts, which execute code when triggered by a transaction from an EOA or another CA. Specially, **0x00...0000** or **0x00...dead** is a generic null address, which is not owned by any user.

Therefore, there are four types of edges: `Transfer` usually shows that the **From** address is the same as the **To** address, and/or the transaction value is non-zero, while `Trade` indicates a transaction where the **From** address is different from the **To** address, and the value is non-zero because of the buying or selling token. There are two types of transactions related to the null address: `Mint` and `Burn`. `Mint` indicates the source node is the null address, meaning a token is minted, while `Burn` presents that the null address is as a target node when a token is burnt.

We can label the edges according to their edge type. In `transaction.csv`, there are four values of `edgelabel`, where `10` represents the **Transfer** edge of *User-to-User*, `11` refers to the **Trade** edge of *User-to-User*, `12` is the **Mint** edge of *Null Address-to-User*, `13` means the **Burn** edge of *User-to-Null Address*. These labels can be used for edge classification.

### NFT
NFTGraph distinguishes itself from prior blockchain-based graph datasets, such as `Bitcoin` and `Ethereum`. As illustrated in Figure 1 (a), assets transferred in previous blockchain networks are identical, such as bitcoin, whereas in NFT transaction networks, assets are unique since each asset is distinct. NFT's inherent property can aid in labeling during graph construction, since transactions with the same NFT collections can be grouped into a community/subgraph and the category of NFT collections can serve as unique community/subgraph labels including art, gaming, and music. The following table presents the statistical outcomes for each category, where PFPs denote the images employed as social media avatars or profile pictures. (In `raw_data/crawler/token_category.csv`.) These labels can be used for graph classification.

![nft_collection.png](/images/nft_collection.png)\
Fig.1 Description on NFT trading networks and collections. (a) NFTs are unique digital assets that can represent anything from art to virtual real estate. NFT trading networks involve the exchange of various digital assets, in contrast to bitcoin networks where all transferred assets are the same (e.g., bitcoin). (b) NFT collections can serve as subgraph labels since NFTs within the same collection share an identical token address, and the concept of NFT collections is similar to that of painting collections in the real world.

| Category | Count |
|:-|:-|
Art | 2872
PFPs | 625
Gaming | 101
Memberships | 93
Photography | 81
Music | 17
Virtual-worlds | 9
Sports-collectibles | 3
Others | 7226
