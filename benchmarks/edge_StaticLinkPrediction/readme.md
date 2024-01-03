## Benchmark: Static Link Prediction
We design a static link prediction task and select five advanced GNN models, 
namely SEAL, SEAL+NGNN, SUREL, SIEG, and AGDN, as well as three basic GNN models: GraphSAGE, GAT, and GCN. 
We use two metrics, namely Mean Reciprocal Rank (MRR) and AUC, to evaluate their performance. 
MRR is the mean of reciprocal ranks measuring the reciprocal ranks over a set of listed results, 
and AUC is the area under the ROC curve. 

### Setup
For the static link prediction task, we employ all edges encompassed within the NFTGraph. These edges correspond to real user transactions, where already existing connections constitute positive edges (= 1). 
Additionally, an equivalent number of edges between nodes are randomly sampled to form negative edges (= 0), as these transactions are non-existent. 
The data division entails a random allocation into the train, validation, and test sets, distributed as 80\%, 10\%, and 10\%, respectively. The numbers of positive and negative edges within these sets are equated. 

### Models and Metrics

We select five advanced GNN models, namely SEAL, SEAL+NGNN, SUREL, SIEG, and AGDN, as well as three basic GNN models: GraphSAGE, GAT, and GCN. The assessment of algorithm performance hinges on their capability to distinguish between positive and negative edges within the test set. Each model undergoes five individual trials, with the evaluation metrics being the Mean Reciprocal Rank (MRR) and the Area Under the ROC Curve (AUC). The average and standard deviation of these metrics across the five runs are computed as the final result. 

### Codes
Thanks to `dgl` package, the reference codes are listed as follows:

SEAL: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/seal_ogbl. \
SEAL+NGNN: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ngnn_seal. \
SUREL:https://github.com/Graph-COM/SUREL. \
SIEG: https://github.com/anonymous20221001/SIEG_OGB. \
AGDN: https://github.com/skepsun/Adaptive-Graph-Diffusion-Networks. \
GraphSAGE: https://github.com/dmlc/dgl/tree/master/examples/pytorch/graphsage. \
GAT: https://github.com/dmlc/dgl/tree/master/examples/pytorch/ogb/ogbn-products/gat. \
GCN: https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn.

### Results
![](/images/static_link_prediction_results.png)