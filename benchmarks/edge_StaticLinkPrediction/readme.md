### Static Link Prediction
#### a. Reproduce

First, change the prefix of your own location of ***NFTGraph*** to the appropriate path: \
`prefix = '/data/.../NFTGraph'`.

Benchmark GCN on the `nftgraph` datasets (10 trials).
```
python run_link_prediction.py --dataset ogbl-nftgraph --model gcn --trials 10 
```


#### b. Random Search

Perform a random search of hyperparameters for the GCN model on the `nftgraph` dataset (100 trials).
```
python random_search_linkpred.py --trial 100 --dataset ogbl-nftgraph --model gcn
```

