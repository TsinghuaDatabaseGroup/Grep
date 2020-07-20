Automatic Database Partition using Graph Embedding
===========================
PyTorch implementation of Automatic Database Partition using Graph Embedding [1].

Description
===========================
We propose a partition key selection framework using graph embedding algorithms. 
This framework mainly includes three parts:

1. **Build graph model:**
Given a set of concurrent queries, we denote operators in their physical plans as vertices V, and denote correlations between operators as edges E, which include data-passing/data-sharing/data-conflict/resource competition. The workload graph is denoted as (V, E) 
2. **Predict performance with graph embedding:**
(**training**)
We first choose a graph embedding model (e.g., simple GCN). And then, with workload graphs as training samples, for each graph, we iteratively use the graph model to predict for 60%  vertices, whose results are used to train the model, and use the left 40% vertices to validate the accuracy.
(**inference**)
Given a workload graph (V, E), we input (V, E) into the trained model and the model gives a performance vector for each vertices in V, like start time, execution time.

3. **Graph Compaction**
For large graph with thousands of  vertices,  we propose a greedy graph compaction algorithm. The basic idea is to greedily merge the vertices into a compound vertex.

Implementation
===========================
## Requirements
  * PyTorch (>=0.4)
  * Python (>=3.6)
  * Matplotlib (>=2.0.0)
  * scikit-learn (>=0.18)

## Usage
1. **Create Data**
```python graph_generate.py --input_path='./data/query_plan/job_pg' --output_path='./data/graph/job_pg```

 %% each column schema corresponds to a graph


2. **Load Data**
```python load_data.py --batch_size=100```

3. **Train Model**
```python train.py --model_name='simple_gcn' --epochs=500 --lr=0.01```

### Python Call Graph

![workflow](/Users/xuanhe/Documents/our-paper/partition/code/Grep/figs/workflow.png)


## Cite

Please cite our paper if you use this code in your own work:

```
@article{TBD}
```