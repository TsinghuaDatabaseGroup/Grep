Automatic Database Partition using Graph Embedding
===========================
PyTorch implementation of Automatic Database Partition using Graph Embedding [1].

Description
===========================
We propose a partition key selection framework using graph embedding algorithms. 
This framework mainly includes three parts:

1. **Build column graph:**
We characterize behaviors of queries on the columns in the form of a graph model, where the vertices denote the features of columns and the edges capture query costs among different columns (e.g., equi-joins). We generate column graphs with different workloads and store them as training data.
2. **Select partitition keys:**
We adopt a graph-based learning model to embed graph features for every column and select columns based on the embedded subgraphs.
(**training**)
We first choose a combination of graph embedding (e.g., simple GCN) and relevance decomposition as the key selection model. And then, with column graphs as training samples, for each graph, we iteratively use the graph model to select partition keys and utilize the performance to tune model parameters.
(**inference**)
Given a column graph (V, E), we input (V, E) into the trained model and the model gives a column vector, where 1 represents the corresponding column is selected and 0 is not.

3. **Evaluate partition performance:**
To reduce the partition overhead, we pre-train a graph-representaion-based evaluation model to estimate the workload performance for each partition strategy.

Implementation
===========================
## Requirements
  * PyTorch (>=0.4)
  * Python (>=3.6)
  * Matplotlib (>=2.0.0)
  * scikit-learn (>=0.18)

## Usage
1. **Create Data**
```python graph_generate.py --db='tpch'```

 %% each workload corresponds to a graph

2. **Train Model**
```python main.py --db='tpch'```

### Python Call Graph

![workflow](https://github.com/DBLearner-stack/Grep/tree/master/figs/workflow.png?raw=true)
