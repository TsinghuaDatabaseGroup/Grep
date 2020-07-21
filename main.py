# from evaluate import evaluate_model
from train import test, train
from model.simple_gcn import GCN
from model.evaluate_model import eva1
from load_data.load_training_sample import load_data
from graph_generate import generate_graph

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim 

from configs.base import feature_num, max_iteration, max_test_iteration, workload_num, db
from pathlib import Path
base_dir = Path()

class arguments():
    def __init__(self):
        self.cuda = False
        self.fastmode = False
        self.seed = 42
        self.epochs = 500
        self.lr = 0.01
        self.weight_decay = 5e-4
        self.hidden = 16
        self.dropout = 0.5
        self.edge_dim = 30
        self.node_dim = 30
args = arguments()

# load training data and train the key selection model
def main():
    """ Partition databases with graph embedding"""

    data_path = Path().joinpath('pmodel_data/serial/graph/')
    '''
        # data.generate
        for wid in range(workload_num[db]):
            generate_graph(wid)
    '''

    evaluator = eva1(pretrained=True) # evaluate.load_model


    # model.initiate
    # model: gcn + relevance decomposition
    model = GCN(nfeat=feature_num,
                nhid=args.hidden,
                nclass=args.node_dim,
                dropout=args.dropout)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    # model training
    for wid in range(max_iteration):
        # data.load
        use_evaluate = 0
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path = data_path, dataset = "sample-plan-" + str(wid))
        if -1 in labels: # no available performance information
            use_evaluate = 1

        # model.train
        t_total = time.time()
        for epoch in range(args.epochs):
            train(epoch, labels, idx_train, idx_val, idx_test, model, optimizer, features, adj, evaluator, use_evaluate)
        print("Optimization Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        # model.valid
        test(labels, idx_train, idx_val, idx_test, model, features, adj)

    for wid in range(max_iteration+1, max_iteration+max_test_iteration):
        # model.test
        adj, features, labels, idx_train, idx_val, idx_test = load_data(path = data_path, dataset = "sample-plan-" + str(wid))
        test(labels, idx_train, idx_val, idx_test, model, features, adj)

        
if __name__ == "__main__":
    main()

print("Well Done!")
