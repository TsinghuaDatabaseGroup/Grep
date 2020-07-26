
from __future__ import division
from __future__ import print_function
from model.simple_gcn import GCN
from load_data.load_training_sample import load_data, accuracy

import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F
import torch.optim as optim

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

def train(epoch, labels, idx_train, idx_val, idx_test, model, optimizer, features, adj, evaluator, use_evaluate):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    # print(output[idx_train])

    if use_evaluate == 1:
        # labels = evaluator(output)
        pass
    loss_train = F.mse_loss(output[idx_train], labels[idx_train])
    
    # loss_train = nn.CrossEntropyLoss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.mse_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

def test(labels, idx_train, idx_val, idx_test, model, features, adj):
    model.eval()
    output = model(features, adj)
    
    loss_test = F.mse_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()))
