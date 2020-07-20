import subprocess
import re
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
import networkx
import configparser
import time

from configs.base import base_dir, get_file, table_job, v_feature_size
from db import table_col

# hyper parameters
workload = 'tpch'
sample_rate = 0.01  # |customer|=1500 |orders|=15000 |lineitem|=60175
class_nm = 8
datanode_nm = 6

# column graph
# initialize
def init_adjacant_matrix(size):
    return np.array([[0]*size]*size)

def init_vertex_matrix(size, length):
    return np.array([[0]*size]*length)

def get_key (dict, value):
    return [k for k, v in dict.items() if v == value]

def generate_small_query(predicates):
    """generate predicates on sampled database"""
    # execute join operations
    nm = 0
    op_queries = []
    for i,join in enumerate(predicates):
        cons = join.split()
        tbls = []    
        for c in cons:
            print(c)
            if c in cols and cols[c] not in tbls:
                tbls.append(cols[c])
        if len(tbls) == 2:
            sql = "select small_" + tbls[0] + ".vertex_id as v1, small_" + tbls[1] + ".vertex_id as v2 from small_" + tbls[0] + ",small_" + tbls[1] + " where " + join + ";"  
    #    sql = "selet " + tbl_name[tbls[0]][0] + ',' + tbl_name[tbls[1]][0] + " from " + tbls[0] + "," + tbls[1] + " where " + join + ";"  
            op_queries.append(sql)

        nm = i
    # print(nm)
    return op_queries
    
def generate_graph(qid):
    """ Generate Column Graph 
        [vertices] column features
        [edges] join costs
    """        
    
    # extract columns of tables
    tbl_name = table_col('tpch')   # {'part': ['p_partkey', ...], ...}
    cols = {}
    for tbl in tbl_name:
        for col in tbl_name[tbl]:
            if col not in cols:
                cols[col.lower()] = tbl.lower()   # {'p_partkey': 'part', ...}

    # Initialize vertices
    vertex_num = len(cols) # column number
    adj = init_adjacant_matrix(vertex_num+20)
    vmatrix = init_vertex_matrix(vertex_num+20, v_feature_size)
    
    # extract query  predicates
    predicates,jc_graph, selects = parse_workload('tpch', qid) #todo
    # Generate edge relations
    kdict = {'c_custkey':0,'c_nationkey':1,'o_custkey':2,"o_orderkey":3,'l_orderkey':4,'l_partkey':5,'l_suppkey':6,'n_nationkey':7, 'n_regionkey':8,'ps_suppkey':9,'ps_partkey':10,'s_suppkey':11,'s_nationkey':12,'p_partkey':13,'r_regionkey':14}
    for p in jc_graph:
        cols = p.split(" = ")
        v1 = kdict[cols[0]]
        v2 = kdict[cols[1]]
        w = jc_graph[p] # query frequency
        adj[v1][v2] = adj[v1][v2] + w
        adj[v2][v1] = adj[v2][v1] + w        
    # compute query costs (excute sqls on sampled data and fetch the result scale)
    op_queries = generate_small_query(predicates)
    for sql in op_queries:
        res = execute_sql(sql)
        # card = len(res) todo
    
    # column features
    # [query]
    for p in selects:
        cols = p.split(" ")
        v = kdict[cols[0]]
        vmatrix[v] = vmatrix[v][-2] + 1
    # [column]
    for i in range(vertex_num):
        vmatrix[i][0] = cols[get_key(kdict, i)[0]]
        
    # write Column Graph (V,E) into files
    with open(graph_path + "sample-plan-" + str(wid) + ".content", "w") as wf:
        for v in vmatrix:
            wf.write(str(v[0]) + "\t" + str(v[1]) + "\t" + str(v[2]) + "\t" + str(v[3]) + "\t" + str(v[4]) + "\n")
    with open(graph_path + "sample-plan-" + str(wid) + ".cites", "w") as wf:
        for e in ematrix:
            wf.write(str(adj[0]) + "\t" + str(adj[1]) + "\t" + str(e[2]) + "\n")
        # print(level, num)
