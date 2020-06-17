#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 11:22:46 2020

@author: nanetsu
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

##1. read data

account=pd.read_csv('~/Desktop/GCN_1/accounts.csv')
alert=pd.read_csv('~/Desktop/GCN_1/alerts.csv')
transaction=pd.read_csv('~/Desktop/GCN_1/transactions.csv')

transaction.columns.tolist()

#2. data preparation 点必须是string格式，加字母会好很多
aa=transaction['SENDER_ACCOUNT_ID']
sender=pd.Series(['S'+str(i) for i in aa],name='sender')
bb=transaction['RECEIVER_ACCOUNT_ID']
receiver=pd.Series(['S'+str(i) for i in bb],name='receiver')

transaction['SENDER_ACCOUNT_ID']=sender
transaction['RECEIVER_ACCOUNT_ID']=receiver

transaction.head()
nn=transaction[ 'TX_AMOUNT']


##2.construct network
G=nx.Graph()
G.add_node(sender.iloc[0])
G.add_nodes_from(sender.iloc[1:])
G.add_nodes_from(receiver)

peredge=transaction.iloc[:,[1,2,4]]
edges=np.array(peredge)
G.add_weighted_edges_from(edges) 
#nx.draw(G)
#plt.show()

#3. properties
G.degree()
point=list(G.degree().keys())
len(point) ##1000个账户

## 4.Graph Embedding-DeepWalk
#该算法主要包括两个步骤，第一步是随机游走采样节点序列，第二歩是使用skip gram modelword2vec学习表达向量
#a.构建同构网络，从网络中的每个节点分别进行随机游走采样，得到局部相关联的训练数据
#b.对采样数据进行SkipGram训练，将离散的网络节点表示成向量化，最大化节点共现
#使用hierarchical softmax来做超大规模分类的分类器
from ge import DeepWalk
    
model=_simulate_walks(G,walk_length=10,num_walks=80,workers=1)
model.train(window_size=5,iter=3)
embeddings=model.get_embeddings()

evaluate_embeddings(embeddings)
plot_embeddings(embeddings)



#graph到numpy的转换
ba = nx.barabasi_albert_graph(10, 5)
a = nx.to_numpy_matrix(ba)
print(a)


import shap
import numpy as np

# select a set of background examples to take an expectation over
background = X[np.random.choice(X.shape[0], 1000, replace=False)]
background.shape

X[1].shape
# explain predictions of the model on three images
e = shap.DeepExplainer(gcn, background)
# ...or pass tensors directly
shap_values = e.shap_values(X[1:1000])
# plot the feature attributions
shap.image_plot(shap_values, -X[1])
X.shape ##1000*1000

X[:,1].shape