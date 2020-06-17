#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 23:28:20 2020

@author: nanetsu
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
##1. Read data
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

##GCN Layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx

def normalize(A , symmetric=True):
	# A = A+I
	A = A + torch.eye(A.size(0))
	# 所有节点的度
	d = A.sum(1)
	if symmetric:
		#D = D^-1/2
		D = torch.diag(torch.pow(d , -0.5))
		return D.mm(A).mm(D)
	else :
		# D=D^-1
		D =torch.diag(torch.pow(d,-1))
		return D.mm(A)
    
class GCN(nn.Module):
	'''
	Z = AXW
	'''
	def __init__(self , A, dim_in , dim_out):
		super(GCN,self).__init__()
		self.A = A
		self.fc1 = nn.Linear(dim_in ,dim_in,bias=False)
		self.fc2 = nn.Linear(dim_in,dim_in//2,bias=False)
		self.fc3 = nn.Linear(dim_in//2,dim_out,bias=False)

	def forward(self,X):
		'''
		计算三层gcn
		'''
		X = F.relu(self.fc1(self.A.mm(X)))
		X = F.relu(self.fc2(self.A.mm(X)))
		return self.fc3(self.A.mm(X))
    
#获得AML数据
A = nx.adjacency_matrix(G).todense()
#A需要正则化
A_normed = normalize(torch.FloatTensor(A),True)   

     
N = len(A)
N #1000个样本
X_dim = N

# 没有节点的特征，简单用一个单位矩阵表示所有节点
X = torch.eye(N,X_dim)
# 正确结果
Y = torch.zeros(N,1).long()
# 计算loss的时候要去掉没有标记的样本
Y_mask = torch.zeros(N,1,dtype=torch.uint8)
     

Y[0][0]=0
Y[3][0]=0
Y[93][0]=1
Y[137][0]=1

#有样本的地方设置为1
Y_mask[0][0]=1
Y_mask[3][0]=1
Y_mask[93][0]=1
Y_mask[137][0]=1


#真实的空手道俱乐部的分类数据
account=pd.read_csv('~/Desktop/GCN_1/accounts.csv')
RealLabel=account['IS_FRAUD'].replace({False:0,True:1})

# 我们的GCN模型
gcn = GCN(A_normed ,X_dim,2)
#选择adam优化器
gd = torch.optim.Adam(gcn.parameters())
Real=torch.Tensor(RealLabel).long()

for i in range(300):
	#转换到概率空间
    y_pred =F.softmax(gcn(X),dim=1)
    #print(y_pred)
	#下面两行计算cross entropy
    loss = (-y_pred.log().gather(1,Y.view(-1,1)))
	#仅保留有标记的样本
    loss = loss.masked_select(Y_mask).mean()

	#梯度下降
	#清空前面的导数缓存
    gd.zero_grad()
	#求导
    loss.backward()
	#一步更新
    gd.step()
    if i%20==0 :
        _,mi = y_pred.max(1)
		#print(mi)
		#计算精确度
        print((mi == Real).float().mean())



