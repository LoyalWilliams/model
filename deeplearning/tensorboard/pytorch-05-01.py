#!/usr/bin/env python
# coding: utf-8

# <h3>5.3 过拟合与欠拟合</h3>

# 5.3.2 dropout正则化

# In[1]:


from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy  as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

boston = load_boston()
X,y   = (boston.data, boston.target)
dim = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
num_train = X_train.shape[0]


# In[2]:


#对训练数据进行标准化
mean=X_train.mean(axis=0)
std=X_train.std(axis=0)


# In[3]:


X_train-=mean
X_train/=std

X_test-=mean
X_test/=std


# In[4]:


train_data=torch.from_numpy(X_train)


# In[5]:


dtype = torch.FloatTensor
train_data.type(dtype)


# In[6]:


#实例化模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = torch.device("cuda:0")
#device1 = torch.device("cuda:1")
train_data=torch.from_numpy(X_train).float()
train_target=torch.from_numpy(y_train).float()
test_data=torch.from_numpy(X_test).float()
test_target=torch.from_numpy(y_test).float()


# In[7]:


net1_overfitting = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

net2_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    nn.BatchNorm1d(num_features=16),
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    nn.BatchNorm1d(num_features=32),  
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)

net1_nb = torch.nn.Sequential(
    torch.nn.Linear(13, 8),
    nn.BatchNorm1d(num_features=8),
    torch.nn.ReLU(),
    torch.nn.Linear(8, 4),
    nn.BatchNorm1d(num_features=4),  
    torch.nn.ReLU(),
    torch.nn.Linear(4, 1),
)

net1_dropped = torch.nn.Sequential(
    torch.nn.Linear(13, 16),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(16, 32),
    torch.nn.Dropout(0.5),  # drop 50% of the neuron
    torch.nn.ReLU(),
    torch.nn.Linear(32, 1),
)


# In[8]:


loss_func = torch.nn.MSELoss()
optimizer_ofit = torch.optim.Adam(net1_overfitting.parameters(), lr=0.01)
optimizer_drop = torch.optim.Adam(net1_dropped.parameters(), lr=0.01)
optimizer_nb = torch.optim.Adam(net1_nb.parameters(), lr=0.01)


# In[9]:


from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs')
for epoch in range(200):
    net1_overfitting.train()
    net1_dropped.train()
    net1_nb.train()
    

    pred_ofit=  net1_overfitting(train_data)
    pred_drop = net1_dropped(train_data)
    pred_nb = net1_nb(train_data)
    
    loss_ofit = loss_func(pred_ofit, train_target)
    loss_drop = loss_func(pred_drop, train_target)
    loss_nb = loss_func(pred_nb, train_target)
    
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    optimizer_nb.zero_grad()
    
    loss_ofit.backward()
    loss_drop.backward()
    loss_nb.backward()

    
    optimizer_ofit.step()
    optimizer_drop.step()
    optimizer_nb.step()
    # 保存loss的数据与epoch数值
    #writer.add_scalar('train_loss', loss_ofit, t)
    writer.add_scalars('train_group_loss',{'trainloss':loss_ofit.item(),'testloss':loss_nb.item()}, epoch)
    # change to eval mode in order to fix drop out effect
    net1_overfitting.eval()
    net1_dropped.eval() 
    net1_nb.eval() 
   
    test_pred_orig = net1_overfitting(test_data)
    test_pred_drop = net1_dropped(test_data)
    test_pred_nb = net1_nb(test_data)
    orig_loss=loss_func(test_pred_orig, test_target)
    drop_loss=loss_func(test_pred_drop, test_target)
    nb_loss=loss_func(test_pred_nb, test_target)
    #writer.add_scalars('test_nb_loss',{'orig_loss':orig_loss.item(),'nb_loss':nb_loss.item()}, epoch)
    writer.add_scalars('test_group_loss',{'droploss':drop_loss.item(),'origloss':orig_loss.item()}, epoch)


# 5.3.3 批量正则化

# In[10]:


from tensorboardX import SummaryWriter
writer = SummaryWriter(log_dir='logs')
for epoch in range(200):
    net1_overfitting.train()
    net1_dropped.train()
    net1_nb.train()
    
    pred_ofit = net1_overfitting(train_data)
    pred_drop = net1_dropped(train_data)
    pred_nb = net1_nb(train_data)
    
    loss_ofit = loss_func(pred_ofit, train_target)
    loss_drop = loss_func(pred_drop, train_target)
    loss_nb = loss_func(pred_nb, train_target)
    
    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    optimizer_nb.zero_grad()
    
    loss_ofit.backward()
    loss_drop.backward()
    loss_nb.backward()
    
    optimizer_ofit.step()
    optimizer_drop.step()
    optimizer_nb.step()
    # 保存loss的数据与epoch数值
    #writer.add_scalar('train_loss', loss_ofit, t)
    writer.add_scalars('train_group_loss',{'trainloss':loss_ofit.item(),'testloss':loss_nb.item()}, epoch)
    writer.add_scalars('test_nb_loss',{'orig_loss':orig_loss.item(),'nb_loss':nb_loss.item()}, epoch)


# In[ ]:




