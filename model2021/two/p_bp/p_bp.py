#!/usr/bin/env python
# coding: utf-8

# 3.2
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logpath = 'rf.log'

train_end = -395
vaild_end = -197

class DataDict:
    def __init__(self) -> None:
        # 模型相关
        df_all = pd.read_excel('data/q2_data.xlsx')
        # 筛选变量列
        sel_col2=['ALogP','minsOH','nRotB','maxaaCH','MDEC-22','VP-1','MDEC-23','hmin',
        'nBondsM','nF10Ring','nHBint4','nHBint6','maxsCH3','ETA_dBetaP','minHBint3',
        'nHBint7','maxwHBa', 'nHBint10','SwHBa','pIC50']
        # self.np_obv=np.array(df_all[obv])
         # 获取训练数据、原始数据、索引等信息
        df_all=df_all[sel_col2]
        df_train=df_all[:train_end]
        df_valid=df_all[train_end:vaild_end]
        df_test=df_all[vaild_end:]
        
        # 标准化
        ss = StandardScaler()
        self.np_train = np.array(df_train)
        self.np_std_train = ss.fit_transform(self.np_train)
        self.np_std_train_mean = ss.mean_
        self.np_std_train_std = np.sqrt(ss.var_)

        self.np_valid = np.array(df_valid)
        self.np_std_valid = (
            self.np_valid-self.np_std_train_mean)/self.np_std_train_std
        self.np_test = np.array(df_test)
        self.np_std_test = (
            self.np_test-self.np_std_train_mean)/self.np_std_train_std

        self.x_train = self.np_std_train[:, :-1]
        self.y_train = self.np_std_train[:, -1]

        self.x_valid = self.np_std_valid[:, :-1]
        self.y_valid = self.np_std_valid[:, -1]

        self.x_test = self.np_std_test[:, :-1]
        self.y_test = self.np_std_test[:, -1]

# def train(data_dict, max_leaf_nodes=10,
#           n_estimators=100,
#           max_depth=7,
#           max_features=9,
#           min_samples_leaf=60,
#           min_samples_split=1200,
#           min_weight_fraction_leaf=0
#           ):
#     # learning_rate=0.05
#     # n_estimators=1
#     # max_depth=9
#     # min_samples_leaf =60
#     # min_samples_split =1200
#     # max_features=9
#     # subsample=0.7
#     # max_leaf_nodes=10
#     # min_weight_fraction_leaf=0
#     params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
#               'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
#     model = RandomForestRegressor(
#         **params)               # 载入模型（模型命名为model)
#     model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）

#     # 模型预测
#     train_predict = model.predict(
#         data_dict.x_train)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
#     valid_predict = model.predict(
#         data_dict.x_valid)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
#     test_predict = model.predict(
#         data_dict.x_test)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
#     # print(train_predict)

#     # 模型评估 mse
#     train_mse = mean_squared_error(data_dict.np_train[:, -1], train_predict)
#     valid_mse = mean_squared_error(data_dict.np_valid[:, -1], valid_predict)
#     test_mse = mean_squared_error(data_dict.np_test[:, -1], test_predict)

#     print_msg = u'\n=============================== train params ====================================\n'+str(params) +\
#         '\n=============================== model evaluate ====================================\n'+f'train_loss: {train_mse:.5f},' +\
#         f'valid_loss: {valid_mse:.5f},' +\
#         f'test_loss: {test_mse:.5f}\n'
 
#     print(print_msg)
#     return valid_mse

    # model.best_score


# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 500
device=torch.device('cuda', 0)
# torch.device('cpu')

torch.set_default_tensor_type(torch.DoubleTensor)

# 生成训练数据
data_dict = DataDict()
# torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
x = torch.tensor(data_dict.x_train)
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# 0.1 * torch.normal(x.size())增加噪点
# y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
y=torch.unsqueeze(torch.tensor(data_dict.y_train), dim=1)
torch_dataset = Data.TensorDataset(x,y)
#得到一个代批量的生成器
loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True)


class Net(torch.nn.Module):
    # 初始化
    def __init__(self,in_dim, n_hidden_1, n_hidden_2, n_hidden_3,  out_dim):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(in_dim, n_hidden_1)
        self.hidden2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.hidden3 = torch.nn.Linear(n_hidden_2, n_hidden_3)
        self.predict = torch.nn.Linear(n_hidden_3, out_dim)
 
    # 前向传递
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.predict(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net_Momentum = Net(19,100,50,20,1)
opt_Momentum = torch.optim.SGD(net_Momentum.parameters(), lr=LR, momentum=0.9)

# net_Momentum=net_Momentum.to(device)

loss_func = torch.nn.MSELoss() 
loss_his = []  # 记录损失 
for epoch in range(EPOCH):
    for step, (batch_x, batch_y) in enumerate(loader):
        output = net_Momentum(batch_x)  # get output for every net
        loss = loss_func(output, batch_y)  # compute loss for every net
        opt_Momentum.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        opt_Momentum.step()  # apply gradients
        loss_his.append(loss.data.numpy())  # loss recoder
    print(loss_his[-1])
    print(epoch)

torch.save(net_Momentum, 'model/m1')
# net_Momentum=torch.load('D:/code/py/model/model/m1')



x = torch.tensor(data_dict.x_valid)
# x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# 0.1 * torch.normal(x.size())增加噪点
# y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))
y=torch.unsqueeze(torch.tensor(data_dict.y_valid), dim=1)

torch.sum(net_Momentum(x)-y)

plt.plot(loss_his, label='Momentum')
# plt.legend(loc='best')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.ylim((0, 0.2))
# plt.show()
# print(net_Momentum(torch.tensor(x)))
# x = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)  
# print(net_Momentum(x).view(-1))
# plt.plot(x,y)
# # plt.plot(x,x.pow(2))
# plt.plot(x,net_Momentum(x).detach().view(-1))
# plt.show()

