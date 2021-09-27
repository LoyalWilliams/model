#!/usr/bin/env python
# coding: utf-8

# 3.2
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 超参数
LR = 0.01
BATCH_SIZE = 32
EPOCH = 500
device=torch.device('cuda', 0)
# torch.device('cpu')


# 生成训练数据
# torch.unsqueeze() 的作用是将一维变二维，torch只能处理二维的数据
x = torch.unsqueeze(torch.linspace(-1, 1, 1000), dim=1)
# 0.1 * torch.normal(x.size())增加噪点
y = x.pow(2) + 0.1 * torch.normal(torch.zeros(*x.size()))

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
net_Momentum = Net(1,100,50,20,1)
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
    print(epoch)


# plt.plot(loss_his, label='Momentum')
# plt.legend(loc='best')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.ylim((0, 0.2))
# plt.show()
# print(net_Momentum(torch.tensor(x)))
# x = torch.unsqueeze(torch.linspace(-10, 10, 100), dim=1)  
# print(net_Momentum(x).view(-1))
plt.plot(x,y)
plt.plot(x,x.pow(2))
plt.plot(x,net_Momentum(x).detach().view(-1))
plt.show()

