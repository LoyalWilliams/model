# 数据预处理 Python 代码

from sklearn.metrics import mean_squared_error
from pytorchtools import EarlyStopping
from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import __init__
import data_source
import torchvision.transforms as transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


p_n = 39
n = 3
LR = 0.001
EPOCH = 1000
batch_size = 20
train_end = -600
valid_end = -300
patience = 20
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 7.8.4 定义模型

class RNN(nn.Module):
    def __init__(self, input_size):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        self.out = nn.Sequential(
            nn.Linear(64, 1)
        )

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # None即隐层状态用0初始化
        out = self.out(r_out)
        return out


# 7.8.5 训练模型
train_loader, test_loader, valid_loader = data_source.create_datasets()

data_dict = data_source.DataDict()
# train_loader, test_loader, valid_loader

# 记录损失值，并用tensorboardx在web上展示
# tensorboard --logdir ./logs
# writer = SummaryWriter(log_dir='logs')

# rnn = RNN(p_n).to(device)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# loss_func = nn.MSELoss()

# for step in range(EPOCH):
#     for tx, ty in train_loader:
#         tx = tx.to(device)
#         ty = ty.to(device)
#         # 在第1个维度上添加一个维度为1的维度，形状变为[batch,seq_len,input_size]
#         output = rnn(torch.unsqueeze(tx, dim=1)).to(device)
#         loss = loss_func(torch.squeeze(output), ty)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     writer.add_scalar('sh300_loss', loss, step)


def train_model(model, trainloader, valid_loader, optimizer, loss_func, patience, n_epochs):

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    writer = SummaryWriter(log_dir='logs')
    # loss_func = nn.MSELoss()

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train()  # prep model for training
        for tx, ty in train_loader:
            tx = tx.to(device)
            ty = ty.to(device)
            # 在第1个维度上添加一个维度为1的维度，形状变为[batch,seq_len,input_size]
            output = model(torch.unsqueeze(tx, dim=1)).to(device)
            loss = loss_func(torch.squeeze(output), ty)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        writer.add_scalar('sh300_loss', loss, epoch)

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for tx, ty in valid_loader:
            tx = tx.to(device)
            ty = ty.to(device)
            # 在第1个维度上添加一个维度为1的维度，形状变为[batch,seq_len,input_size]
            output = model(torch.unsqueeze(tx, dim=1)).to(device)
            loss = loss_func(torch.squeeze(output), ty)
            valid_losses.append(loss.item())

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        epoch_len = len(str(n_epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{n_epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses


rnn = RNN(p_n).to(device)
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()
train_model(rnn, train_loader, valid_loader,
            optimizer, loss_func, patience, EPOCH)
# torch.save(rnn, 'model/m1')
# rnn=torch.load('D:/code/py/model/model/m1')

# 保存整个网络
# torch.save(net, PATH)
# # 保存网络中的参数, 速度快，占空间少
# torch.save(net.state_dict(),PATH)
# #--------------------------------------------------
# #针对上面一般的保存方法，加载的方法分别是：
# model_dict=torch.load(PATH)
# model_dict=model.load_state_dict(torch.load(PATH))

# 7.8.6 测试模型

# # 对数据进行预处理，规范化及转换为Tensor
# df_test_numpy = np.array(df_all)
# # df_test_numpy = np.delete(df_test_numpy, [-1], axis=1)

# df_test_numpy = (df_test_numpy - df_numpy_mean) / df_numpy_std
# df_tensor = torch.Tensor(df_test_numpy[:, :-1])

# generate_data_train = []
# generate_data_test = []

# for i in range(0, len(df_all)):
#     x = df_tensor[i].to(device)
#     # rnn的输入必须是3维，故需添加两个1维的维度，最后成为[1,1,input_size]
#     x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)

#     y = rnn(x).to(device)
#     generate_data_test.append(torch.squeeze(
#         y).detach().cpu().numpy() * df_numpy_std[-1] + df_numpy_mean[-1])


# plt.clf()
# # x -> y   日期 -> 收盘+y （明天）  实际 收盘+1
# # x -> y   日期 -> 收盘+y-1   实际 收盘
# plt.plot(df_all[train_end:train_end+100].index, df_all[train_end:train_end+100]
#          ['收盘-0']+df_all[train_end:train_end+100].y, label='real-data')
# plt.plot(df_all[train_end:train_end+100].index, df_all[train_end:train_end+100]['收盘-0'] +
#          generate_data_test[train_end:train_end+100], label='generate_test')
# plt.legend()
# plt.show()
# mean_squared_error(df_all[train_end:train_end+100]
#                    ['收盘-0']+df_all[train_end:train_end+100].y, df_all[train_end:train_end+100]['收盘-0'] +
#                    generate_data_test[train_end:train_end+100])

test_loss=[]
predict=[]
test_x=[]
data_dict.np_std_train_mean
# rnn=torch.load('D:/code/py/model/model/m1')
# i=0
# test_data=data_dict.np_std_test[0:100]
# x=torch.unsqueeze(torch.tensor(test_data[:,0:-1]).float(), dim=1)
# predict=rnn(x.to(device)).view(-1).detach().cpu().numpy()

# plt.clf()
# plt.plot(data_dict.test_index[0:100],predict)

# plt.show()