# 数据预处理 Python 代码

from sklearn.preprocessing import StandardScaler
from tensorboardX import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def generate_data_by_n_days(series, n, co, obv):
    if len(series) <= n:
        raise Exception(
            "The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = pd.DataFrame()
    print(n)
    print(series.shape)
    for i in range(n):
        new_co = list(map(lambda x: x + "-" + str(n-i-1), co))
        df[new_co] = pd.DataFrame(
            series[i:-(n - i)].values.tolist(), index=series[0:-n].index)
    df['y'] = series[obv][n:].values.tolist()
    return df


# 参数n与上相同。train_end表示的是后面多少个数据作为测试集。
def readData(column='high', n=3, all_too=True, index=False, train_end=-500):
    ###########################################
    df = pd.read_excel(
        'C:\\Users\\25416\\my\\pytest\\paper1\\au.xlsx', index_col=0)
    df = df.rename(columns=lambda x: x.replace(
        "'", "").replace('"', '').replace(" ", ""))
    # df = pd.read_csv("sh300.csv", index_col=0)
    # #以日期为索引
    df.index = list(map(lambda x: datetime.datetime.strptime(
        str.strip(x), "%Y/%m/%d"), df.index))
    df["涨幅"] = df["收盘"].diff()
    df = df.drop(df.index[0])

    # #获取每天的最高价
    df_column = df[co].copy()

    # #生成训练数据
    df_generate = generate_data_by_n_days(df_column, n, co, "涨幅")
    # 拆分为训练集和测试集
    df_column_train, df_column_test = df_generate[:
                                                  train_end], df_generate[train_end:]
    # if all_too:
    return df_column_train, df_column_test, df_generate, df.index.tolist()[n:]
    # return df_generate_train


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


class mytrainset(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


p_n = 39
n = 3
LR = 0.001
EPOCH = 10000
batch_size = 20
train_end = -600

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 7.8.5 训练模型


co = ["收盘", "涨幅", "成交量", "MA.MA1", "MA.MA2", "MA.MA3", "MA.MA4", "VOL.VOLUME",
      "VOL.MAVOL1", "VOL.MAVOL2", "MACD.DIF", "MACD.DEA", "MACD.MACD"]
# 通过一个序列来生成一个31*(count(*)-train_end)矩阵（用于处理时序的数据）
# 其中最后一列维标签数据。就是把当天的前n天作为参数，当天的数据作为label


# 获取训练数据、原始数据、索引等信息
df_column_train, df_column_test, df_all, df_index = readData(
    co, 3, train_end=train_end)
# print(df.shape)
# 可视化数据
close = np.array(df_all["收盘-0"])
plt.plot(df_index, close, label='real-data')
plt.legend(loc='upper right')


# 对数据进行预处理，规范化及转换为Tensor
ss = StandardScaler()
# ss = MinMaxScaler()

df_numpy = np.array(df_column_train)

std_data = ss.fit_transform(df_numpy)

# origin_data = ss.inverse_transform(std_data)

df_numpy_mean = ss.mean_
df_numpy_std = np.sqrt(ss.var_)

# df_numpy = (df_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(std_data)
# print(df_numpy.shape)

trainset = mytrainset(df_tensor)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)


# 记录损失值，并用tensorboardx在web上展示
# tensorboard –-logdir ./logs
# writer = SummaryWriter(log_dir='logs')

# rnn = RNN(p_n).to(device)
# optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
# loss_func = nn.MSELoss()

# for step in range(EPOCH):
#     for tx, ty in trainloader:
#         tx = tx.to(device)
#         ty = ty.to(device)
#         # 在第1个维度上添加一个维度为1的维度，形状变为[batch,seq_len,input_size]
#         output = rnn(torch.unsqueeze(tx, dim=1)).to(device)
#         loss = loss_func(torch.squeeze(output), ty)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     writer.add_scalar('sh300_loss', loss, step)

# torch.save(rnn, 'model/m1')
rnn=torch.load('D:/code/py/model/model/m1')

# 保存整个网络
# torch.save(net, PATH)
# # 保存网络中的参数, 速度快，占空间少
# torch.save(net.state_dict(),PATH)
# #--------------------------------------------------
# #针对上面一般的保存方法，加载的方法分别是：
# model_dict=torch.load(PATH)
# model_dict=model.load_state_dict(torch.load(PATH))

# 7.8.6 测试模型

# 对数据进行预处理，规范化及转换为Tensor
df_test_numpy = np.array(df_all)
# df_test_numpy = np.delete(df_test_numpy, [-1], axis=1)

df_test_numpy = (df_test_numpy - df_numpy_mean) / df_numpy_std
df_tensor = torch.Tensor(df_test_numpy[:, :-1])

generate_data_train = []
generate_data_test = []

for i in range(0, len(df_all)):
    x = df_tensor[i].to(device)
    # rnn的输入必须是3维，故需添加两个1维的维度，最后成为[1,1,input_size]
    x = torch.unsqueeze(torch.unsqueeze(x, dim=0), dim=0)

    y = rnn(x).to(device)
    generate_data_test.append(torch.squeeze(
        y).detach().cpu().numpy() * df_numpy_std[-1] + df_numpy_mean[-1])


plt.clf()
# x -> y   日期 -> 收盘+y （明天）  实际 收盘+1
# x -> y   日期 -> 收盘+y-1   实际 收盘
plt.plot(df_all[train_end:train_end+100].index, df_all[train_end:train_end+100]
         ['收盘-0']+df_all[train_end:train_end+100].y, label='real-data')
plt.plot(df_all[train_end:train_end+100].index, df_all[train_end:train_end+100]['收盘-0'] +
         generate_data_test[train_end:train_end+100], label='generate_test')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error

print(mean_squared_error(df_all[train_end:].y,generate_data_test[train_end:] ))
print(mean_squared_error(df_all[0:train_end].y,generate_data_test[0:train_end] ))
