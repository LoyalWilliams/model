# 数据预处理 Python 代码
import os
import sys
# 获取当前文件的目录
curPath = os.path.abspath(os.path.dirname(__file__))
index = curPath.index('stock_predict')
sys.path.append(curPath[:index]+'stock_predict')
index2 = curPath.index('deeplearning')

from sklearn.metrics import mean_squared_error
from pytorchtools import EarlyStopping
from sklearn.preprocessing import StandardScaler
import pandas as pd
import datetime
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

data_path = curPath[:index2]+'data/au.xlsx'
p_n = 39
n = 3
LR = 0.001
EPOCH = 10000
batch_size = 20
train_end = -600
valid_end = -300
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
co = ["收盘", "涨幅", "成交量", "MA.MA1", "MA.MA2", "MA.MA3", "MA.MA4", "VOL.VOLUME",
      "VOL.MAVOL1", "VOL.MAVOL2", "MACD.DIF", "MACD.DEA", "MACD.MACD"]

# # 获取训练数据、原始数据、索引等信息
# df_train, df_valid, df_test, df_all, df_index = readData(
#     co, 3, train_end=train_end)


def generate_data_by_n_days(series, n, co, obv):
    if len(series) <= n:
        raise Exception(
            "The Length of series is %d, while affect by (n=%d)." % (len(series), n))
    df = pd.DataFrame()
    # # print(n)
    # print(series.shape)
    for i in range(n):
        new_co = list(map(lambda x: x + "-" + str(n-i-1), co))
        df[new_co] = pd.DataFrame(
            series[i:-(n - i)].values.tolist(), index=series[0:-n].index)
    df['y'] = series[obv][n:].values.tolist()
    return df


# 参数n与上相同。train_end表示的是后面多少个数据作为测试集。
def readData(column, n=3, train_end=-600, valid_end=-300, obv="涨幅"):
    ###########################################
    df = pd.read_excel(
        data_path, index_col=0)
    df = df.rename(columns=lambda x: x.replace(
        "'", "").replace('"', '').replace(" ", ""))
    # df = pd.read_csv("sh300.csv", index_col=0)
    # #以日期为索引
    df.index = list(map(lambda x: datetime.datetime.strptime(
        str.strip(x), "%Y/%m/%d"), df.index))
    df["涨幅"] = df["收盘"].diff()
    df = df.drop(df.index[0])
    df_column = df[co].copy()

    # 生成训练数据
    df_generate = generate_data_by_n_days(df_column, n, co, obv)
    # 拆分为训练集和测试集,验证集
    df_train, df_valid, df_test = df_generate[:
                                              train_end], df_generate[train_end:valid_end], df_generate[valid_end:]

    return df_train, df_valid, df_test, df_generate, df.index.tolist()[n:]


class mytrainset(Dataset):
    def __init__(self, data):
        self.data, self.label = data[:, :-1].float(), data[:, -1].float()

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def create_datasets(batch_size=20):

    # 获取训练数据、原始数据、索引等信息
    df_train, df_valid, df_test ,df_all, df_index = readData(
        co, 3, train_end=train_end)
    # 训练集，对数据进行预处理，规范化及转换为Tensor
    ss = StandardScaler()
    df_numpy = np.array(df_train)
    std_data = ss.fit_transform(df_numpy)
    df_numpy_mean = ss.mean_
    df_numpy_std = np.sqrt(ss.var_)

    df_tensor = torch.Tensor(std_data)
    trainset = mytrainset(df_tensor)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    # 验证集和测试集，对数据进行预处理，规范化及转换为Tensor
    df_all_numpy = np.array(df_all)

    df_all_numpy = (df_all_numpy - df_numpy_mean) / df_numpy_std

    valid_loader = DataLoader(mytrainset(
        torch.Tensor(df_all_numpy[train_end:valid_end])), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mytrainset(
        torch.Tensor(df_all_numpy[valid_end:])), batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, valid_loader



class DataDict:
    def __init__(self, obv="涨幅") -> None:
        # 模型相关
        p_n = 39
        n = 3
        LR = 0.001
        EPOCH = 10000
        batch_size = 20
        train_end = -600
        valid_end = -300
        co = ["收盘", "涨幅", "成交量", "MA.MA1", "MA.MA2", "MA.MA3", "MA.MA4", "VOL.VOLUME",
              "VOL.MAVOL1", "VOL.MAVOL2", "MACD.DIF", "MACD.DEA", "MACD.MACD"]

        # 获取训练数据、原始数据、索引等信息
        df_train, df_valid, df_test, df_all, df_index = readData(
            co, 3, train_end=train_end, obv=obv)

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
