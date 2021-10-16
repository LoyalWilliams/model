from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

logpath = 'gbdt.log'


train_end = -395
vaild_end = -197


class DataDict:
    def __init__(self) -> None:
        # 模型相关
        df_all = pd.read_excel('data/q2_data.xlsx')
        # 筛选变量列
        sel_col2 = ['ALogP', 'minsOH', 'nRotB', 'maxaaCH', 'MDEC-22', 'VP-1', 'MDEC-23', 'hmin',
                    'nBondsM', 'nF10Ring', 'nHBint4', 'nHBint6', 'maxsCH3', 'ETA_dBetaP', 'minHBint3',
                    'nHBint7', 'maxwHBa', 'nHBint10', 'SwHBa', 'pIC50']
        # self.np_obv=np.array(df_all[obv])
        # 获取训练数据、原始数据、索引等信息
        df_all = df_all[sel_col2]
        df_train = df_all[:train_end]
        df_valid = df_all[train_end:vaild_end]
        df_test = df_all[vaild_end:]

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


def train(data_dict, max_leaf_nodes=10,
          n_estimators=100,
          max_depth=7,
          max_features=9,
          learning_rate=0.05,
          min_samples_leaf=60,
          min_samples_split=1200,
          subsample=0.7,
          min_weight_fraction_leaf=0
          ):
    # learning_rate=0.05
    # n_estimators=1
    # max_depth=9
    # min_samples_leaf =60
    # min_samples_split =1200
    # max_features=9
    # subsample=0.7
    # max_leaf_nodes=10
    # min_weight_fraction_leaf=0
    params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
    model = GradientBoostingRegressor(
        **params)               # 载入模型（模型命名为model)
    model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）

    # 模型预测
    train_predict = model.predict(
        data_dict.x_train)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    valid_predict = model.predict(
        data_dict.x_valid)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    test_predict = model.predict(
        data_dict.x_test)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    # print(train_predict)

    # 模型评估 mse
    train_mse = mean_squared_error(data_dict.np_train[:, -1], train_predict)
    valid_mse = mean_squared_error(data_dict.np_valid[:, -1], valid_predict)
    test_mse = mean_squared_error(data_dict.np_test[:, -1], test_predict)

    print_msg = u'\n=============================== train params ====================================\n'+str(params) +\
        '\n=============================== model evaluate ====================================\n'+f'train_loss: {train_mse:.5f},' +\
        f'valid_loss: {valid_mse:.5f},' +\
        f'test_loss: {test_mse:.5f}\n'

    print(print_msg)
    return valid_mse


if __name__ == '__main__':
    data_dict = DataDict()
    max_leaf_nodes = 11
    n_estimators = 872
    max_depth = 41
    max_features = 14
    learning_rate = 0.23187255859375
    min_samples_leaf = 0.15362548828125
    min_samples_split = 0.23052978515625
    subsample = 0.93115234375
    min_weight_fraction_leaf = 0.1491881333170553

    train(data_dict, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
          min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, subsample=subsample, min_weight_fraction_leaf=min_weight_fraction_leaf)
