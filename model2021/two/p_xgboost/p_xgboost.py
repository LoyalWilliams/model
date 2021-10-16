from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
import pandas as pd


logpath = 'xgboost.log'


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


def train(data_dict, max_depth=20,
          min_child_weight=20,
          learning_rate=1,
          n_estimators=1000,
          subsample=1,
          colsample_bytree=1,
          gamma=20,
          reg_alpha=20,
          reg_lambda=20):
    # data_dict=DataDict()
    # 模型训练
    # max_depth = 20
    # min_child_weight = 20
    # learning_rate = 1
    # n_estimators = 1000
    # subsample = 1
    # colsample_bytree = 1
    # gamma = 20
    # reg_alpha = 20
    # reg_lambda = 20
    params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
              'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}
    model = XGBRegressor(**params)               # 载入模型（模型命名为model)
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

    # model.best_score


if __name__ == '__main__':
    data_dict = DataDict()
    max_depths = 93
    n_estimatorss = 560
    min_child_weights = 7.756072067535658
    learning_rates = 0.223876953125
    subsamples = 0.950927734375
    colsample_bytrees = 0.86456298828125
    gammas = 1.1154217354649942
    reg_alphas = 1.6801516729418675
    reg_lambdas = 15.323544782809382
    train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
          subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)
