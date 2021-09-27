from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import __init__
import data_source


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
    # print(np_std_train.shape)
    # print(np_std_train_std.shape)
    # 模型预测
    train_predict = model.predict(
        data_dict.x_train)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    valid_predict = model.predict(
        data_dict.x_valid)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    test_predict = model.predict(
        data_dict.x_test)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    print(train_predict)

    # 模型评估 mse
    train_mse = mean_squared_error(data_dict.np_train[:, -1], train_predict)
    valid_mse = mean_squared_error(data_dict.np_valid[:, -1], valid_predict)
    test_mse = mean_squared_error(data_dict.np_test[:, -1], test_predict)

    print_msg = (f'train_loss: {train_mse:.5f}\n' +
                 f'valid_loss: {valid_mse:.5f}\n' +
                 f'test_loss: {test_mse:.5f}\n')
    print(print_msg)
    with open(data_dict.log, 'w') as f:
        f.write(print_msg)
    return valid_mse

    # model.best_score


class DataDict:
    def __init__(self) -> None:
        # 模型相关
        self.data_path = 'C:\\Users\\25416\\my\\pytest\\paper1\\au.xlsx'
        self.log = 'd:\\code\\py\\model\\deeplearning\\stock_predict\\p_xgboost\\mse.log'
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
        df_train, df_valid, df_test, df_all, df_index = data_source.readData(
            co, 3, train_end=train_end)

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


if __name__ == '__main__':
    data_dict = DataDict()
    train(data_dict)
