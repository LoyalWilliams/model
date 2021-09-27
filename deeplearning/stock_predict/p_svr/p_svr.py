
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm
import numpy as np
from sklearn.metrics import mean_squared_error
import __init__
from data_source import DataDict

logpath = 'svr.log'


def train(data_dict, kernel='rbf',
          C=1,
          degree=1,
          gamma=1,
          coef0=1,
          ):
    params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0}
    print('===============================训练参数====================================')
    print(params)
    model = svm.SVR(**params)               # 载入模型（模型命名为model)
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
    with open(logpath, 'a') as f:
        f.write(print_msg)
    return valid_mse


if __name__ == '__main__':
    data_dict = DataDict()
    train(data_dict)
