from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import __init__
import data_source

logpath = 'xgboost.log'


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
    with open(logpath, 'a') as f:
        f.write(print_msg)
    return valid_mse

    # model.best_score


if __name__ == '__main__':
    data_dict = data_source.DataDict()
    train(data_dict)
