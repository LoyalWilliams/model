# 问题一数据预处理 Python 代码
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.inspection import permutation_importance

logpath = 'rf.log'

train_end = -395
vaild_end = -197


class DataDict:
    def __init__(self) -> None:
        # 模型相关
        df_all = pd.read_excel('data/clean2.xlsx')
        # 筛选变量列
        sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311,322]]
        df_all=df_all[sel_col]

        df_all=df_all.sample(frac=1)
        df_train=df_all[:train_end]
        df_valid=df_all[train_end:vaild_end]
        df_test=df_all[vaild_end:]
        # df_valid, df_test, df_all, df_index=df4[]
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

        # 标准化后的数据
        self.x_train = self.np_std_train[:, :-1]
        self.y_train = self.np_std_train[:, -1]
        self.x_valid = self.np_std_valid[:, :-1]
        self.y_valid = self.np_std_valid[:, -1]

        self.x_test = self.np_std_test[:, :-1]
        self.y_test = self.np_std_test[:, -1]
        # self.df_index=df_index
        # self.train_index=df_index[0:train_end]
        # self.valid_index=df_index[train_end:valid_end]
        # self.test_index=df_index[valid_end:]




def train(data_dict, max_leaf_nodes=10,
          n_estimators=100,
          max_depth=7,
          max_features=9,
          min_samples_leaf=60,
          min_samples_split=1200,
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
    params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
    model = RandomForestRegressor(
        **params)               # 载入模型（模型命名为model)
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
    # with open(logpath, 'a') as f:
    #     f.write(print_msg)
    return valid_mse


if __name__ == '__main__':
    data_dict = DataDict()
    # learning_rate=0.05
    n_estimators=457
    max_depth=27
    min_samples_leaf =0.0006103515625
    min_samples_split =0.01171875
    max_features=13
    max_leaf_nodes=49
    min_weight_fraction_leaf=0.00042729825418141864
    train(data_dict,n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split
    ,max_features=max_features,max_leaf_nodes=max_leaf_nodes)
    # model = RandomForestRegressor()               # 载入模型（模型命名为model)
    # model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）
    # model.get_params

# data_dict
# result = permutation_importance(
# model, data_dict.x_train, data_dict.y_train, n_repeats=10, random_state=42, n_jobs=2)
# zip(sel_col,result.importances_mean)
# sorted(zip(result.importances_mean,sel_col),key=lambda x:-x[0])



# X, y = make_hastie_10_2(random_state=0)
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
#     max_depth=1, random_state=0).fit(X, y)
# clf.feature_importances_

# np.delete(np.array(df4.columns),0)
# dict(zip(np.delete(np.array(df4.columns),0),model.feature_importances_))

# sorted(zip(model.feature_importances_,np.delete(np.array(df4.columns),0)),key=lambda x:-x[0])








