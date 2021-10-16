from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd


logpath = 'xgboost.log'


train_end = -395
vaild_end = -197

class DataDict:
    def __init__(self,obv) -> None:
        # 模型相关
        df_all = pd.read_excel('data/q3_data.xlsx')
        # 筛选变量列
        drop_sel=set(['SMILES','Caco-2','CYP3A4','hERG','HOB','MN'])
        self.np_obv=np.array(df_all[obv])
        df_all=df_all.drop(drop_sel, axis=1)
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
        self.x_train = self.np_std_train
        self.y_train = self.np_obv[:train_end]
        self.x_valid = self.np_std_valid
        self.y_valid = self.np_obv[train_end:vaild_end]

        self.x_test = self.np_std_test
        self.y_test = self.np_obv[vaild_end:]
        # self.df_index=df_index
        # self.train_index=df_index[0:train_end]
        # self.valid_index=df_index[train_end:valid_end]
        # self.test_index=df_index[valid_end:]


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
    model = XGBClassifier(**params)               # 载入模型（模型命名为model)
    model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）
    # print(np_std_train.shape)
    # print(np_std_train_std.shape)
    # 模型预测
    train_predict = model.predict(data_dict.x_train)
    valid_predict = model.predict(data_dict.x_valid)
    test_predict = model.predict(data_dict.x_test)
    # print(train_predict)

    # 模型评估 mse
    train_mse = accuracy_score(data_dict.y_train, train_predict)
    valid_mse = accuracy_score(data_dict.y_valid, valid_predict)
    test_mse = accuracy_score(data_dict.y_test, test_predict)

    print_msg = u'\n=============================== train params ====================================\n'+str(params) +\
        '\n=============================== model evaluate ====================================\n'+f'train_loss: {train_mse:.5f},' +\
        f'valid_loss: {valid_mse:.5f},' +\
        f'test_loss: {test_mse:.5f}\n'
 
    print(print_msg)
    return valid_mse



if __name__ == '__main__':
    # data_dict = DataDict('CYP3A4')
    # max_depths        = 77
    # n_estimatorss     = 888
    # min_child_weights = 10.420648272126282
    # learning_rates    = 0.36590576171875
    # subsamples        = 0.646240234375
    # colsample_bytrees = 0.4742431640625
    # gammas            = 2.6152901279072873
    # reg_alphas        = 1.3111164517076557
    # reg_lambdas       = 5.986122078407587
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)

    # data_dict = DataDict('Caco-2')
    # max_depths        =45
    # n_estimatorss     =903
    # min_child_weights =4.946536813876396
    # learning_rates    =0.86822509765625
    # subsamples        =0.627197265625
    # colsample_bytrees =0.251220703125
    # gammas            =0.3749861716696612
    # reg_alphas        =5.78501047138394
    # reg_lambdas       =18.628458513101627
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)

    # data_dict = DataDict('hERG')
    # max_depths        =72
    # n_estimatorss     =814
    # min_child_weights =1.7938300851062206
    # learning_rates    =0.8089599609375
    # subsamples        =0.3648681640625
    # colsample_bytrees =0.7520751953125
    # gammas            =0.2176674563120129
    # reg_alphas        =6.5381871726500425
    # reg_lambdas       =10.75565626394754
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)

    # data_dict = DataDict('MN')    
    # max_depths        =13
    # n_estimatorss     =901
    # min_child_weights =0.450364877185353
    # learning_rates    =0.26324462890625
    # subsamples        =0.90228271484375
    # colsample_bytrees =0.81109619140625
    # gammas            =0.2780161972663775
    # reg_alphas        =0.40718233941016924
    # reg_lambdas       =18.638224175354672
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)


    data_dict = DataDict('HOB') 
    max_depths        =35
    n_estimatorss     =619
    min_child_weights =1.9396283707747297
    learning_rates    =0.5322265625
    subsamples        =0.93365478515625
    colsample_bytrees =0.81610107421875
    gammas            =3.172314347512617
    reg_alphas        =1.5100918201134494
    reg_lambdas       =2.9509084736193603
    train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
          subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)


