from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle


train_end = -395
vaild_end = -197

class DataDict:
    def __init__(self,obv) -> None:
        # 模型相关
        df_all = pd.read_excel('data/q3_data.xlsx')
        # 筛选变量列
        drop_sel=set(['SMILES','Caco-2','CYP3A4','hERG','HOB','MN','pIC50'])
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

    # pickle.dump(model, open("model/q3_CYP3A4_model", "wb"))
    # pickle.dump(model, open("model/q3_Caco-2_model", "wb"))
    # pickle.dump(model, open("model/q3_hERG_model", "wb"))
    # pickle.dump(model, open("model/q3_MN_model", "wb"))
    pickle.dump(model, open("model/q3_HOB_model", "wb"))
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
    # max_depths        =98
    # n_estimatorss     =835
    # min_child_weights =6.582056358552393
    # learning_rates    =0.88189697265625
    # subsamples        =0.507080078125
    # colsample_bytrees =0.33221435546875
    # gammas            =9.583776793582128
    # reg_alphas        =1.4049583624205109
    # reg_lambdas       =9.03194058204873
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)

    # data_dict = DataDict('Caco-2')
    # max_depths        =32
    # n_estimatorss     =680
    # min_child_weights =5.196324143692565
    # learning_rates    =0.5970458984375
    # subsamples        =0.8787841796875
    # colsample_bytrees =0.7271728515625
    # gammas            =2.6043800521089633
    # reg_alphas        =0.8056671358762202
    # reg_lambdas       =14.394738749461172
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)

    # data_dict = DataDict('hERG')
    # max_depths        =88
    # n_estimatorss     =282
    # min_child_weights =1.7330998729700964
    # learning_rates    =0.87445068359375
    # subsamples        =0.93231201171875
    # colsample_bytrees =0.315185546875
    # gammas            =0.29213063099148173
    # reg_alphas        =2.423791594663981
    # reg_lambdas       =11.038631586576791
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)

    # data_dict = DataDict('MN')
    # max_depths        =28
    # n_estimatorss     =726
    # min_child_weights =9.282872325410176
    # learning_rates    =0.75
    # subsamples        =0.6002197265625
    # colsample_bytrees =0.5777587890625
    # gammas            =1.8197701254658716
    # reg_alphas        =0.05920432740908588
    # reg_lambdas       =10.64609774054619
    # train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
    #       subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)


    data_dict = DataDict('HOB') 
    max_depths        =23
    n_estimatorss     =811
    min_child_weights =16.229691427961075
    learning_rates    =0.91937255859375
    subsamples        =0.87701416015625
    colsample_bytrees =0.0101318359375
    gammas            =0.5423757262257622
    reg_alphas        =7.140224991703001
    reg_lambdas       =17.88855700896076
    train(data_dict, max_depth=max_depths, n_estimators=n_estimatorss, min_child_weight=min_child_weights, learning_rate=learning_rates,
          subsample=subsamples, colsample_bytree=colsample_bytrees, gamma=gammas, reg_alpha=reg_alphas, reg_lambda=reg_lambdas)


