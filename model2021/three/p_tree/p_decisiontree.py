from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
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


def train(data_dict, max_leaf_nodes=10,
          max_depth=7,
          max_features=9,
          min_samples_leaf=60,
          min_samples_split=1200,
          min_weight_fraction_leaf=0
          ):
    params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
    model = DecisionTreeClassifier(
        **params)               # 载入模型（模型命名为model)
    model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）

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
    # data_dict = DataDict('Caco-2')
    # max_leaf_nodes           =19
    # max_depth                =27
    # max_features             =110
    # min_samples_leaf         =0.00555419921875
    # min_samples_split        =0.05859375
    # min_weight_fraction_leaf =0.016115248443413503
    # train(data_dict,max_leaf_nodes=max_leaf_nodes,max_features=max_features,min_samples_leaf=min_samples_leaf,
    # min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf)
    
    # data_dict = DataDict('CYP3A4')
    # max_leaf_nodes           =29
    # max_depth                =41
    # max_features             =7
    # min_samples_leaf         =0.23931884765625
    # min_samples_split        =0.54248046875
    # min_weight_fraction_leaf =0.2405689171041387
    # train(data_dict,max_leaf_nodes=max_leaf_nodes,max_features=max_features,min_samples_leaf=min_samples_leaf,
    # min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf)


    # data_dict = DataDict('HOB')
    # max_leaf_nodes           =28
    # max_depth                =12
    # max_features             =169
    # min_samples_leaf         =0.064453125
    # min_samples_split        =0.57879638671875
    # min_weight_fraction_leaf =0.06525454767427664
    # train(data_dict,max_leaf_nodes=max_leaf_nodes,max_features=max_features,min_samples_leaf=min_samples_leaf,
    # min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf)

    # data_dict = DataDict('hERG')
    # max_leaf_nodes           =34
    # max_depth                =5
    # max_features             =241
    # min_samples_leaf         =0.1278076171875
    # min_samples_split        =0.93157958984375
    # min_weight_fraction_leaf =0.26022463679648394
    # train(data_dict,max_leaf_nodes=max_leaf_nodes,max_features=max_features,min_samples_leaf=min_samples_leaf,
    # min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf)


    data_dict = DataDict('MN')
    max_leaf_nodes           =8
    max_depth                =25
    max_features             =75
    min_samples_leaf         =0.01220703125
    min_samples_split        =0.03167724609375
    min_weight_fraction_leaf =0.017824441460139177
    train(data_dict,max_leaf_nodes=max_leaf_nodes,max_features=max_features,min_samples_leaf=min_samples_leaf,
    min_samples_split=min_samples_split,min_weight_fraction_leaf=min_weight_fraction_leaf)

