from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

logpath = 'gbdt.log'


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
    model = GradientBoostingClassifier(
        **params)               # 载入模型（模型命名为model)
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
    data_dict = DataDict('CYP3A4')
    max_leaf_nodes           =37
    n_estimators             =576
    max_depth                =45
    max_features             =177
    learning_rate            =0.5823974609375
    min_samples_leaf         =0.181884765625
    min_samples_split        =0.08282470703125
    subsample                =0.47210693359375
    min_weight_fraction_leaf =0.13343914052008302

    train(data_dict, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
          min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, subsample=subsample, min_weight_fraction_leaf=min_weight_fraction_leaf)

    # data_dict = DataDict('MN')
    # max_leaf_nodes           =2
    # n_estimators             =494
    # max_depth                =13
    # max_features             =120
    # learning_rate            =0.9859619140625
    # min_samples_leaf         =0.020751953125
    # min_samples_split        =0.73321533203125
    # subsample                =0.8399658203125
    # min_weight_fraction_leaf =0.019899890123306067
    # train(data_dict, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
    #       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, subsample=subsample, min_weight_fraction_leaf=min_weight_fraction_leaf)

    # data_dict = DataDict('Caco-2')
    # max_leaf_nodes           =25
    # n_estimators             =927
    # max_depth                =34
    # max_features             =158
    # learning_rate            =0.62066650390625
    # min_samples_leaf         =0.11676025390625
    # min_samples_split        =0.18145751953125
    # subsample                =0.88775634765625
    # min_weight_fraction_leaf =0.11256256867293371
    # train(data_dict, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
    #       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, subsample=subsample, min_weight_fraction_leaf=min_weight_fraction_leaf)


    # data_dict = DataDict('hERG')
    # max_leaf_nodes           =15
    # n_estimators             =666
    # max_depth                =33
    # max_features             =296
    # learning_rate            =0.42877197265625
    # min_samples_leaf         =0.0740966796875
    # min_samples_split        =0.16741943359375
    # subsample                =0.95684814453125
    # min_weight_fraction_leaf =0.15980954706385056
    # train(data_dict, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
    #       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, subsample=subsample, min_weight_fraction_leaf=min_weight_fraction_leaf)


    # data_dict = DataDict('HOB')
    # max_leaf_nodes           =7
    # n_estimators             =796
    # max_depth                =17
    # max_features             =240
    # learning_rate            =0.56414794921875
    # min_samples_leaf         =0.0030517578125
    # min_samples_split        =0.14141845703125
    # subsample                =0.71466064453125
    # min_weight_fraction_leaf =0.03259675253326822
    # train(data_dict, max_leaf_nodes=max_leaf_nodes, n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, learning_rate=learning_rate,
    #       min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, subsample=subsample, min_weight_fraction_leaf=min_weight_fraction_leaf)



