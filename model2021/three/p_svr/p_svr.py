from sklearn import datasets, svm
from sklearn.metrics import mean_squared_error
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


def train(data_dict, kernel='rbf',
          C=1,
          degree=1,
          gamma=1,
          coef0=1,
          ):
    params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0}
    model = svm.SVC(**params)               # 载入模型（模型命名为model)
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
    # kernel='linear'
    # degree=8
    # C     =2.481555938720703
    # gamma =48.54907989501953
    # coef0 =35.8707770068903
    # train(data_dict,kernel=kernel,degree=degree,C=C,gamma=gamma,coef0=coef0)
   

    # data_dict = DataDict('MN')
    # kernel='linear'
    # degree=5
    # C     =5.950641632080078
    # gamma =75.25835037231445
    # coef0 =84.26493097775553
    # train(data_dict,kernel=kernel,degree=degree,C=C,gamma=gamma,coef0=coef0)
    
    # data_dict = DataDict('HOB')
    # kernel='linear'
    # degree=3
    # C     =3.6070823669433594
    # gamma =40.64168930053711
    # coef0 =67.76644493717664
    # train(data_dict,kernel=kernel,degree=degree,C=C,gamma=gamma,coef0=coef0)
   

    # data_dict = DataDict('Caco-2')
    # kernel='linear'
    # degree=1
    # C     =83.84809494018555
    # gamma =44.15740966796875
    # coef0 =60.71516105190378
    # train(data_dict,kernel=kernel,degree=degree,C=C,gamma=gamma,coef0=coef0)
   

    data_dict = DataDict('hERG')
    kernel='linear'
    degree=8
    C     =1.3010025024414062
    gamma =91.88404083251953
    coef0 =53.59454497770784
    train(data_dict,kernel=kernel,degree=degree,C=C,gamma=gamma,coef0=coef0)
   
