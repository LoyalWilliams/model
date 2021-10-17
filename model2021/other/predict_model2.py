from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle

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





# load model from file
q2_xg_model = pickle.load(open("D:/code/py/model/model/q2_xg_model", "rb"))
data_dict = DataDict()

df1_test = pd.read_excel('data/ERα_activity.xlsx', sheet_name='test')
df2_test = pd.read_excel('data/Molecular_Descriptor.xlsx', sheet_name='test')
df_ADMET_test = pd.read_excel('data/ADMET.xlsx', sheet_name='test')

# 筛选变量列
sel_col2 = ['ALogP', 'minsOH', 'nRotB', 'maxaaCH', 'MDEC-22', 'VP-1', 'MDEC-23', 'hmin',
            'nBondsM', 'nF10Ring', 'nHBint4', 'nHBint6', 'maxsCH3', 'ETA_dBetaP', 'minHBint3',
            'nHBint7', 'maxwHBa', 'nHBint10', 'SwHBa']

# 标准化
test2x=np.array(df2_test[sel_col2])

test2x_std=(test2x-data_dict.np_std_train_mean[:-1])/data_dict.np_std_train_std[:-1]
# 第二题 模型预测
train_predict = q2_xg_model.predict(
    test2x_std)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]

res=pd.DataFrame()
res['IC50_nM']=np.power(10,9-train_predict)
res['pIC50']=train_predict
res.to_excel('q2_answer.xlsx',index=False)








