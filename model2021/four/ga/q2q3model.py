from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import pandas as pd
import pickle



train_end = -395
vaild_end = -197


class DataDict2:
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
        self.sel_col2=sel_col2
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

class DataDict3:
    def __init__(self,obv) -> None:
        # 模型相关
        df_all = pd.read_excel('data/q3_data.xlsx')
        # 筛选变量列
        drop_sel=set(['SMILES','Caco-2','CYP3A4','hERG','HOB','MN','pIC50'])
        self.np_obv=np.array(df_all[obv])
        df_all=df_all.drop(drop_sel, axis=1)
        self.df_all=df_all
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
        


class Models:
    def __init__(self) -> None:
        # load model from file
        self.q3_CYP3A4_model=pickle.load(open("D:/code/py/model/model/q3_CYP3A4_model", "rb"))
        self.q3_Caco_model=pickle.load(open("D:/code/py/model/model/q3_Caco-2_model", "rb"))
        self.q3_hERG_model=pickle.load(open("D:/code/py/model/model/q3_hERG_model", "rb"))
        self.q3_MN_model=pickle.load(open("D:/code/py/model/model/q3_MN_model", "rb"))
        self.q3_HOB_model=pickle.load(open("D:/code/py/model/model/q3_HOB_model", "rb"))
        self.q2_xg_model = pickle.load(open("D:/code/py/model/model/q2_xg_model", "rb"))




if __name__ == '__main__':
    # df1_test = pd.read_excel('data/ERα_activity.xlsx', sheet_name='test')
    df2_test = pd.read_excel('data/Molecular_Descriptor.xlsx', sheet_name='test')
    # df_ADMET_test = pd.read_excel('data/ADMET.xlsx', sheet_name='test')

    df_all = pd.read_excel('data/q3_data.xlsx')
    # 筛选变量列
    drop_sel=set(['SMILES','Caco-2','CYP3A4','hERG','HOB','MN','pIC50'])
    sel_col=set(df_all.columns)-drop_sel

    # 筛选变量列
    # drop_sel=set(['SMILES','Caco-2','CYP3A4','hERG','HOB','MN'])
    # df_all=df2_test.drop(drop_sel, axis=1)

    test2x=np.array(df2_test[sel_col])

    # 标准化
    data_dict=DataDict3('HOB')
    test2x_std=(test2x-data_dict.np_std_train_mean)/data_dict.np_std_train_std

    # 第3题 模型预测

    # load model from file
    q3_CYP3A4_model=pickle.load(open("D:/code/py/model/model/q3_CYP3A4_model", "rb"))
    q3_Caco_model=pickle.load(open("D:/code/py/model/model/q3_Caco-2_model", "rb"))
    q3_hERG_model=pickle.load(open("D:/code/py/model/model/q3_hERG_model", "rb"))
    q3_MN_model=pickle.load(open("D:/code/py/model/model/q3_MN_model", "rb"))
    q3_HOB_model=pickle.load(open("D:/code/py/model/model/q3_HOB_model", "rb"))

    CYP3A4_predict = q3_CYP3A4_model.predict(test2x_std)
    Caco_predict = q3_Caco_model.predict(test2x_std)
    hERG_predict = q3_hERG_model.predict(test2x_std)
    MN_predict = q3_MN_model.predict(test2x_std)
    HOB_predict = q3_HOB_model.predict(test2x_std)







