# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import q2q3model

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 322  # 初始化Dim（决策变量维数）

        # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        varTypes = [0]*Dim


        self.data_dict2 = q2q3model.DataDict2()
        self.data_dict3 = q2q3model.DataDict3('HOB')
        self.models = q2q3model.Models()

        df_all=self.data_dict3.df_all

        # 目标列索引
        sel_col2=self.data_dict2.sel_col2.copy()
        sel_col2.pop()
        self.f_col= [ list(df_all.columns).index(i) for i in sel_col2]

        desc=df_all.describe()
        # up_df=desc.loc['max']+(desc.loc['max']-desc.loc['min'])*0.15
        # down_df=desc.loc['min']-(desc.loc['max']-desc.loc['min'])*0.15
        up_df=desc.loc['max']
        down_df=desc.loc['min']
        # up_df=(desc.loc['max']-desc.loc['min'])*0.15
        lb = list(down_df)  # 决策变量下界
        ub = list(up_df)  # 决策变量上界
        lbin = [1]*Dim   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        # rows=Vars.shape[0]
        # for i in range(rows):
        #     x=Vars[i,:]
        #     self.models.q3_Caco_model(x)

        # 标准化
        Vars_g=(Vars-self.data_dict3.np_std_train_mean)/self.data_dict3.np_std_train_std
        Vars_f=(Vars[:,self.f_col]-self.data_dict2.np_std_train_mean[:-1])/self.data_dict2.np_std_train_std[:-1]


        # 采用可行性法则处理约束
        # print(type(Vars_g))
        # print(Vars_g)
        cv=3-(self.models.q3_CYP3A4_model.predict(Vars_g)+self.models.q3_Caco_model.predict(Vars_g)+\
        self.models.q3_hERG_model.predict(Vars_g)+self.models.q3_HOB_model.predict(Vars_g)+self.models.q3_MN_model.predict(Vars_g))
        # print(cv)
        # print(cv.shape)
        pop.CV = np.array([cv]).T
        obj=self.models.q2_xg_model.predict(Vars_f)
        obj=obj*self.data_dict2.np_std_train_std[-1]+self.data_dict2.np_std_train_mean[-1]
        pop.ObjV = np.array([obj]).T
        # pop.CV = np.hstack((np.array([cv,obj]).T,pop.ObjV))
        

