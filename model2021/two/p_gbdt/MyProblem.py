# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import numpy as np
from sklearn.model_selection import train_test_split
import p_gbdt

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 9  # 初始化Dim（决策变量维数）

        varTypes = [1,1,1,1,0,0,0,0,0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [1,10,0,0,0,0,0,0,0]  # 决策变量下界
        ub = [50, 1000, 50, 19, 1,0.5,1,1,0.5]  # 决策变量上界
        lbin = [0,0,0,0,0,0,0,0,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.data_dict = p_gbdt.DataDict()
    

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        max_leaf_nodes            =Vars[:, [0]]
        n_estimators              =Vars[:, [1]]
        max_depth                 =Vars[:, [2]]
        max_features              =Vars[:, [3]]
        learning_rate             =Vars[:, [4]]
        min_samples_leaf          =Vars[:, [5]]
        min_samples_split         =Vars[:, [6]]
        subsample                 =Vars[:, [7]]
        min_weight_fraction_leaf  =Vars[:, [8]]

        objvs=[]
        for i in range(0,n_estimators.shape[0]):
            params = {'data_dict':self.data_dict,'max_depth': int(max_depth[i,0]), 'learning_rate': learning_rate[i,0], 
              'n_estimators': int(n_estimators[i,0]), 'subsample': subsample[i,0], 'min_samples_leaf': min_samples_leaf[i,0], 'min_samples_split': min_samples_split[i,0],
              'max_features': int(max_features[i,0]), 'max_leaf_nodes': int(max_leaf_nodes[i,0]), 'min_weight_fraction_leaf': min_weight_fraction_leaf[i,0]}
            objv=p_gbdt.train(**params)
            objvs.append(objv)

        pop.ObjV=np.array([objvs]).T

