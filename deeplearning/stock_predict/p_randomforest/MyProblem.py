# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import __init__
import data_source
import p_rf
import json


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 7  # 初始化Dim（决策变量维数）

        # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        varTypes = [1, 1, 1, 1, 0, 0, 0]
        lb = [1, 0, 0, 0, 0, 0, 0]  # 决策变量下界
        ub = [50, 1000, 50, 38, 0.5, 1, 0.5]  # 决策变量上界
        lbin = [0, 0, 0, 0, 0, 0, 1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)
        self.data_dict = data_source.DataDict()

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        max_leaf_nodes = Vars[:, [0]]
        n_estimators = Vars[:, [1]]
        max_depth = Vars[:, [2]]
        max_features = Vars[:, [3]]
        min_samples_leaf = Vars[:, [4]]
        min_samples_split = Vars[:, [5]]
        min_weight_fraction_leaf = Vars[:, [6]]

        objvs = []
        models = []
        best_index = 0
        best_params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
                       'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
        with open('d:\code\py\model\deeplearning\stock_predict\params.log', 'w') as outfile:
            outfile.write(str(best_params))
            outfile.write('\n')
        for i in range(0, n_estimators.shape[0]):
            params = {'data_dict': self.data_dict, 'max_depth': int(max_depth[i, 0]),
                      'n_estimators': int(n_estimators[i, 0]),  'min_samples_leaf': min_samples_leaf[i, 0], 'min_samples_split': min_samples_split[i, 0],
                      'max_features': int(max_features[i, 0]), 'max_leaf_nodes': int(max_leaf_nodes[i, 0]), 'min_weight_fraction_leaf': min_weight_fraction_leaf[i, 0]}
            print(params)
            objv = p_rf.train(**params)
            objvs.append(objv)
        print(Vars[0, :])

        pop.ObjV = np.array([objvs]).T
