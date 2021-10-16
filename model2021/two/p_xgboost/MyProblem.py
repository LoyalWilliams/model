# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import numpy as np
from sklearn.model_selection import train_test_split
import p_xgboost

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 9  # 初始化Dim（决策变量维数）

        varTypes = [1,1,0,0,0,0,0,0,0]  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0, 10, 0, 0, 0, 0,0,0,0]  # 决策变量下界
        ub = [100, 1000, 20, 1, 1, 1,20,20,20]  # 决策变量上界
        lbin = [1,0,1,0,0,0,1,1,1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        self.data_dict = p_xgboost.DataDict()
    

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        # x1 = Vars[:, [0]]
        # x2 = Vars[:, [1]]
        # x3 = Vars[:, [2]]
        # pop.ObjV = 4 * x1 + 2 * x2 + x3  # 计算目标函数值，赋值给pop种群对象的ObjV属性
        # 采用可行性法则处理约束
        # pop.CV = np.hstack([2 * x1 + x2 - 1,
        #                     x1 + 2 * x3 - 2,
        #                     np.abs(x1 + x2 + x3 - 1)])
        max_depths        =Vars[:, [0]]
        n_estimatorss =Vars[:, [1]]
        min_child_weights    =Vars[:, [2]]
        learning_rates     =Vars[:, [3]]
        subsamples        =Vars[:, [4]]
        colsample_bytrees =Vars[:, [5]]
        gammas            =Vars[:, [6]]
        reg_alphas        =Vars[:, [7]]
        reg_lambdas       =Vars[:, [8]]
        objvs=[]
        for i in range(0,max_depths.shape[0]):
            # print('############')
            # print(n_estimatorss[i])
            objv=p_xgboost.train(self.data_dict, max_depth=int(max_depths[i,0]),
                min_child_weight=min_child_weights[i,0],
                learning_rate=learning_rates[i,0],
                n_estimators=int(n_estimatorss[i,0]),
                subsample=subsamples[i,0],
                colsample_bytree=colsample_bytrees[i,0],
                gamma=gammas[i,0],
                reg_alpha=reg_alphas[i,0],
                reg_lambda=reg_lambdas[i,0])
            objvs.append(objv)
        print(Vars[0,:])

        pop.ObjV=np.array([objvs]).T

    
