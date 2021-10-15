# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import numpy as np
import p_svr


class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self,obv):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 5  # 初始化Dim（决策变量维数）

        # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        varTypes = [1, 1, 0, 0, 0]
        lb = [0, 0, 0, 0, 0]  # 决策变量下界
        ub = [2, 10, 100, 100, 100]  # 决策变量上界
        lbin = [1, 1, 0, 0, 1]   # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1]*Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）

        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim,
                            varTypes, lb, ub, lbin, ubin)
        self.data_dict = p_svr.DataDict(obv)
        # poly去掉
        self.kernel = ['sigmoid', 'linear', 'rbf']

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        kernels = Vars[:, [0]]
        degrees = Vars[:, [1]]
        Cs = Vars[:, [2]]
        gammas = Vars[:, [3]]
        coef0s = Vars[:, [4]]
        objvs = []
        models = []
        best_index = 0

        best_params = {'kernel': kernels, 'C': Cs,
                       'degree': degrees, 'gamma': gammas, 'coef0': coef0s}
        for i in range(0, kernels.shape[0]):
            # print('############')
            # print(n_estimatorss[i])
            params = {'data_dict':self.data_dict,'kernel': self.kernel[int(kernels[i, 0])], 'C': Cs[i, 0],
                      'degree': degrees[i, 0], 'gamma': gammas[i, 0], 'coef0': coef0s[i, 0]}

            objv = p_svr.train(**params)
            objvs.append(objv)

        pop.ObjV = np.array([objvs]).T
