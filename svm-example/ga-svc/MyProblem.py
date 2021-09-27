# -*- coding: utf-8 -*-
"""MyProblem.py"""
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
import joblib

class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self):
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [-1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = 2  # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = [0, 0]  # 决策变量下界
        ub = [10, 10]  # 决策变量上界
        lbin = [0, 0]  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1, 1]  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # Load the data
        # iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
        iris = datasets.load_iris()

        x_vals = np.array(iris.data[:, :2])
        y_vals = np.array(iris.target)

        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(x_vals, y_vals, test_size=.2,random_state=0)


    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        Cs = Vars[:, [0]]
        gammas = Vars[:, [1]]
        objvs=[]
        models=[]
        best_index=0
        for i in range(0,Cs.shape[0]):
            clf = svm.SVC(kernel='linear', C=Cs[i],decision_function_shape='ovo')
            # clf = svm.SVC(kernel='rbf', gamma=gammas[i], C=Cs[i],decision_function_shape='ovo')
            model = clf.fit(self.x_train, self.y_train) 
            pre=model.predict(self.x_validation)
            objvs.append(np.sum(pre==self.y_validation)/pre.shape[0])
            models.append(model)
        tmp=np.array(objvs)
        print(np.min(objvs))

        joblib.dump(models[tmp.argmin()],"model.pkl")
        pop.ObjV=np.array([objvs]).T
        # print(err)
