from scipy.spatial.distance import correlation as dcor
from sklearn.inspection import permutation_importance
import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.inspection import permutation_importance

logpath = 'rf.log'

train_end = -395
vaild_end = -197

class DataDict:
    def __init__(self) -> None:
        # 模型相关
        df_all = pd.read_excel('data/clean2.xlsx')
        # 筛选变量列
        sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311,322]]
        df_all=df_all[sel_col]

        df_all=df_all.sample(frac=1)
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
        self.x_train = self.np_std_train[:, :-1]
        self.y_train = self.np_std_train[:, -1]
        self.x_valid = self.np_std_valid[:, :-1]
        self.y_valid = self.np_std_valid[:, -1]

        self.x_test = self.np_std_test[:, :-1]
        self.y_test = self.np_std_test[:, -1]
        # self.df_index=df_index
        # self.train_index=df_index[0:train_end]
        # self.valid_index=df_index[train_end:valid_end]
        # self.test_index=df_index[valid_end:]




def train(data_dict, max_leaf_nodes=10,
          n_estimators=100,
          max_depth=7,
          max_features=9,
          min_samples_leaf=60,
          min_samples_split=1200,
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
    params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
    model = RandomForestRegressor(
        **params)               # 载入模型（模型命名为model)
    model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）

    # print(np_std_train.shape)
    # print(np_std_train_std.shape)
    # 模型预测
    train_predict = model.predict(
        data_dict.x_train)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    valid_predict = model.predict(
        data_dict.x_valid)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    test_predict = model.predict(
        data_dict.x_test)*data_dict.np_std_train_std[-1]+data_dict.np_std_train_mean[-1]
    # print(train_predict)
    
    # 模型评估 mse
    train_mse = mean_squared_error(data_dict.np_train[:, -1], train_predict)
    valid_mse = mean_squared_error(data_dict.np_valid[:, -1], valid_predict)
    test_mse = mean_squared_error(data_dict.np_test[:, -1], test_predict)

    print_msg = u'\n=============================== train params ====================================\n'+str(params) +\
        '\n=============================== model evaluate ====================================\n'+f'train_loss: {train_mse:.5f},' +\
        f'valid_loss: {valid_mse:.5f},' +\
        f'test_loss: {test_mse:.5f}\n'
 
    print(print_msg)
    # with open(logpath, 'a') as f:
    #     f.write(print_msg)
    return valid_mse


if __name__ == '__main__':
    data_dict = DataDict()
    # learning_rate=0.05
    n_estimators=457
    max_depth=27
    min_samples_leaf =0.0006103515625
    min_samples_split =0.01171875
    max_features=13
    max_leaf_nodes=49
    min_weight_fraction_leaf=0.00042729825418141864
    # train(data_dict,n_estimators=n_estimators,max_depth=max_depth,min_samples_leaf=min_samples_leaf,min_samples_split=min_samples_split
    # ,max_features=max_features,max_leaf_nodes=max_leaf_nodes)

    params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
    model = RandomForestRegressor(
        **params)               # 载入模型（模型命名为model)
    model.fit(data_dict.x_train, data_dict.y_train)            # 训练模型（训练集）

    result = permutation_importance(
    model, data_dict.x_train, data_dict.y_train, n_repeats=20, random_state=42, n_jobs=2)
    df_all = pd.read_excel('data/clean2.xlsx')
    # 筛选变量列
    # sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311,322]]

    sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311]]
    zip(sel_col,result.importances_mean)
    sorted(zip(result.importances_mean,sel_col),key=lambda x:-x[0])
    dcor(df_all.drop(np.array(sel_col.columns),axis=1))


    importances=dict(zip(sel_col,result.importances_mean))





from scipy.spatial.distance import pdist, squareform
import numpy as np

def distcorrXY(X, Y):
    """ Compute the distance correlation function
    
    >>> a = [1,2,3,4,5]
    >>> b = np.array([1,2,9,4,4])
    >>> distcorr(a, b)
    0.762676242417
    """
    X = np.atleast_1d(X)
    Y = np.atleast_1d(Y)
    if np.prod(X.shape) == len(X):
        X = X[:, None]
    if np.prod(Y.shape) == len(Y):
        Y = Y[:, None]
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    n = X.shape[0]
    if Y.shape[0] != X.shape[0]:
        raise ValueError('Number of samples must match')
    a = squareform(pdist(X))
    b = squareform(pdist(Y))
    A = a - a.mean(axis=0)[None, :] - a.mean(axis=1)[:, None] + a.mean()
    B = b - b.mean(axis=0)[None, :] - b.mean(axis=1)[:, None] + b.mean()
    
    dcov2_xy = (A * B).sum()/float(n * n)
    dcov2_xx = (A * A).sum()/float(n * n)
    dcov2_yy = (B * B).sum()/float(n * n)
    dcor = np.sqrt(dcov2_xy)/np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
    return dcor

def distcorr(A):
    b=np.zeros((A.shape[1],A.shape[1]))
    for i in range(A.shape[1]):
        for j in range(A.shape[1]):
            b[i,j]=distcorrXY(A[:,i],A[:,j])
    return b

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# sns.heatmap(df)
b=distcorr(np.array(df_all[sel_col]))
a_b=np.where(b>0.8)
cor_var=[]
for i in range(len(a_b[0])):
    if a_b[1][i]>a_b[0][i]:
        print(str(a_b[0][i])+','+str(a_b[1][i]))
        cor_var.append((a_b[0][i],a_b[1][i]))

 # 筛选变量列
sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311]]
zip(sel_col,result.importances_mean)

drop_col=[]
for a,b in cor_var:
    print(importances[sel_col[a]] < importances[sel_col[b]])
    if importances[sel_col[a]] < importances[sel_col[b]]:
        drop_col.append(sel_col[a])
    else:
        drop_col.append(sel_col[b])

# [:,1].shape
sel_col2=list(set(sel_col)-set(drop_col))
b2=distcorr(np.array(df_all[sel_col2]))
b_df=pd.DataFrame(b2)
for i in range(len(sel_col2)):
    b_df.rename(columns={i:sel_col2[i]},inplace=True)
sns.heatmap(b_df)

#######################################################
# sel_col2=['ALogP','minsOH','nRotB','maxaaCH','MDEC-22','VP-1','MDEC-23','hmin','nBondsM','nF10Ring','nHBint4','nHBint6','maxsCH3','ETA_dBetaP','minHBint3','nHBint7','maxwHBa', 'nHBint10','SwHBa']