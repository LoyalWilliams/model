from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
import numpy as np
import matplotlib.pyplot as plt


data = np.array([[ 1., -1.,  2.],
                    [ 2.,  0.,  0.],
                    [ 0.,  1., -1.]])
ss = StandardScaler()
# ss = MinMaxScaler()
std_data = ss.fit_transform(data)

origin_data = ss.inverse_transform(std_data)
print('data is ',data)
print('after standard ',std_data)
print('after inverse ',origin_data)
print('after standard mean and std is ',np.mean(std_data), np.std(std_data))

# 标准化等价于以下式子
(data-ss.mean_)/np.sqrt(np.sum((data-ss.mean_)**2/3,axis=0))


