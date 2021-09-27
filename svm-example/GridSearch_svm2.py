
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
import joblib

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
# x_vals = np.array([[x[1],x[2]] for x in iris.data])
x_vals = np.array([[x[1],x[2],x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])


x_train, x_validation, y_train, y_validation = train_test_split(x_vals, y_vals, test_size=.2,random_state=0)


C = 1.0  # SVM regularization parameter
gamma=0.7

clf = svm.SVR(kernel='rbf', gamma=gamma, C=C)
models = clf.fit(x_train, y_train) 
pre=models.predict(x_validation)

err=np.sum((pre-y_validation)**2)/pre.shape[0]
print(err)
# print(pre.shape[0])
# 网格搜索法
Cs=np.arange(0.0001,10,0.1)
gammas=np.arange(0.0001,10,0.1)
cg=np.matrix([[1,2,999]])
for C in Cs:
    for gamma in gammas:
        clf = svm.SVR(kernel='rbf', gamma=gamma, C=C)
        models = clf.fit(x_train, y_train) 
        pre=models.predict(x_validation)

        err=np.sum((pre-y_validation)**2)/pre.shape[0]
        cg=np.r_[cg,[[C,gamma,err]]]
print(cg.shape)
cg[cg[:,2].argmin()]
# matrix([[1.6001    , 0.4001    , 0.11568699]])
# 网格搜索法
Cs=np.arange(1,2,0.01)
gammas=np.arange(0.0001,1,0.01)
cg=np.matrix([[1,2,999]])

for C in Cs:
    for gamma in gammas:
        clf = svm.SVR(kernel='rbf', gamma=gamma, C=C)
        models = clf.fit(x_train, y_train) 
        pre=models.predict(x_validation)

        err=np.sum((pre-y_validation)**2)/pre.shape[0]
        cg=np.r_[cg,[[C,gamma,err]]]

cg[cg[:,2].argmin()]
# matrix([[1.62      , 0.3801    , 0.11554415]])
