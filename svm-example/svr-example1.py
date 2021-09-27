
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split


# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
# x_vals = np.array([[x[1],x[2]] for x in iris.data])
x_vals = np.array([[x[1],x[2],x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])


x_train, x_validation, y_train, y_validation = train_test_split(x_vals, y_vals, test_size=.2,random_state=0)

# print(y_validation)
print(y_train)
C = 1.0  # SVM regularization parameter
gamma=0.7

clf = svm.SVR(kernel='rbf', gamma=gamma, C=C)
models = clf.fit(x_train, y_train) 
pre=models.predict(x_validation)

err=np.sum((pre-y_validation)**2/pre.shape)
print(err)



