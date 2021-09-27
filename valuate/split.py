
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()
x_vals = np.array([[x[1],x[2],x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])


x_train, x_validation, y_train, y_validation = train_test_split(x_vals, y_vals, test_size=.2,random_state=0)