from sklearn.preprocessing import StandardScaler  # 归一化
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
import numpy as np

# Load the data
# iris.data = [(Sepal Length, Sepal Width, Petal Length, Petal Width)]
iris = datasets.load_iris()
x_vals = np.array([[x[1],x[2],x[3]] for x in iris.data])
y_vals = np.array([y[0] for y in iris.data])


x_train, x_validation, y_train, y_validation = train_test_split(x_vals, y_vals, test_size=.2,random_state=0)

# 归一化操作
scaler = StandardScaler()
x_train_data = scaler.fit_transform(x_train)
x_test_data = scaler.transform(x_validation)

from sklearn.model_selection import GridSearchCV  # 在sklearn中主要是使用GridSearchCV调参

svc_model = svm.SVR(kernel='rbf')
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}  # param_grid:我们要调参数的列表(带有参数名称作为键的字典)，此处共有14种超参数的组合来进行网格搜索，进而选择一个拟合分数最好的超平面系数。
grid_search = GridSearchCV(svc_model, param_grid, n_jobs=8, verbose=1)  # n_jobs:并行数，int类型。(-1：跟CPU核数一致；1:默认值)；verbose:日志冗长度。默认为0：不输出训练过程；1：偶尔输出；>1：对每个子模型都输出。
grid_search.fit(x_train_data, y_train.ravel())  # 训练，默认使用5折交叉验证
best_parameters = grid_search.best_estimator_.get_params()  # 获取最佳模型中的最佳参数
print("cv results are" % grid_search.best_params_, grid_search.cv_results_)  # grid_search.cv_results_:给出不同参数情况下的评价结果。
print("best parameters are" % grid_search.best_params_, grid_search.best_params_)  # grid_search.best_params_:已取得最佳结果的参数的组合；
print("best score are" % grid_search.best_params_, grid_search.best_score_)  # grid_search.best_score_:优化过程期间观察到的最好的评分。
# for para, val in list(best_parameters.items()):
#     print(para, val)
svm_model = svm.SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'])  # 最佳模型