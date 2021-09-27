from sklearn import datasets
from sklearn.mixture import GaussianMixture


iris=datasets.load_iris()
gmm=GaussianMixture(n_components=3,covariance_type="full",max_iter=100,random_state=0)
# gmm=GaussianMixture(n_components=3,covariance_type="diag",max_iter=100,random_state=0)
gmm.fit(iris.data)

gmm.n_iter_
gmm.predict([[i for i in iris.data[110]]])

# 模型评估
from sklearn import metrics
print(metrics.adjusted_mutual_info_score(iris.target,gmm.predict(iris.data)))
print(metrics.adjusted_rand_score(iris.target,gmm.predict(iris.data)))