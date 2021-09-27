from sklearn import datasets
from sklearn.cluster import Birch


iris=datasets.load_iris()
birch=Birch(n_clusters=3)
birch.fit(iris.data)

birch.labels_
birch.predict([[i for i in iris.data[110]]])

# 模型评估
from sklearn import metrics
print(metrics.adjusted_mutual_info_score(iris.target,birch.predict(iris.data)))
print(metrics.adjusted_rand_score(iris.target,birch.predict(iris.data)))