from sklearn import datasets
from sklearn.cluster import KMeans

iris=datasets.load_iris()
kmeans=KMeans(n_clusters=3,random_state=0).fit(iris.data)

# 训练标签数量
kmeans.labels_
kmeans.predict([[i for i in iris.data[110]]])

# 模型评估
from sklearn import metrics
print(metrics.adjusted_mutual_info_score(iris.target,kmeans.predict(iris.data)))
print(metrics.adjusted_rand_score(iris.target,kmeans.predict(iris.data)))
print(metrics.precision_score(iris.target,kmeans.predict(iris.data),average=None))
