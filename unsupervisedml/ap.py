from sklearn import datasets
from sklearn.cluster import AffinityPropagation

iris=datasets.load_iris()
ap=AffinityPropagation(preference=-8).fit(iris.data)

# 训练标签数量
ap.labels_
ap.predict([[i for i in iris.data[110]]])

# 模型评估
from sklearn import metrics
print(metrics.adjusted_mutual_info_score(iris.target,ap.predict(iris.data)))
print(metrics.adjusted_rand_score(iris.target,ap.predict(iris.data)))