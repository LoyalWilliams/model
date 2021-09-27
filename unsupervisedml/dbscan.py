from sklearn import datasets
from sklearn.cluster import DBSCAN


iris=datasets.load_iris()
dbscan=DBSCAN(eps=0.2,min_samples=5)
dbscan.fit(iris.data)

dbscan.labels_
# 不准