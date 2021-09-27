from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris=load_iris()
clf=RandomForestClassifier(n_estimators=20,bootstrap=True,
oob_score=True)
clf.fit(iris.data,iris.target)
clf.predict([[i for i in iris.data[0]]])
clf.predict([[i for i in iris.data[110]]])