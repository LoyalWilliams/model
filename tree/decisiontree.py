from sklearn import tree
from sklearn import datasets
import numpy as np

iris=datasets.load_iris()
clf=tree.DecisionTreeClassifier(criterion="entropy")

clf.fit(iris.data,iris.target)
clf.predict([[i for i in iris.data[0]]])

import graphviz
dot_data = tree.export_graphviz(clf,out_file=None,feature_names=[u'年龄',u'收入',u'存款'],
class_names=[u'普通',u'vip'],filled=True,rotate=True)
graph=graphviz.Source(dot_data)
graph.render('mytree')