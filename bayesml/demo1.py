from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

iris=datasets.load_iris()
gnb=GaussianNB()
gnb.fit(iris.data,iris.target)

# 先验概率
gnb.class_prior_
# 训练标签数量
gnb.class_count_
# 查看参数
gnb.theta_
gnb.sigma_
# gnb.predict()#预测

# 多项式和伯努利贝叶斯
# from sklearn.naive_bayes import BernoulliNB,MultinomialNB
# BernoulliNB()