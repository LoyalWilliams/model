import numpy as np
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])
from sklearn.svm import SVC,SVR
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier
from sklearn.externals import joblib
clf = SVC()
clf.fit(X, y) 
print(clf.predict([[-0.8, -1]]))

X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=.2,random_state=0)
# 训练模型
model = OneVsRestClassifier(SVC(kernel='linear',probability=True,random_state=random_state))
print("[INFO] Successfully initialize a new model !")
print("[INFO] Training the model…… ")
clt = model.fit(X_train,y_train)
print("[INFO] Model training completed !")
# 保存训练好的模型，下次使用时直接加载就可以了
joblib.dump(clt,"F:/python/model/conv_19_80%.pkl")
print("[INFO] Model has been saved !")

y_test_pred = clt.predict(X_test)
ov_acc = metrics.accuracy_score(y_test_pred,y_test)
print("overall accuracy: %f"%(ov_acc))
print("===========================================")
acc_for_each_class = metrics.precision_score(y_test,y_test_pred,average=None)
print("acc_for_each_class:\n",acc_for_each_class)
print("===========================================")
avg_acc = np.mean(acc_for_each_class)
print("average accuracy:%f"%(avg_acc))
