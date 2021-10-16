import pandas as pd
import numpy as np


#dt

max_leaf_nodes = 27
max_depth = 45
max_features = 16
min_samples_leaf = 0.032958984375
min_samples_split = 0.0093994140625
min_weight_fraction_leaf = 0.033390306433890855
params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}

pd.DataFrame([params]).to_excel('p_dt2.xlsx',index=False)


#svm

kernel='linear'
degree=1
C     =79.19960021972656
gamma =93.1406021118164
coef0 =81.6318336790406
params = {'kernel': kernel, 'C': C,
            'degree': degree, 'gamma': gamma, 'coef0': coef0}
pd.DataFrame([params]).to_excel('p_svm2.xlsx',index=False)

#rf
max_leaf_nodes = 50
n_estimators = 529
max_depth = 25
max_features = 9
min_samples_leaf = 0.00140380859375
min_samples_split = 0.00445556640625
min_weight_fraction_leaf = 0.004150897326333781

params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
pd.DataFrame([params]).to_excel('p_rf2.xlsx',index=False)


# gbdt

max_leaf_nodes = 11
n_estimators = 872
max_depth = 41
max_features = 14
learning_rate = 0.23187255859375
min_samples_leaf = 0.15362548828125
min_samples_split = 0.23052978515625
subsample = 0.93115234375
min_weight_fraction_leaf = 0.1491881333170553
params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
            'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf}
pd.DataFrame([params]).to_excel('p_gbdt2.xlsx',index=False)

# xgboost


max_depth = 93
n_estimator = 560
min_child_weight = 7.756072067535658
learning_rate = 0.223876953125
subsample = 0.950927734375
colsample_bytree = 0.86456298828125
gamma = 1.1154217354649942
reg_alpha = 1.6801516729418675
reg_lambda= 15.323544782809382

params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
            'n_estimator': n_estimator, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
            'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda}

pd.DataFrame([params]).to_excel('p_xgboost2.xlsx',index=False)
