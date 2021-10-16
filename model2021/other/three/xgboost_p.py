import pandas as pd
import numpy as np

pps=pd.DataFrame()


# data_dict = DataDict('CYP3A4')
max_depth        = 77
n_estimator     = 888
min_child_weight = 10.420648272126282
learning_rate    = 0.36590576171875
subsample        = 0.646240234375
colsample_bytree = 0.4742431640625
gamma            = 2.6152901279072873
reg_alpha        = 1.3111164517076557
reg_lambda       = 5.986122078407587
params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
              'n_estimator': n_estimator, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
              'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,'分类目标':'CYP3A4'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


# data_dict = DataDict('Caco-2')
max_depth        =45
n_estimator     =903
min_child_weight =4.946536813876396
learning_rate    =0.86822509765625
subsample        =0.627197265625
colsample_bytree =0.251220703125
gamma            =0.3749861716696612
reg_alpha        =5.78501047138394
reg_lambda       =18.628458513101627

params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
              'n_estimator': n_estimator, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
              'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,'分类目标':'Caco-2'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

# data_dict = DataDict('hERG')
max_depth        =72
n_estimator     =814
min_child_weight =1.7938300851062206
learning_rate    =0.8089599609375
subsample        =0.3648681640625
colsample_bytree =0.7520751953125
gamma            =0.2176674563120129
reg_alpha        =6.5381871726500425
reg_lambda       =10.75565626394754

params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
              'n_estimator': n_estimator, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
              'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,'分类目标':'hERG'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


# data_dict = DataDict('MN')    
max_depth        =13
n_estimator     =901
min_child_weight =0.450364877185353
learning_rate    =0.26324462890625
subsample        =0.90228271484375
colsample_bytree =0.81109619140625
gamma            =0.2780161972663775
reg_alpha        =0.40718233941016924
reg_lambda       =18.638224175354672

params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
              'n_estimator': n_estimator, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
              'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,'分类目标':'MN'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

# data_dict = DataDict('HOB') 
max_depth        =35
n_estimator     =619
min_child_weight =1.9396283707747297
learning_rate    =0.5322265625
subsample        =0.93365478515625
colsample_bytree =0.81610107421875
gamma            =3.172314347512617
reg_alpha        =1.5100918201134494
reg_lambda       =2.9509084736193603

params = {'max_depth': max_depth, 'min_child_weight': min_child_weight, 'learning_rate': learning_rate,
              'n_estimator': n_estimator, 'subsample': subsample, 'colsample_bytree': colsample_bytree, 'gamma': gamma,
              'reg_alpha': reg_alpha, 'reg_lambda': reg_lambda,'分类目标':'HOB'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

pps.to_excel('p_xgboost.xlsx',index=False)