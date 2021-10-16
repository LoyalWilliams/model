import pandas as pd
import numpy as np

pps=pd.DataFrame()


max_leaf_nodes           =20
n_estimators             =976
max_depth                =24
max_features             =124
learning_rate            =0.49346923828125
min_samples_leaf         =0.050537109375
min_samples_split        =0.46014404296875
subsample                =0.8082275390625
min_weight_fraction_leaf =0.04755219142961788

params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
              '分类目标':'CYP3A4'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)



max_leaf_nodes           =18
n_estimators             =587
max_depth                =7
max_features             =192
learning_rate            =0.4119873046875
min_samples_leaf         =0.102783203125
min_samples_split        =0.75750732421875
subsample                =0.934326171875
min_weight_fraction_leaf =0.13032596752533268

params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
              '分类目标':'MN'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


max_leaf_nodes           =25
n_estimators             =927
max_depth                =34
max_features             =158
learning_rate            =0.62066650390625
min_samples_leaf         =0.11676025390625
min_samples_split        =0.18145751953125
subsample                =0.88775634765625
min_weight_fraction_leaf =0.11256256867293371

params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
              '分类目标':'Caco-2'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)



max_leaf_nodes           =15
n_estimators             =666
max_depth                =33
max_features             =296
learning_rate            =0.42877197265625
min_samples_leaf         =0.0740966796875
min_samples_split        =0.16741943359375
subsample                =0.95684814453125
min_weight_fraction_leaf =0.15980954706385056

params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
              '分类目标':'hERG'}
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

max_leaf_nodes           =7
n_estimators             =796
max_depth                =17
max_features             =240
learning_rate            =0.56414794921875
min_samples_leaf         =0.0030517578125
min_samples_split        =0.14141845703125
subsample                =0.71466064453125
min_weight_fraction_leaf =0.03259675253326822

params = {'max_depth': max_depth, 'learning_rate': learning_rate, 'learning_rate': learning_rate,
              'n_estimators': n_estimators, 'subsample': subsample, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
              'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
              '分类目标':'HOB'}
 

pps=pps.append(pd.DataFrame([params]),ignore_index=True)

pps.to_excel('p_gbdt.xlsx',index=False)