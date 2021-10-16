import pandas as pd
import numpy as np

pps=pd.DataFrame()


# data_dict = DataDict('MN')
max_leaf_nodes            = 13
n_estimators              = 404
max_depth                 = 6
max_features              = 120
min_samples_leaf          = 0.12274169921875
min_samples_split         = 0.1075439453125
min_weight_fraction_leaf  = 0.06122573556342327

params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
             '分类目标':'MN'}
             
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


# data_dict = DataDict('HOB')
max_leaf_nodes            = 29
n_estimators              = 391
max_depth                 = 43
max_features              = 228
min_samples_leaf          = 0.00537109375
min_samples_split         = 0.00927734375
min_weight_fraction_leaf  = 0.003418386033451349

params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
             '分类目标':'HOB'}
             
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

# data_dict = DataDict('hERG')
max_leaf_nodes            = 23
n_estimators              = 26
max_depth                 = 11
max_features              = 134
min_samples_leaf          = 0.0849609375
min_samples_split         = 0.2476806640625
min_weight_fraction_leaf  = 0.16225125137345867

params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
             '分类目标':'hERG'}
             
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

# data_dict = DataDict('Caco-2')
max_leaf_nodes            = 29
n_estimators              = 307
max_depth                 = 19
max_features              = 308
min_samples_leaf          = 0.23931884765625
min_samples_split         = 0.62603759765625
min_weight_fraction_leaf  = 0.495177633988524

params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
             '分类目标':'Caco-2'}
             
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

# data_dict = DataDict('CYP3A4')
max_leaf_nodes            = 13
n_estimators              = 466
max_depth                 = 9
max_features              = 238
min_samples_leaf          = 0.0408935546875
min_samples_split         = 0.036865234375
min_weight_fraction_leaf  = 0.09492125503601515

params = {'max_depth': max_depth, 'n_estimators': n_estimators, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 'min_weight_fraction_leaf': min_weight_fraction_leaf,
             '分类目标':'CYP3A4'}

pps=pps.append(pd.DataFrame([params]),ignore_index=True)

pps.to_excel('p_rf.xlsx',index=False)