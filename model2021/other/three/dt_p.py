import pandas as pd
import numpy as np


pps=pd.DataFrame()

max_leaf_nodes           =19
max_depth                =27
max_features             =110
min_samples_leaf         =0.00555419921875
min_samples_split        =0.05859375
min_weight_fraction_leaf =0.016115248443413503
params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 
            'min_weight_fraction_leaf': min_weight_fraction_leaf,'分类目标':'Caco-2'}
            
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


max_leaf_nodes           =29
max_depth                =41
max_features             =7
min_samples_leaf         =0.23931884765625
min_samples_split        =0.54248046875
min_weight_fraction_leaf =0.2405689171041387
params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 
            'min_weight_fraction_leaf': min_weight_fraction_leaf,'分类目标':'CYP3A4'}
            
pps=pps.append(pd.DataFrame([params]),ignore_index=True)



max_leaf_nodes           =28
max_depth                =12
max_features             =169
min_samples_leaf         =0.064453125
min_samples_split        =0.57879638671875
min_weight_fraction_leaf =0.06525454767427664
params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 
            'min_weight_fraction_leaf': min_weight_fraction_leaf,'分类目标':'HOB'}
            
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


max_leaf_nodes           =34
max_depth                =5
max_features             =241
min_samples_leaf         =0.1278076171875
min_samples_split        =0.93157958984375
min_weight_fraction_leaf =0.26022463679648394
params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 
            'min_weight_fraction_leaf': min_weight_fraction_leaf,'分类目标':'hERG'}
            
pps=pps.append(pd.DataFrame([params]),ignore_index=True)


max_leaf_nodes           =8
max_depth                =25
max_features             =75
min_samples_leaf         =0.01220703125
min_samples_split        =0.03167724609375
min_weight_fraction_leaf =0.017824441460139177

params = {'max_depth': max_depth, 'min_samples_leaf': min_samples_leaf, 'min_samples_split': min_samples_split,
            'max_features': max_features, 'max_leaf_nodes': max_leaf_nodes, 
            'min_weight_fraction_leaf': min_weight_fraction_leaf,'分类目标':'MN'}
            
pps=pps.append(pd.DataFrame([params]),ignore_index=True)

pps.to_excel('p_dt.xlsx',index=False)