from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

df_all = pd.read_excel('data/std-clean.xlsx')
# 筛选变量列
# sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311,322]]
# df_all=df_all[sel_col]
df_all=df_all.drop(['pIC50'], axis=1)
estimator = PCA(n_components=30)
pca_x = estimator.fit_transform(df_all)
# estimator.explained_variance_
# pd.DataFrame(pca_x)
pd.DataFrame(pca_x).to_excel('data/pca.xlsx',index=False)