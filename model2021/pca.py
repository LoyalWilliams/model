from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

df_all = pd.read_excel('data/std-clean.xlsx')
# 筛选变量列
# sel_col=df_all.columns[[0, 1, 43, 79, 92, 94, 95, 97, 98, 116, 154, 176, 188, 200, 202, 205, 214, 247, 250, 284, 285, 302, 308, 311,322]]
# df_all=df_all[sel_col]
df_all=df_all.drop(['pIC50'], axis=1)
estimator = PCA(n_components=29)
pca_x = estimator.fit_transform(df_all)
# estimator.explained_variance_
# pd.DataFrame(pca_x)
df_pca=pd.DataFrame(pca_x)
# df_pca.to_excel('data/pca.xlsx',index=False)

# 整合问题三的数据,智能利用训练集的信息
df_ADMET = pd.read_excel('data/ADMET.xlsx', sheet_name='training')
df_all=pd.concat([df_pca, df_ADMET], axis=1)
# 打乱顺序，保存
# df_all.sample(frac=1).to_excel('data/q3_data.xlsx',index=False)
