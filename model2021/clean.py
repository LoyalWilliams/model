#问题一数据预处理 Python 代码
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df1 = pd.read_excel('data/ERα_activity.xlsx', sheet_name='training')
df2 = pd.read_excel('data/Molecular_Descriptor.xlsx', sheet_name='training')
df3 = df1.drop(['SMILES', 'IC50_nM'], axis=1)
df4=pd.concat([df2, df3], axis=1)

df_all=df4.drop(['SMILES'], axis=1)


##########################
####### 删除数据缺失超过75%的变量
des=df4.describe()
sele=pd.DataFrame()
for cn in des.columns:
    if (des[cn]['min']==0 and des[cn]['75%']==0) or des[cn]['std']==0:
        sele[cn]=des[cn]

df_all=df4.drop(np.array(sele.columns),axis=1)
df_all.to_excel('data/clean.xlsx',index=False)

df_all=df_all.drop(['SMILES'], axis=1)
df_all.to_excel('data/clean2.xlsx',index=False)
# 数据标准化
std_data=(df_all - np.mean(df_all)) / np.std(df_all)
std_data.to_excel('data/std-clean.xlsx',index=False)

# 第2题数据
df_all = pd.read_excel('data/clean2.xlsx')
df_all.sample(frac=1).to_excel('data/q2_data.xlsx',index=False)

# 整合问题三的数据
df_clean = pd.read_excel('data/clean2.xlsx')
df_ADMET = pd.read_excel('data/ADMET.xlsx', sheet_name='training')
df_all=pd.concat([df_clean, df_ADMET], axis=1)
# 打乱顺序，保存
df_all.sample(frac=1).to_excel('data/q3_data.xlsx',index=False)



df1_test = pd.read_excel('data/ERα_activity.xlsx', sheet_name='test')
df2_test = pd.read_excel('data/Molecular_Descriptor.xlsx', sheet_name='test')
df_ADMET_test = pd.read_excel('data/ADMET.xlsx', sheet_name='test')



# 第2题数据
df_all = pd.read_excel('data/clean2.xlsx')
df_all.sample(frac=1).to_excel('data/q2_data.xlsx',index=False)

# 整合问题4的数据
df_clean = pd.read_excel('data/clean2.xlsx').drop(['pIC50'], axis=1)
df_ADMET = pd.read_excel('data/ADMET.xlsx', sheet_name='training')
df_clean['y']=df_ADMET['Caco-2']+df_ADMET['CYP3A4']+df_ADMET['hERG']+df_ADMET['HOB']+df_ADMET['MN']
df_clean.to_excel('data/q4_data.xlsx',index=False)

