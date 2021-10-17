
import numpy as np
import q2q3model
import pandas as pd



sel_col4=['minsOH','hmin','VP-1','ETA_dBetaP','nRotB']


data_dict2 = q2q3model.DataDict2()
data_dict3 = q2q3model.DataDict3('HOB')
models = q2q3model.Models()
all_col=list(data_dict3.df_all.columns)

df_all=data_dict3.df_all

# 2目标列索引
sel_col2=data_dict2.sel_col2.copy()
sel_col2.pop()
f_col= [ list(df_all.columns).index(i) for i in sel_col2]




# data_dict2 = q2q3model.DataDict2()
# data_dict3 = q2q3model.DataDict3('HOB')
# models = q2q3model.Models()

# df_all=data_dict3.df_all

# # 目标列索引
# sel_col2=data_dict2.sel_col2.copy()
# sel_col2.pop()
# f_col= [ list(df_all.columns).index(i) for i in sel_col2]

# desc=df_all.describe()
# desc.loc['max']-desc.loc['min']
# up_df=desc.loc['max']
# down_df=desc.loc['min']-(desc.loc['max']-desc.loc['min'])*0.15
# # 标准化
# Vars_g=(Vars-data_dict3.np_std_train_mean)/data_dict3.np_std_train_std
# Vars_f=(Vars[:,f_col]-data_dict2.np_std_train_mean[:-1])/data_dict2.np_std_train_std[:-1]


# # 采用可行性法则处理约束
# # print(type(Vars_g))
# # print(Vars_g)
# cv=3-(models.q3_CYP3A4_model.predict(Vars_g)+models.q3_Caco_model.predict(Vars_g)+\
# models.q3_hERG_model.predict(Vars_g)+models.q3_HOB_model.predict(Vars_g)+models.q3_MN_model.predict(Vars_g))





def select(df_all):
    df_all=df_all.copy()
    np_all=np.array(df_all)
    # 标准化
    np_std_all=(np_all-data_dict3.np_std_train_mean)/data_dict3.np_std_train_std
    
    # 约束>=3
    cv=3-(models.q3_CYP3A4_model.predict(np_std_all)+models.q3_Caco_model.predict(np_std_all)+\
models.q3_hERG_model.predict(np_std_all)+models.q3_HOB_model.predict(np_std_all)+models.q3_MN_model.predict(np_std_all))
    df_all['cv']=cv
    sel_df=df_all[df_all['cv']<=0]
    sel_df=df_all.drop(['cv'], axis=1)
    # 标准化
    sel_std_df=(sel_df[sel_col2]-data_dict2.np_std_train_mean[:-1])/data_dict2.np_std_train_std[:-1]

    # f活性值
    sel_df['f']=models.q2_xg_model.predict(np.array(sel_std_df))
    percent25=np.percentile(np.array(sel_df['f']), 5)
    sel_df=sel_df[sel_df['f']<percent25]
    return sel_df.drop(['f'], axis=1)

# def estimate(sel_df):
#     pd.DataFrame().mean

def generat():
    # 
    rand_data=np.random.rand(100000,5)
    desc=df_all.describe()[sel_col4]
    rand_data=np.array(desc.loc['min'])+rand_data*np.array((desc.loc['max']-desc.loc['min']))

    ga_data=pd.read_csv('data/ga_data.csv',header=None)
    ga_dict=dict(zip(all_col,np.array(ga_data)[0,:]))
    gen_data=pd.DataFrame(columns=all_col)
    for  i in range(len(sel_col4)):
        gen_data[sel_col4[i]]=rand_data[:,i]
    for col in all_col:
        if col in sel_col4:
            continue
        gen_data[col]=ga_dict[col]
    return gen_data

res=select(generat() )
res[sel_col4]