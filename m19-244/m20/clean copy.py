#问题一数据预处理 Python 代码
import pandas as pd
import numpy as np

df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\附件一：325个样本数据.xlsx',header=None)
df=df.drop([0],axis=1)
for i in range(16,df.shape[1]):
    if pd.isnull(df[i][2]) or df[i][2]==0 :
        df.rename(columns={i:df[i][1]},inplace=True)
    else:
        df.rename(columns={i:df[i][2]},inplace=True)
df.rename(columns={1:'时间'},inplace=True)
for i in range(2,9):
    df.rename(columns={i:'原料:'+df[i][2]},inplace=True)
for i in range(9,12):
    df.rename(columns={i:'产品:'+df[i][2]},inplace=True)
for i in range(12,14):
    df.rename(columns={i:'待生:'+df[i][2]},inplace=True)
for i in range(14,16):
    df.rename(columns={i:'再生:'+df[i][2]},inplace=True)

df=df.dropna(axis=1, how='all')
df=df.dropna(axis=1, how='all')
df=df.drop([0,1,2])

# df.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\clean1.xlsx',index=False)


# 3σ准则
df1=df.drop(['时间'],axis=1).values
std1=df.std().values
std2=std1.reshape(1,std1.shape[0])
mean1=df.mean().values
mean2=mean1.reshape(1,mean1.shape[0])
ones1=np.ones((df.shape[0],1))

boolmat=abs(df1-ones1.dot(mean2))<3*ones1.dot(std2)


a=boolmat.T[0]
boolmat.T[1]&boolmat.T[1]
for i in range(1,14):
    a=a&boolmat.T[i]
# 删除不符合条件的数据
df=df.drop(np.where(a==False)[0])

df.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\clean1.xlsx',index=False)


# ###################
# dfen1=dfen.drop(['时间'],axis=1).values
# std1=dfen.std().values
# std2=std1.reshape(1,std1.shape[0])
# mean1=dfen.mean().values
# mean2=mean1.reshape(1,mean1.shape[0])
# ones1=np.ones((dfen.shape[0],1))

# boolmat=abs(dfen1-ones1.dot(mean2))<3*ones1.dot(std2)


# a=boolmat.T[0]
# boolmat.T[1]&boolmat.T[1]
# for i in range(1,14):
#     a=a&boolmat.T[i]
# # 删除不符合条件的数据
# dfen2=dfen.drop(np.where(a==False)[0])



###########################################
# 取英文名
df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\附件一：325个样本数据.xlsx',header=None)
df=df.drop([0],axis=1)
for i in range(16,df.shape[1]):
    if pd.isnull(df[i][1]) or df[i][1]==0 :
        df.rename(columns={i:df[i][1]},inplace=True)
    else:
        df.rename(columns={i:df[i][1]},inplace=True)
df.rename(columns={1:'时间'},inplace=True)
for i in range(2,9):
    df.rename(columns={i:'原料:'+df[i][2]},inplace=True)
for i in range(9,12):
    df.rename(columns={i:'产品:'+df[i][2]},inplace=True)
for i in range(12,14):
    df.rename(columns={i:'待生:'+df[i][2]},inplace=True)
for i in range(14,16):
    df.rename(columns={i:'再生:'+df[i][2]},inplace=True)

df=df.drop([0,1,2])

df.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\cen.xlsx',index=False)

##########################







des=df.describe()
des['one']=1
sele=pd.DataFrame(columns=['one'])
for cn in des.columns:
    if des[cn]['min']==0:
        print(des[cn])
sele=sele.drop(['one'],axis=1)
# sele.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\clean2.xlsx',index=True)



dfen=pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\cen.xlsx')
des=dfen.describe()
des['one']=1
sele=pd.DataFrame(columns=['one'])
for cn in des.columns:
    if des[cn]['min']==0 and des[cn]['25%']==0:
        sele[cn]=des[cn]
# sele.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\clean2.xlsx',index=True)

pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\clean1.xlsx')

sele=sele.drop(['one'],axis=1)
delc=sele.columns.copy()
# delc.append('S-ZORB.AT_5201.PV')
dfen=dfen.drop(['S-ZORB.AT_5201.PV'],axis=1)
dfen.drop([370],axis=1)
# dfen.drop(['one'],axis=1)
for cn in delc:
    dfen=dfen.drop([cn],axis=1)
dfen=dfen.drop([370],axis=1)
dfen.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\last-clean.xlsx',index=False)


#####################
# 小于0的
sele=pd.DataFrame(columns=['one'])
for cn in des.columns:
    if des[cn]['min']<=0 :
        sele[cn]=des[cn]
sele.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\le0.xlsx')
