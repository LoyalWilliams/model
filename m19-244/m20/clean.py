#问题一数据预处理 Python 代码
import pandas as pd
import numpy as np


###########################################
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
print(df.columns)
# df=df.drop([369],axis=1)
df=df.drop([370],axis=1)

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

df.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\cen.xlsx',index=False)

##########################
####### 删除数据缺失超过25%的变量
dfen=pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\cen.xlsx')
des=dfen.describe()
des['one']=1
sele=pd.DataFrame(columns=['one'])
for cn in des.columns:
    if des[cn]['min']==0 and des[cn]['25%']==0:
        sele[cn]=des[cn]

sele=sele.drop(['one'],axis=1)
delc=sele.columns.copy()
## S-ZORB.AT_5201.PV 数据异常
# delc.append('S-ZORB.AT_5201.PV')
dfen=dfen.drop(['S-ZORB.AT_5201.PV'],axis=1)
for cn in delc:
    dfen=dfen.drop([cn],axis=1)
dfen.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\last-clean.xlsx',index=False)

