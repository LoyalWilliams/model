from numpy.core import numeric
import pandas as pd
import numpy as np


df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\附件四：354个操作变量信息.xlsx')
# df=df.drop([0],axis=1)
def lb(x):
    if x.startswith('-'):
        x1=x[1:len(x)]
        return -float(x1[0:x1.index('-')]) 
    else:
        return float(x[0:x.index('-')]) 
def ub(x):
    if x.startswith('-'):
        x1=x[1:len(x)]
        x2=x1[x1.index('-')+1:len(x1)].replace('（','').replace('(','').replace('）','').replace(')','').replace('-','')
        return -float(x2) 
    else:
        x1=x[x.index('-')+1:len(x)].replace('（','').replace('(','').replace('）','').replace(')','').replace('-','')
        return float(x1) 
res=pd.DataFrame(columns=[])
res['位号']=df['位号']
res['lb']=df['取值范围'].map(lb)
res['ub']=df['取值范围'].map(ub)

############## 标准化 #####################
cl_df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\last-clean.xlsx')
cl_des=cl_df.drop(['时间','原料:硫含量,μg/g','原料:辛烷值RON','原料:饱和烃,v%（烷烃+环烷烃）','原料:烯烃,v%','原料:芳烃,v%','原料:溴值\n,gBr/100g','原料:密度(20℃),\nkg/m³','产品:硫含量,μg/g','产品:辛烷值RON','产品:RON损失\n（不是变量）','待生:焦炭,wt%','待生:S, wt%','再生:焦炭,wt%','再生:S, wt%']
,axis=1).describe()
res=res[res['位号'].isin(cl_des.columns)]
# for i in cl_des.columns:
#     res.iloc[i][1]=(res['lb'][i]-cl_des[res['位号'][i]]['mean'])/cl_des[res['位号'][i]]['std']
#     res.iloc[i][2]=(res['ub'][i]-cl_des[res['位号'][i]]['mean'])/cl_des[res['位号'][i]]['std']
    
res2=np.arange(0, 3, 1).reshape(1,3)
for i in range(0,res.values.T[0].shape[0]):
    lb=(res.values[i][1]-cl_des[res.values[i][0]]['mean'])/cl_des[res.values[i][0]]['std']
    ub=(res.values[i][2]-cl_des[res.values[i][0]]['mean'])/cl_des[res.values[i][0]]['std']
    res2=np.r_[res2,np.array([[res.values[i][0],lb,ub]])] 
res2=np.delete(res2, 0, axis=0)
res3=pd.DataFrame(res2,columns=res.columns)
res3['lb']=res3['lb'].map(lambda x:float(x))
res3['ub']=res3['ub'].map(lambda x:float(x))
res3.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\range.xlsx',index=False)

########################   有效数据
selec=['原料:辛烷值RON','再生:焦炭,wt%','再生:S, wt%','S-ZORB.TE_2005.PV','S-ZORB.PT_2101.PV','S-ZORB.PC_5101.PV','S-ZORB.TC_5005.PV','S-ZORB.LC_5001.PV','S-ZORB.LC_5101.PV','S-ZORB.FT_5101.PV','S-ZORB.PT_9402.PV','S-ZORB.FT_9401.PV','S-ZORB.PT_9401.PV','S-ZORB.PDC_2502.PV','S-ZORB.FC_2501.PV','S-ZORB.FT_1001.PV','S-ZORB.FC_1203.PV','S-ZORB.PC_1202.PV','S-ZORB.TC_2801.PV','S-ZORB.TE_2001.DACA','S-ZORB.TE_2004.DACA','S-ZORB.TC_3102.DACA','S-ZORB.TE_1102.DACA','S-ZORB.TE_1103.DACA.PV','S-ZORB.TE_1104.DACA.PV','S-ZORB.TE_1102.DACA.PV','S-ZORB.TE_1106.DACA.PV','S-ZORB.FT_1006.TOTALIZERA.PV','S-ZORB.FT_5204.TOTALIZERA.PV']
cl_df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\last-clean.xlsx')
sel_data=pd.DataFrame(columns=[])
for cn in selec:
    sel_data[cn]=cl_df[cn]
sel_data['RON损失']=cl_df['产品:RON损失\n（不是变量）']
sel_data.to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\sel_data.xlsx',index=False)
