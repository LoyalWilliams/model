#问题二 特征选择
import pandas as pd
import numpy as np
import datetime
import tqdm

# H(x)=-sum(pi*logpi)
def calc_ent(x):
    """
        calculate shanno ent of x
    """

    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp

    return ent

# H(y|xi)=sum(Pi*H(Y|X=xi))
def calc_condition_ent(x, y):
    """
        calculate ent H(y|x)
    """
    # calc ent(y|x)
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent

    return ent

def calc_ent_grap(x,y):
    """
        calculate ent grap
    """

    base_ent = calc_ent(y)
    condition_ent = calc_condition_ent(x, y)
    ent_grap = base_ent - condition_ent

    return ent_grap


df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\clean1.xlsx')
df1=df.drop(['时间','''产品:RON损失
（不是变量）'''],axis=1)

columns={}
for i in range(0,df1.columns.values.shape[0]):
    columns[df1.columns.values[i]]=i
    columns[i]=df1.columns.values[i]


quantile_table=df1.quantile([0, 1], numeric_only=True)
class_table=np.zeros((1, df1.columns.values.shape[0]))
for i in range(0,df1.columns.values.shape[0]):
    cname=columns[i]
    min1=quantile_table[cname][0]
    max1=quantile_table[cname][1]
    q1=float(max1-min1)/2+min1
    tmpv=(df1[cname]<q1).values.astype(int)
    # q1=float(max1-min1)/6+min1
    # q2=float(max1-min1)*2/6+min1
    # q3=float(max1-min1)*3/6+min1
    # q4=float(max1-min1)*4/6+min1
    # q5=float(max1-min1)*5/6+min1
    # tmpv=(df1[cname]<q1).values.astype(int)
    # tmp2=(df1[cname]<q2).values.astype(int)
    # tmp3=(df1[cname]<q3).values.astype(int)
    # tmp4=(df1[cname]<q4).values.astype(int)
    # tmp5=(df1[cname]<q5).values.astype(int)
    # tmp6=(df1[cname]<=max1).values.astype(int)
    # tmpv+=tmp2+tmp3+tmp4+tmp5+tmp6
    if (class_table==np.zeros((df1.shape[0],1))).all():
        class_table=tmpv
    else:
        class_table=np.c_[class_table,tmpv]

clt_df=pd.DataFrame(class_table)
for i in range(0,class_table.shape[1]):
    clt_df.rename(columns={i:columns[i]},inplace=True)



D='原料:辛烷值RON'
C=set(df1.columns.values)-{D}


def HD():
    HDv=clt_df.groupby([D]).size().values
    ent=0
    for i in range(0,HDv.shape[0]):
        p = float(HDv[i]) / HDv.sum()
        logp = np.log2(p)
        ent -= p * logp
    HD=ent
    return HD

def HDA(A):
    # A=['原料:芳烃,v%','原料:密度(20℃),\nkg/m³']
    if not A:
        return HD()
    A=list(A)
    AD=list(A.copy())
    # clt_df.groupby([A,D]).size()
    AD.insert(0,D)
    clt_df.groupby([D]).size().reset_index(name='times')['times']
    a1=clt_df.groupby(AD).size().reset_index(name='times1')
    a2=clt_df.groupby(A).size().reset_index(name='times2')
    a3=pd.merge(a1, a2,  how='left', left_on=A, right_on = A)
    a3['HAi1']=a3['times1']/a3['times2']
    a4=a3.groupby(A).apply(lambda x: -np.sum(x['HAi1']*np.log2(x['HAi1']))).reset_index(name='HAi2')
    a5=pd.merge(a4, a2,  how='left', left_on=A, right_on = A)
    HDA=np.sum(a5['HAi2']*a5['times2'].values)/a5['times2'].values.sum()
    return HDA

def gainAD(A):
    return HD()-HDA(A)

def SGFout(a,A):
    return HDA(A)-HDA(A|{a})
def SGFin(a,A):
    return HDA(A-{a})-HDA(A)

R=set()
# 步骤2
ai=C.pop()
C=C|{ai}
if SGFin(ai,C)>0:
    R=R|{ai}

while True:
    # 步骤3
    if gainAD(R)==gainAD(C):
        # 步骤5
        fl=True
        for ri in R:
        # ri=R.pop()
        # R=R|{ri}
            if gainAD(R-{ri})>gainAD(R):
                R=R-{ri}
                fl=False
                break
        if fl:
            break
    else:
        # 步骤4
        CsR=C-R
        maxarg=0
        c=''
        for bi in CsR:
            sgf=SGFout(bi,R)
            if sgf>maxarg:
                maxarg=sgf
                c=bi
        R=R|{c}
R      







from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression


df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\last-clean.xlsx')
df1=df.drop(['时间','''产品:RON损失
（不是变量）'''],axis=1)


# 创建selector，定义检验方法，因为这是分类问题，而且特征全部是数值变量，选择'f_classif'
# 保留关联程度最高的4个特征
selector = SelectKBest(score_func=f_regression, k=30)
X=df.drop(['时间','''产品:RON损失
（不是变量）''','原料:辛烷值RON'],axis=1)
# 拟合数据，同时提供特征和目标变量
results = selector.fit(X, df['原料:辛烷值RON'].values)

# 查看每个特征的分数和p-value
# results.scores_: 每个特征的分数
# results.pvalues_: 每个特征的p-value
# results.get_support(): 返回一个布尔向量，True表示选择该特征，False表示放弃，也可以返回索引
features = pd.DataFrame({
    "feature": X.columns,
    "score": results.scores_,
    "pvalue": results.pvalues_,
    "select": results.get_support()
})
features.sort_values("score", ascending=False)
features.sort_values("score", ascending=False).to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\fetures.xlsx')




from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
