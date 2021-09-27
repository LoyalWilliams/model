import pandas as pd
import numpy as np

company=["A","B","C"]

data=pd.DataFrame({
    "company":[company[x] for x in np.random.randint(0,len(company),10)],
    "salary":np.random.randint(5,50,10),
    "age":np.random.randint(15,50,10)
}
)
data['one']=1

# 计算不同公司员工的平均年龄以及薪水的中位数，可以利用字典进行聚合操作的指定：
data.groupby('company').agg({'salary':'median','age':'mean'})

# 现在需要在原数据集中新增一列avg_salary
avg_salary_dict = data.groupby('company')['salary'].mean().to_dict()
data['avg_salary']= data['company'].map(avg_salary_dict)
# 如果使用transform的话，仅需要一行代码：
data['avg_salary'] = data.groupby('company')['salary'].transform('mean')

# 假设我现在需要获取各个公司年龄最大的员工的数据，该怎么实现呢
def get_oldest_staff(x):
    df = x.sort_values(by = 'age',ascending=True)
    return df.iloc[-1,:]

oldest_staff = data.groupby('company',as_index=False).apply(get_oldest_staff)



# map  apply  applymap
# 三板斧#①使用字典进行映射
data["gender"] = data["gender"].map({"男":1, "女":0})
#②使用函数
def gender_map(x):
    gender = 1 if x == "男" else 0
    return gender
#注意这里传入的是函数名，不带括号
data["gender"] = data["gender"].map(gender_map)

# DataFrame数据处理
# 1. apply  0：行   1：列
# 沿着0轴求和
data[["height","weight","age"]].apply(np.sum, axis=0)
# 沿着0轴取对数
data[["height","weight","age"]].apply(np.log, axis=0)

# 现在想将DataFrame中所有的值保留两位小数显示
data.applymap(lambda x:"%.2f" % x)