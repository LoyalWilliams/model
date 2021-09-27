#问题一数据预处理 Python 代码
import pandas as pd
import numpy as np


df = pd.read_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\附件一：325个样本数据.xlsx',header=None)
df.describe().to_excel('D:\\课程\\建模\\2020年第十七届华为杯研究生数学建模竞赛完整题目(A,B,C,D,E,F)内含附件\\2020年B题\\数模题\\description.xlsx',index=True)
