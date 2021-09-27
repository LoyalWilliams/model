#问题一数据预处理 Python 代码
import pandas as pd
import numpy as np
import datetime
import tqdm
def segments(rawData):
    seg_bound=0
    n=1
    maxl=0
    maxb=0
    count=0
    while n<len(rawData)-2:
        if rawData.iat[n-1,1]==0 and rawData.iat[n,1]!=0 and n-seg_bound>180:
            seg_bound=n-180
            n+=1
        elif rawData.iat[n-1,1]==0 or rawData.iat[n,1]!=0:
            n+=1
        else:
            if n-seg_bound>maxl:
                maxl=n-seg_bound
                maxb=seg_bound
            count+=1
            n+=1
            seg_bound=n
    return maxl,maxb
def checkMissing(rawData):
    jump=[]
    for i in range(1,len(rawData)):
        diff =(rawData.iat[i, 0]-rawData.iat[i-1, 0]).total_seconds()
        if diff>3:
            jump.append(diff)
    return jump
def checkSlow(rawData):
    seg_bound=0
    n=1
    maxl=0
    maxb=0
    count=0
    while n<len(rawData)-2:
        if rawData.iat[n-1,1]==0 and rawData.iat[n,1]!=0 and n-seg_bound>180:
            seg_bound=n-180
            n+=1
        elif rawData.iat[n-1,1]==0 or rawData.iat[n,1]!=0:
            n+=1
        else:
            if n-seg_bound>maxl:
                maxl=n-seg_bound
                maxb=seg_bound
            count+=1
            n+=1
            seg_bound=n
    return maxl,maxb
def process(rawData):
    print(list(rawData))
    i=0
    while i<len(rawData):
        rawData.loc[i, 'GPS车速(m/s)']=rawData.iat[i,2]/3.6
        i+=1
    i=1
    while i<len(rawData):
        timediff = rawData.iat[i,0]-rawData.iat[i-1,0]
        speeddiff = rawData.iat[i,3]-rawData.iat[i-1,3]
        rawData.loc[i, '加速度(m/s^2)']=speeddiff/timediff
        i+=1
    print(rawData.head(10))
    rawData.to_excel('D:/code/py/model/文件2预处理.xlsx')

def removeAbnormalAcceleration(rawData):
    print('删前',rawData.shape[0])
    i=0
    while i<len(rawData):
        if rawData.iat[i,4]<-8:
            rawData.drop(rawData.index[i],inplace=True)
        else:
            i+=1
    print('删后', rawData.shape[0])
    findMissing(rawData)
    rawData.to_excel('D:/code/py/model/文件2预.xlsx')
def findStartupException(rawData):
    res=[]
    i=1
    while i<len(rawData):
        if rawData.iat[i-1,2]==0 and rawData.iat[i,2]!=0:
            j=i-1
            while rawData.iat[i,2]<=100 and rawData.iat[i,0]-rawData.iat[j,0]<=7:
                i+=1
            if rawData.iat[i,2]>100 and rawData.iat[i,0]-rawData.iat[j,0]<=7:
                res.append(j)
        else:
            i+=1
    return res
def findMissing(rawData):
    res = []
    for i in range(1,len(rawData)):
        diff =(rawData.iat[i,1]-rawData.iat[i-1,1]).total_seconds()
        if 1<diff:
            print(rawData.iat[i-1,1],diff)
def findDuplicate(rawData):
    res=[]
    i=0
    while i<len(rawData)-1:
        if (rawData.iat[i+1,1]-rawData.iat[i,1]).total_seconds()==1 and rawData.iat[i,2]>10 and rawData.iat[i+1,8]==rawData.iat[i,8] and rawData.iat[i+1,9]==rawData.iat[i,9]:
            j=i
            while (rawData.iat[i+1,1]-rawData.iat[i,1]).total_seconds()==1 and rawData.iat[i,2]>10 and rawData.iat[i+1,8]==rawData.iat[i,8] and rawData.iat[i+1,9]==rawData.iat[i,9]:
                i+=1
            res.append([j,i+1-j])
        else:
            i+=1
    for r in res:
        print(r[0],r[1])
def removeError(rawData):
    print('删前',rawData.shape[0])
    i=1
    while i<len(rawData):
        if rawData.iat[i,3]>10 and rawData.iat[i-1,9]==rawData.iat[i,9] and rawData.iat[i-1,10]==rawData.iat[i,10]:
            rawData.drop(rawData.index[i], inplace=True)
        else:
            i+=1
        print('删重复后', rawData.shape[0])
    i=0
    while i<len(rawData):
        if rawData.iat[i,5]>4 or rawData.iat[i, 5]<-8:
            rawData.drop(rawData.index[i], inplace=True)
        else:
            i+=1
        print('删加速后', rawData.shape[0])
    i = 0
    while i<len(rawData):
        if rawData.iat[i,9]==0 or rawData.iat[i,10]==0 or rawData.iat[i,3]>120:
            rawData.drop(rawData.index[i], inplace=True)
        else:
            i+=1
    print('删其他异常后', rawData.shape[0])
    # rawData[' ']=rawData[' '].astype('str')
    rawData.to_excel('/Users/yangkai/Downloads/questionD/ 3 .xlsx')
def Interpolation(rawData):
    col = list(rawData)
    print(col)
    data= list(np.array(rawData))
    print(len(rawData))
    i=1
    while i<len(data):
        if 1<data[i][1]-data[i-1][1]<=3:
            if data[i][1]-data[i-1][1]==2:
                date=data[i-1][1]+1
                raw = [0,date,0]
                for j in range(3, 18):
                    raw.append((data[i][j]+data[i-1][j])/2)
                raw[5]=raw[4]-data[i-1][4]
                data[i][5]=data[i][4]-raw[4]
                data.insert(i, raw)
                i+=2
            else:
                print(data[i][1])
                date1=data[i-1][1]+1
                date2=data[i-1][1]+2
                raw1 = [0,date1,0]
                raw2 = [0,date2,0]
                for j in range(3,18):
                    raw1.append((data[i][j]+2*data[i-1][j])/3)
                    raw2.append((2*data[i][j]+data[i-1][j])/3)
                raw1[5]=raw1[4]-data[i-1][4]
                raw2[5]=raw2[4]-raw1[4]
                data[i][5]=data[i][4]-raw2[4]
                data.insert(i, raw2)
                data.insert(i, raw1)
                i+=3
        else:
            i+=1
    data=np.array(data)
    outData = pd.DataFrame(data, columns=col)
    print(len(outData))
    # outData[' ']=outData[' '].astype('str')
    # outData.to_excel('/Users/yangkai/Downloads/questionD/ 3 .xlsx')
def i️dleSpeed(rawData):
    i=0
    print(list(rawData))
    while i<len(rawData)-1:
        if rawData.iat[i,3]==0 and rawData.iat[i+1,3]!=0:
            i+=1
            left=i
            maxs=rawData.iat[i,3]
            while i<len(rawData)-1:
                maxs=max(maxs, rawData.iat[i,3])
                if rawData.iat[i,3]!=0 and rawData.iat[i+1,3]==0:
                    right=i
                    if maxs<=10:
                        for j in range(left,right+1):
                            rawData.iat[j,3]=0
                        i+=1
                        break
                    else:
                        i+=1
        else:
            i+=1
    rawData.to_excel('D:/code/py/model/文件1final.xlsx')
def i️dleSpeedprocess(rawData):
    i=len(rawData) - 1
    print(i)
    count=0
    while i>=0:
        if rawData.iat[i,3]==0:
            count+=1
            if count>180:
                while rawData.iat[i,3]==0:
                    rawData.drop(rawData.index[i], inplace=True)
                    i-=1
            else:
                    i-=1
        else:
            count=0
            i-=1
    print(len(rawData))
df = pd.read_excel('C:/Users/25416/Desktop/19/2019年中国研究生数学建模竞赛D题/原始数据/文件1.xlsx')
print(list(df))
# df[' '] = pd.to_datetime(df[' '], format='%Y/%m/%d %H:%M:%S.000.')
# begin = datetime.datetime.strptime('2017-11-01 00:00:00', '%Y-%m-%d %H:%M:%S')
# df['unix '] = df[' '].map(lambda x:(x-begin).total_seconds())
# col=list(df)
# rawData = df[col[1:-3]]
df['时间'] = df['时间'].astype('str')
# process(df)
# i=0
# while i<len(df):
# if df.iat[i, 9] == 0 or df.iat[i, 10] == 0 or df.iat[i, 3] > 120:
# df.drop(df.index[i], inplace=True)
# else:
# i += 1
# df.to_excel('/Users/yangkai/Downloads/questionD/ 3 .xlsx')
#df[' ']=pd.to_datetime(df[' '],format='%Y-%m-%d %H:%M:%S')
# df['DIFF']=df[' '].diff(1).dt.seconds
# df['DIFF'].fillna(0)
print(df.head(10))
# print(len(df))
# print(list(df))
i️dleSpeedprocess(df)
# i️dleSpeed(df)
i️dleSpeed(df)

# Interpolation(df)
# removeError(df)
# removeAbnormalAcceleration(df)