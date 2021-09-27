#问题二运动学片段提取 Python 代码
import numpy as np
import pandas as pd
def getTimepiece(df):
    print(list(df))
    seg=[[0]]
    for i in range(len(df)-1):
        if df.iat[i,3]!=0 and df.iat[i+1,3]==0:
            if df.iat[i+1,1]-df.iat[i,1]==1:
                seg[-1].append(i)
            else:
                seg.pop(-1)
            seg.append([i+1])
    seg.pop(-1)
    i=0
    while i<len(seg):
        if seg[i][1]-seg[i][0]!=df.iat[seg[i][1],1]-df.iat[seg[i][0],1]:
            seg.pop(i)
        elif seg[i][1]-seg[i][0]>500 or seg[i][1]-seg[i][0]<0:
            seg.pop(i)
        else:
            i+=1
    for i in range(len(seg)):
        count=0
        for j in range(seg[i][0],seg[i][1]+1):
            if df.iat[j,1]==0:
                count+=1
            else:
                break
        if count>180:
            seg[i][0]=seg[i][0]+count-180
    # print(' :',len(seg))
    # temp=[x[1]-x[0]+1 for x in seg]
    # temp.sort()
    # for t in temp:
    # print(t)
    return seg
def extract(begin, end, data):
    print(begin)
    Ta=0
    Td=0
    Tc=0
    Ti=0
    S=0
    vmax=0
    amax=0
    amin=0
    jiasusum=0
    jiansusum=0
    xmax=0
    xjz=0
    ymax=0
    yjz=0
    zmax=0
    zjz=0
    zsmax=0
    zsjz=0
    njmax=0
    njjz=0
    yhmax=0
    yhjz=0
    tbmax=0
    tbjz=0
    rbmax=0
    rbjz=0
    fhmax=0
    fhjz=0
    jqmax=0
    jqjz=0
    jsstd=0
    for i in range(begin,end+1):
        jsstd+=data[i][5]**2
        if data[i][5] > 0.1:
            if (i>begin and data[i-1][5] > 0.1) or (i<end and data[i+1][5] > 0.1):
                Ta += 1
                jiasusum += data[i][5]
            elif data[i][5] < -0.1:
                if (i>begin and data[i-1][5] < -0.1) or (i<end and data[i+1][5] < -0.1):
                    Td += 1
                    jiansusum += data[i][5]
            else:
                Tc += 1
        if data[i][4]==0:
            Ti += 1
        S += data[i][4]
        vmax = max(vmax,data[i][4])
        amax = max(amax,data[i][5])
        amin = min(amin,data[i][5])
        xmax = max(xmax,data[i][6])
        xjz += data[i][6]
        ymax = max(ymax,data[i][7])
        yjz += data[i][7]
        zmax = max(zmax,data[i][8])
        zjz += data[i][8]
        zsmax = max(zsmax,data[i][11])
        zsjz += data[i][11]
        njmax = max(njmax,data[i][12])
        njjz += data[i][12]
        yhmax = max(yhmax,data[i][13])
        yhjz += data[i][13]
        tbmax = max(tbmax,data[i][14])
        tbjz += data[i][14]
        rbmax = max(rbmax,data[i][15])
        rbjz += data[i][15]
        fhmax = max(fhmax,data[i][16])
        fhjz += data[i][16]
        jqmax = max(jqmax,data[i][17])
        jqjz += data[i][17]
    T = end-begin+1
    pjsd=S/T
    pjxssd=S/(T-Ti)
    pjjias=jiasusum/Ta
    pjjians=jiansusum/Td
    sdstd=0
    for i in range(begin,end+1):
        sdstd += (data[i][4]-pjsd)**2
    sdstd/=end-begin
    sdstd=sdstd**0.5
    jsstd/=end-begin-1
    jsstd=jsstd**0.5
    xjz /= T
    yjz /= T
    zjz /= T
    zsjz /= T
    njjz /= T
    yhjz /= T
    tbjz /= T
    rbjz /= T
    fhjz /= T
    jqjz /= T
    res=[data[begin][1], data[end][1], T, Ta, Td, Tc, Ti, S, vmax, amax, amin, pjsd, pjxssd,
    pjjias, pjjians, sdstd, jsstd, Ta/T, Td/T, Tc/T, Ti/T, xmax, xjz, ymax, yjz, zmax,
    zjz, zsmax, zsjz, njmax, njjz, yhmax, yhjz, tbmax, tbjz, rbmax, rbjz, fhmax, fhjz,
    jqmax, jqjz]
    print(res)
    return res
def getFeature(df):
    seg = getTimepiece(df)
    feature=[]
    source = list(np.array(df))
    for s in seg:
        try:
            feature.append(extract(s[0],s[1],source))
        except:
            pass
    data=np.array(feature)
    
    col = ['开始', '结束', '行驶时间', '加速时间', '减速时间', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', 'X ', 'X ', 'Y ', 'Y ',
    'Z ', 'Z ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', ' ', ' ',
    ' ']
    outData = pd.DataFrame(data, columns=col)
    outData.to_excel('C:/Users/25416/Desktop/文件1.xlsx')

df = pd.read_excel('C:/Users/25416/Desktop/19/2019年中国研究生数学建模竞赛D题/原始数据/文件1.xlsx')
getFeature(df)