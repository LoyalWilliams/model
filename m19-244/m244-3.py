#问题三行驶工况构建 Python 代码
import numpy as np
import pandas as pd
def extract(begin, end, data):
    print(begin)
    Ta,Td,Tc,Ti,S=0,0,0,0,0
    vmax,amax,amin=0,0,0
    jiasusum,jiansusum=0,0
    xmax,xjz=0,0
    ymax,yjz=0,0
    zmax,zjz=0,0
    zsmax,zsjz=0,0
    njmax,njjz=0,0
    yhmax,yhjz=0,0
    tbmax,tbjz=0,0
    rbmax,rbjz=0,0
    fhmax,fhjz=0,0
    jqmax,jqjz=0,0
    jsstd=0,0
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
def evaluate(df):
    T,S,Ta,Td,Tc,Ti=0,0,0,0,0,0
    pjjiasd,pjjiansd=0,0
    sdstd,jsdstd=0,0
    xjz,yjz,zjz=0,0,0
    zsjz,njjz=0,0
    yhjz,tbjz=0,0
    krjz,fhjz=0,0
    jqjz=0
    for i in range(len(df)):
        T+=df.iat[i, 3]
        Ta+=df.iat[i, 4]
        Td+=df.iat[i, 5]
        Tc+=df.iat[i, 6]
        Ti+=df.iat[i, 7]
        S+=df.iat[i,8]
        pjjiasd+=df.iat[i,14]*df.iat[i, 4]
        pjjiansd+=df.iat[i,15]*df.iat[i, 5]
        sdstd+=df.iat[i,16]**2*(df.iat[i,3]-1)
        jsdstd+=df.iat[i,17]**2*(df.iat[i,3]-2)
        xjz+=df.iat[i,22]*df.iat[i,3]
        yjz+=df.iat[i,24]*df.iat[i,3]
        zjz+=df.iat[i,26]*df.iat[i,3]
        zsjz+=df.iat[i,28]*df.iat[i,3]
        njjz+=df.iat[i,30]*df.iat[i,3]
        yhjz+=df.iat[i,32]*df.iat[i,3]
        tbjz+=df.iat[i,34]*df.iat[i,3]
        krjz+=df.iat[i,36]*df.iat[i,3]
        fhjz+=df.iat[i,38]*df.iat[i,3]
        jqjz+=df.iat[i,40]*df.iat[i,3]
    print(' ',S/T)
    print(' ',S/(T-Ti))
    print(' ',pjjiasd/Ta)
    print(' ',pjjiansd/Td)
    print(' ',(sdstd/(T-1))**0.5)
    print(' ', (jsdstd/(T-2))**0.5)
    print(' ',Ta/T)
    print(' ',Td/T)
    print(' ',Tc/T)
    print(' ',Ti/T)
    print('X ',xjz/T)
    print('Y ',yjz/T)
    print('Z ',zjz/T)
    print(' ',zsjz/T)
    print(' ',njjz/T)
    print(' ',yhjz/T)
    print(' ',tbjz/T)
    print(' ',krjz/T)
    print(' ',fhjz/T)
    print(' ',jqjz/T)
df = pd.read_excel('/Users/yangkai/Downloads/questionD/ - 123 .xlsx')
evaluate(df)