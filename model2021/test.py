import pandas as pd
import numpy as np

pps=pd.DataFrame()
kernel='linear'
degree=8
C     =2.481555938720703
gamma =48.54907989501953
coef0 =35.8707770068903

params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0,'分类目标':'CYP3A4'}

pps=pps.append(pd.DataFrame([params]),ignore_index=True)

kernel='linear'
degree=5
C     =5.950641632080078
gamma =75.25835037231445
coef0 =84.26493097775553

params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0,'分类目标':'MN'}

pps=pps.append(pd.DataFrame([params]),ignore_index=True)


kernel='linear'
degree=3
C     =3.6070823669433594
gamma =40.64168930053711
coef0 =67.76644493717664

params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0,'分类目标':'HOB'}

pps=pps.append(pd.DataFrame([params]),ignore_index=True)


kernel='linear'
degree=1
C     =83.84809494018555
gamma =44.15740966796875
coef0 =60.71516105190378

params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0,'分类目标':'Caco-2'}

pps=pps.append(pd.DataFrame([params]),ignore_index=True)

kernel='linear'
degree=8
C     =1.3010025024414062
gamma =91.88404083251953
coef0 =53.59454497770784
params = {'kernel': kernel, 'C': C,
              'degree': degree, 'gamma': gamma, 'coef0': coef0,'分类目标':'hERG'}

pps=pps.append(pd.DataFrame([params]),ignore_index=True)
pps.to_excel('p_svr.xlsx',index=False)