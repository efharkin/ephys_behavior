# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 16:54:59 2019

@author: svc_ccg
"""

import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot as plt

logFile = r"Z:\newscale_log_07132019.txt"
df = pd.read_csv(logFile, header=None, names=['time', 'probeID', 'x', 'y', 'z', 'relx', 'rely', 'relz'])
df['time'] = pd.to_datetime(df['time'])
df = df.set_index('time')

pid = ' SN34029'
dateOfInterest = '2019-04-25'
pdf = df.loc[dateOfInterest]

for pid in np.unique(pdf.probeID.values):
    tempdf = pdf.loc[pdf.probeID==pid]
    tempdf.plot(y='z')
    diffs = tempdf['z'].diff()
    timeDeltas = tempdf.index.to_series().diff().astype('timedelta64[s]')
    
    #find the first time such that the next 20 deltas are all small
    rollingDelta = timeDeltas.rolling(10, win_type='boxcar').sum().dropna()
    start = rollingDelta.where(rollingDelta<1).dropna().index[0]
    end = rollingDelta.where(rollingDelta<1).dropna().index[-1]
    
    #insertionTimes = timeDeltas.where((timeDeltas<0.1)&(timeDeltas.shift(periods=-1)<0.1).dropna(axis=0)
    #insertionTimes = timeDeltas.where((timeDeltas<0.1)&(timeDeltas.shift(periods=-10)<0.1)&(timeDeltas.shift(periods=-20)<0.1)).dropna(axis=0)
    #insertionTimes = timeDeltas.where((timeDeltas<0.1)&(abs(diffs))<10).dropna(axis=0)
    
    
    insertiondf = tempdf.loc[start:end]
    
    ax = plt.gca()
    ax.set_title(pid)
    ax.plot(start, insertiondf.iloc[0, 3], 'go')
    ax.plot(end, insertiondf.iloc[-1, 3], 'ro')
    
    print(pid)
    print('Insertion start coords: ' + str(insertiondf.iloc[0, 0:4]))
    print('Insertion end coords: ' + str(insertiondf.iloc[-1, 0:4]))
    print('\n\n')