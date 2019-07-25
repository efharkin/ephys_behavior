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

probe = ' SN31058'
dateOfInterest = '2019-04-25'
pdf = df.loc[(df.probeID==probe)]
pdf = pdf.loc[dateOfInterest]

timeDeltas = pdf.index.to_series().diff().dt.microseconds
timeDeltas[0] = 0

plt.plot(timeDeltas, pdf['z'])
