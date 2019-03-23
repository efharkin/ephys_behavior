# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:53:04 2019

@author: svc_ccg
"""

import numpy as np
import fileIO
from matplotlib import pyplot as plt

datFile = fileIO.getFile()

d = np.memmap(datFile, dtype = 'int16', mode = 'c')
d = np.reshape(d, (int(d.size/384), 384))

sampleRate = 30000
recordingDuration = d.shape[0]/float(sampleRate)
chunksize = sampleRate*10

# get channel offsets
offsets = np.median(d[:chunksize], axis=0).astype('int16')

#offsets = []
#for chan in np.arange(384):
#    offsets.append(np.median(d[:chunksize, chan]).astype('int16'))
#offsets = np.array(offsets)

plt.figure()
plt.plot(np.std(d[:chunksize], axis=0))

minReferenceChannel = 0
maxReferenceChannel = 150
for ind in np.arange(0, d.shape[0], chunksize):
    start = ind
    end = ind + chunksize if ind + chunksize <= d.shape[0] else d.shape[0]

    #subtract offsets
    d[start:end, :] = d[start:end, :] - offsets[None,:]
    
    #subtract median across channels for every time point
    median_values = np.median(d[start:end, minReferenceChannel:maxReferenceChannel], axis = 1)
    d[start:end, :] = d[start:end, :] - median_values[:, None]

#plt.figure()
#plt.imshow(d[:chunksize].T, aspect='auto')    

plt.plot(np.std(d[:chunksize], axis=0))
