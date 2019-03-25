# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:53:04 2019

@author: svc_ccg
"""

import numpy as np
from matplotlib import pyplot as plt


def medianSubtraction(datFile, minReferenceChannel=0, maxReferenceChannel=150, sampleRate=30000, channelNumber=384, readMode = 'r+', plot=True):
    '''subtract channel offsets and apply a common median referencing
    inputs:
        datFile: path to binary data file
        minReferenceChannel/maxReferenceChannel: first and last channels bracketing the part of the probe that will used for median calculation
        sampleRate: in Hz
        channelNumber: total probe channel count
        readMode: if 'r+' changes will be written in place on disk; if 'c', data will be copied and written to separate file
        plot: if True, plots standard deviation for each channel before and after referencing
    '''
    

    d = np.memmap(datFile, dtype = 'int16', mode = readMode)
    d = np.reshape(d, (int(d.size/channelNumber), channelNumber))
    
    chunksize = sampleRate*10
    
    # get channel offsets
    offsets = np.median(d[:chunksize], axis=0).astype('int16')
    
    # plot pre filter standard deviation
    if plot:
        fig, ax = plt.subplots()
        ax.plot(np.std(d[:chunksize], axis=0))
    
    for ind in np.arange(0, d.shape[0], chunksize):
        start = ind
        end = ind + chunksize if ind + chunksize <= d.shape[0] else d.shape[0]
    
        #subtract offsets
        d[start:end, :] = d[start:end, :] - offsets[None,:]
        
        #subtract median across channels for every time point
        median_values = np.median(d[start:end, minReferenceChannel:maxReferenceChannel], axis = 1)
        d[start:end, :] = d[start:end, :] - median_values[:, None]
    
    
    # plot post filter standard deviation
    if plot:
        ax.plot(np.std(d[:chunksize], axis=0))
        ax.set_xlabel('channel')
        ax.set_ylabel('standard deviation')

    # if opened as copy, save median subtracted data to file
    if readMode == 'c':
        outputDir, outputFile = os.path.split(datFile)
        outputFile, ext = os.path.splitext(outputFile)
        outputFile = outputFile + '_medianSubtracted' + ext        
        
        d.astype('int16').tofile(os.path.join(outputDir, outputFile))
    
    del(d)
        