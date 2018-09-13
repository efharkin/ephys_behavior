# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os
import glob
from sync import sync
import probeSync
import behavSync
import numpy as np
import matplotlib.pyplot as plt
from probeData import formatFigure
from visual_behavior.visualization.extended_trials.daily import make_daily_figure

def makePSTH(spike_times, trial_start_times, trial_duration, bin_size = 0.1):
    counts = np.zeros(int(trial_duration/bin_size))    
    for ts in trial_start_times:
        for ib, b in enumerate(np.arange(ts, ts+trial_duration, bin_size)):
            c = np.sum((spike_times>=b) & (spike_times<b+bin_size))
            counts[ib] += c
    return counts/len(trial_start_times)/bin_size


# psth for hit and miss trials for each image
preTime = 1
postTime = 1
binSize = 0.05
binCenters = np.arange(-preTime,postTime,binSize)+binSize/2
for pid in probeIDs:
    for u in probeSync.getOrderedUnits(units[pid]):
        fig = plt.figure(facecolor='w',figsize=(8,10))
        spikes = units[pid][u]['times']
        for i,img in enumerate(imageNames):
            ax = plt.subplot(imageNames.size,1,i+1)
            for resp,clr in zip((hit,miss),'rb'):
                selectedTrials = resp & (changeImage==img) & (~ignore)
                changeTimes = frameRising[np.array(trials['change_frame'][selectedTrials]).astype(int)]
                psth = makePSTH(spikes,changeTimes-preTime,preTime+postTime,binSize)
                ax.plot(binCenters,psth,clr)



