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


dataDir = "\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\Behavior\\08152018_385531"
sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = sync.Dataset(sync_file)

units = probeSync.getUnitData(syncDataset)

pkl_file = glob.glob(os.path.join(dataDir, '*.pkl'))[0]
trials,frameRising,frameFalling = behavSync.getBehavData(syncDataset,pkl_file)

#align trials to clock
trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameRising[trial_start_frames]
trial_end_times = frameFalling[trial_end_frames]

trial_ori = np.array(trials['initial_ori'])
    
notNullTrials = trials['change_frame'].notnull()
change_frames = np.array(trials['change_frame'][notNullTrials]).astype(int)
change_times = frameRising[change_frames]
change_ori = np.array(trials['change_ori'])[notNullTrials]


#make psth for units

def makePSTH(spike_times, trial_start_times, trial_duration, bin_size = 0.1):
    counts = np.zeros(int(trial_duration/bin_size))    
    for ts in trial_start_times:
        for ib, b in enumerate(np.arange(ts, ts+trial_duration, bin_size)):
            c = np.sum((spike_times>=b) & (spike_times<b+bin_size))
            counts[ib] += c
    return counts/len(trial_start_times)/bin_size

traceTime = np.linspace(-2, 10, 120)
goodUnits = getOrderedUnits(units)
for u in goodUnits:
    spikes = units[u]['times']
    psthVert = makePSTH(spikes, change_times[np.logical_or(change_ori==90, change_ori==270)]-2, 12)
    psthHorz = makePSTH(spikes, change_times[np.logical_or(change_ori==0, change_ori==180)]-2, 12)
    fig, ax = plt.subplots(1, 2)
    fig.suptitle(str(u) + ': ' + str(units[u]['peakChan']))
    ax[0].plot(traceTime, psthVert)
    ax[1].plot(traceTime, psthHorz)
    for a in ax:    
        formatFigure(fig, a, '', 'time, s', 'FR, Hz')
    

#Make summary pdf of unit responses    
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
    
multipage(os.path.join(dataDir, 'behaviorPSTHs_08022018.pdf'))