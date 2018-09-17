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


probes_to_run = ('A', 'B', 'C')

def makePSTH(spikes,startTimes,windowDur,binSize=0.1):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    return counts.mean(axis=0)/binSize

def get_ccg(spikes1, spikes2, auto=False, width=0.1, bin_width=0.0005, plot=True):

    d = []                   # Distance between any two spike times
    n_sp = len(spikes2)  # Number of spikes in the input spike train

    
    i, j = 0, 0
    for t in spikes1:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spikes2[i] < t - width:
            i += 1
        while j < n_sp and spikes2[j] < t + width:
            j += 1

        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spikes2[i:j] - t)

    
    d = np.array(d)
    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b, endpoint=True)
    [h, hb] = np.histogram(d, bins=b)
    hh = h.astype(np.float)/(len(spikes1)*len(spikes2))**0.5
    
    if auto:
        hh[n_b-1] = 0 #mask the 0 bin for autocorrelations
    if plot:          
        fig,ax = plt.subplots()
        ax.bar(hb[:-1], hh, bin_width)
        ax.set_xlim([-width,width])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize='xx-small')
        
    return hh, hb

def find_spikes_per_trial(spikes, trial_starts, trial_ends):
    spike_counts = np.zeros(len(trial_starts))
    for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        spike_counts[i] = ((spikes>=ts) & (spikes<te)).sum()  
    return spike_counts
    
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
    

# make psth for units for all flashes of each image
preTime = 0.1
postTime = 0.5
binSize = 0.005
binCenters = np.arange(-preTime,postTime,binSize)+binSize/2
image_flash_times = frameRising[np.array(core_data['visual_stimuli']['frame'])]
image_id = np.array(core_data['visual_stimuli']['image_name'])
for pid in probes_to_run:
    plt.close('all')
    for u in probeSync.getOrderedUnits(units[pid]):
        fig, ax = plt.subplots(imageNames.size)
        fig.suptitle('Probe ' + pid + ': ' + str(u))   
        spikes = units[pid][u]['times']
        maxFR = 0
        for i,img in enumerate(imageNames):
            this_image_times = image_flash_times[image_id==img]         
            psth = makePSTH(spikes,this_image_times-preTime,preTime+postTime,binSize)
            maxFR = max(maxFR, psth.max())
            ax[i].plot(binCenters,psth, 'k')
        
        for ia, a in enumerate(ax):
            a.set_ylim([0, maxFR])
            a.set_yticks([0, round(maxFR+0.5)])
            a.text(postTime*1.05, maxFR/2, imageNames[ia])
            formatFigure(fig, a, xLabel='time (s)', yLabel='FR (Hz)')
            if ia < ax.size-1:
                a.axis('off')
    multipage(os.path.join(dataDir, 'behaviorPSTHs_allflashes_' + pid + '.pdf'))
    


##make psth for units during gratings task
#traceTime = np.linspace(-2, 10, 120)
#goodUnits = probeSync.getOrderedUnits(units)
#for u in goodUnits:
#    spikes = units[u]['times']
#    psthVert = makePSTH(spikes, change_times[np.logical_or(change_ori==90, change_ori==270)]-2, 12)
#    psthHorz = makePSTH(spikes, change_times[np.logical_or(change_ori==0, change_ori==180)]-2, 12)
#    fig, ax = plt.subplots(1, 2)
#    fig.suptitle(str(u) + ': ' + str(units[u]['peakChan']))
#    ax[0].plot(traceTime, psthVert)
#    ax[1].plot(traceTime, psthHorz)
#    for a in ax:    
#        formatFigure(fig, a, '', 'time, s', 'FR, Hz')

    

#######   Analyze RF #########
#First get stimulus pickle file
rfstim_pickle_file = glob.glob(os.path.join(dataDir, '*brain_observatory_stimulus.pkl'))[0]
stim_dict = pd.read_pickle(rfstim_pickle_file)
pre_blank_frames = int(stim_dict['pre_blank_sec']*stim_dict['fps'])
rfstim = stim_dict['stimuli'][0]

#extract trial stim info (xpos, ypos, ori)
sweep_table = np.array(rfstim['sweep_table'])   #table with rfstim parameters, indexed by sweep order to give stim for each trial
sweep_order = np.array(rfstim['sweep_order'])   #index of stimuli for sweep_table for each trial
sweep_frames = np.array(rfstim['sweep_frames']) #(start frame, end frame) for each trial

trial_xpos = np.array([pos[0] for pos in sweep_table[sweep_order, 0]])
trial_ypos = np.array([pos[1] for pos in sweep_table[sweep_order, 0]])
trial_ori = sweep_table[sweep_order, 3]

xpos = np.unique(trial_xpos)
ypos = np.unique(trial_ypos)
ori = np.unique(trial_ori)

#get first frame for this stimulus (first frame after end of behavior session)
first_rf_frame = trials['endframe'].values[-1] + pre_blank_frames + 1
rf_frameRising = frameRising[first_rf_frame:]
trial_start_times = rf_frameRising[np.array([f[0] for f in sweep_frames]).astype(np.int)]
trial_end_times = rf_frameRising[np.array([f[1] for f in sweep_frames]).astype(np.int)]
resp_latency = 0.05
preTime = 0
postTime = 0.5
binSize = 0.01
psthSize = np.arange(preTime, postTime, binSize).size
pid = 'B'
for u in probeSync.getOrderedUnits(units[pid]):
    spikes = units[pid][u]['times']
    #trial_spikes, _ = np.histogram(spikes, bins=np.append(trial_start_times, trial_end_times[-1])+resp_latency)
    trial_spikes = find_spikes_per_trial(spikes, trial_start_times + resp_latency, trial_start_times+resp_latency+0.2)
    respMat = np.zeros([xpos.size, ypos.size, ori.size])
    for (x, y, o, tspikes) in zip(trial_xpos, trial_ypos, trial_ori, trial_spikes):
        respInd = tuple([np.where(xpos==x)[0][0], np.where(ypos==y)[0][0], np.where(ori==o)[0][0]])
        respMat[respInd] += tspikes
    
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]
    fig, ax = plt.subplots()
    fig.suptitle('Probe ' + pid + ': ' + str(u))    
    im = ax.imshow(respMat[:, :, bestOri].T, interpolation='none', origin='lower')
    plt.colorbar(im)

multipage(os.path.join(dataDir, 'rfheatmap_Probe' + pid + '.pdf'))

respMats = np.array(respMats)
plt.figure(str(resp_latency))
plt.imshow(np.mean(np.max(respMats, 3), axis=0), interpolation='none')
respMats = np.array(respMats)
plt.figure(str(resp_latency))
plt.imshow(np.mean(np.max(respMats, 3), axis=0), interpolation='none')


respMats = []
for u in probeSync.getOrderedUnits(units[pid]):
    spikes = units[pid][u]['times']
    respMat = np.zeros([psthSize, xpos.size, ypos.size, ori.size])
    for trialType in np.unique(sweep_order):
        this_trial_starts = trial_start_times[sweep_order==trialType]
        this_trial_psth = makePSTH(spikes, this_trial_starts-preTime, preTime+postTime, binSize)
        this_trial_params = sweep_table[trialType]

        xInd = np.where(xpos==this_trial_params[0][0])
        yInd = np.where(ypos==this_trial_params[0][1])
        oriInd = np.where(ori==this_trial_params[3])
        
        respMat[:, xInd, yInd, oriInd] = this_trial_psth[:, None, None]     
        
        
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]
    fig, ax = plt.subplots(ypos.size, xpos.size)
    fig.suptitle(u)
    for x in np.arange(xpos.size):
        for y in np.arange(ypos.size):
            ax[y,x].plot(respMat[:, x, y, bestOri], 'k')
            ax[y,x].set_ylim([0, respMat.max()])
            ax[y,x].axis('off')
            









