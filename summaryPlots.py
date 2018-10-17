# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:30:18 2018

@author: svc_ccg
"""
from __future__ import division
import os, cv2
import probeSync
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import patches
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


def all_unit_summary(probesToAnalyze, units, dataDir, runSpeed, runTime, name_tag = ''):
    plt.close('all')
    
    for pid in probesToAnalyze:
        multipageObj = PdfPages(os.path.join(dataDir, 'SummaryPlots_' + pid + name_tag + '.pdf'))
        orderedUnits = probeSync.getOrderedUnits(units[pid])
        for u in orderedUnits:
            plot_unit_summary(pid, u, units, multipageObj)
        
#        multipage(os.path.join(dataDir, 'summaryPlots_' + pid + '.pdf'))
#        plt.close('all')
        multipageObj.close()
        
def plot_unit_summary(pid, uid, units, multipageObj=None):
    spikes = units[pid][uid]['times']
    fig = plt.figure(facecolor='w', figsize=(16,12))
    if 'ccfRegion' in units[pid][uid] and units[pid][uid]['ccfRegion'] is not None:
        figtitle = 'Probe: ' + str(pid) + ', unit: ' + str(uid) + ' ' + units[pid][uid]['ccfRegion']
    else:
        figtitle = 'Probe: ' + str(pid) + ', unit: ' + str(uid)
        
    fig.suptitle(figtitle)
    
    gs = gridspec.GridSpec(8, 7)
    gs.update(top=0.95, bottom = 0.35, left=0.05, right=0.95, wspace=0.3)
    
    rfaxes = [plt.subplot(gs[i,1]) for i in range(8)]
    plot_rf(spikes, rfaxes, resp_latency=0.05)
    
#    allflashax = plt.subplot(gs[:, :7])
#    plot_psth_all_flashes(spikes, frameTimes, core_data, allflashax, preTime = 0.1, postTime = 0.75)
#    
#    allrespax = plt.subplot(gs[:, 8:14])
#    plot_psth_hits_vs_misses(spikes, frameTimes, trials, allrespax, preTime = 1.5, postTime = 3, average_across_images=False)
    
    plt.close(fig)


def getSDF(spikes,startTimes,windowDur,sigma=0.02,sampInt=0.001,avg=True):
        t = np.arange(0,windowDur+sampInt,sampInt)
        counts = np.zeros((startTimes.size,t.size-1))
        for i,start in enumerate(startTimes):
            counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,t)[0]
        sdf = scipy.ndimage.filters.gaussian_filter1d(counts,sigma/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        return sdf,t[:-1]
        
def makePSTH(spikes,startTimes,windowDur,binSize=0.1, avg=True):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    if avg:
        return counts.mean(axis=0)/binSize
    else:
        return np.array(counts)/binSize
        
def find_spikes_per_trial(spikes, trial_starts, trial_ends):
    spike_counts = np.zeros(len(trial_starts))
    for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        spike_counts[i] = ((spikes>=ts) & (spikes<te)).sum()  
    return spike_counts




def plot_rf(spikes, axes, resp_latency=0.05):
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
    first_rf_frame = trials['endframe'].values[-1] + rf_pre_blank_frames + 1
    rf_frameTimes = frameTimes[first_rf_frame:]
    rf_trial_start_times = rf_frameTimes[np.array([f[0] for f in sweep_frames]).astype(np.int)]
    
    trial_spikes = find_spikes_per_trial(spikes, rf_trial_start_times+resp_latency, rf_trial_start_times+resp_latency+0.2)
    respMat = np.zeros([ypos.size, xpos.size, ori.size])
    for (y, x, o, tspikes) in zip(trial_ypos, trial_xpos, trial_ori, trial_spikes):
        respInd = tuple([np.where(ypos==y)[0][0], np.where(xpos==x)[0][0], np.where(ori==o)[0][0]])
        respMat[respInd] += tspikes
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]  
    r = respMat[:,:,bestOri]
    
    gridSpacingDeg = xpos[1]-xpos[0]
    gridSpacingPix = int(round(imagePixPerDeg*gridSpacingDeg))
    a = r.copy()
    a -= a.min()
    a *= 255/a.max()
    a = np.repeat(np.repeat(a.astype(np.uint8),gridSpacingPix,axis=0),gridSpacingPix,axis=1)
    alpha = np.zeros_like(images[0])
    y0,x0 = (int(images[0].shape[i]/2-a.shape[i]/2) for i in (0,1))
    alpha[y0:y0+a.shape[0],x0:x0+a.shape[1]] = a
    
    for ax,img,imname in zip(axes,images,imageNames):
        rgba = np.stack((img,)*3+(alpha,),axis=2)
        ax.patch.set_alpha(0.0)
        ax.imshow(rgba)
        ax.set_axis_off()
        ax.set_title(imname,fontsize=12)




