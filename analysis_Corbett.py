# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os
import glob
from sync import sync
import probeSync
import numpy as np
import matplotlib.pyplot as plt
from probeData import formatFigure
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
import pandas as pd
import scipy
import clust

probes_to_run = ('A', 'B')

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

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def add_fig_to_multipage(multipageObj, fig):
    fig.savefig(multipageObj, format='pdf')
    
    

# make psth for units for all flashes of each image
#preTime = 0.1
#postTime = 0.5
#binSize = 0.005
#binCenters = np.arange(-preTime,postTime,binSize)+binSize/2
#image_flash_times = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
#image_id = np.array(core_data['visual_stimuli']['image_name'])
#for pid in probes_to_run:
#    plt.close('all')
#    for u in probeSync.getOrderedUnits(units[pid]):
#        fig, ax = plt.subplots(imageNames.size)
#        fig.suptitle('Probe ' + pid + ': ' + str(u))   
#        spikes = units[pid][u]['times']
#        maxFR = 0
#        for i,img in enumerate(imageNames):
#            this_image_times = image_flash_times[image_id==img]         
#            psth = makePSTH(spikes,this_image_times-preTime,preTime+postTime,binSize)
#            maxFR = max(maxFR, psth.max())
#            ax[i].plot(binCenters,psth, 'k')
#        
#        for ia, a in enumerate(ax):
#            a.set_ylim([0, maxFR])
#            a.set_yticks([0, round(maxFR+0.5)])
#            a.text(postTime*1.05, maxFR/2, imageNames[ia])
#            formatFigure(fig, a, xLabel='time (s)', yLabel='FR (Hz)')
#            if ia < ax.size-1:
#                a.axis('off')
#    multipage(os.path.join(dataDir, 'behaviorPSTHs_allflashes_' + pid + '.pdf'))
    
def plot_psth_all_flashes(spikes, frameTimes, core_data, axis, preTime = 0.1, postTime = 0.5, sdfSigma=0.005):
    image_flash_times = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
    image_id = np.array(core_data['visual_stimuli']['image_name'])
    imageNames = np.unique(image_id)
    
    sdfs = []
    latencies = []
    for i,img in enumerate(imageNames):
        this_image_times = image_flash_times[image_id==img]         
        sdf, t = getSDF(spikes,this_image_times-preTime,preTime+postTime, sigma=sdfSigma)
        latency = find_latency(sdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
        latencies.append(latency)
        sdfs.append(sdf)
        
    maxFR = np.max(sdfs)
    for ip, sdf in enumerate(sdfs):
        axis.plot(t - preTime, sdf + ip*maxFR, 'k')
        if ~np.isnan(latencies[ip]):
            axis.plot(latencies[ip]/1000. - preTime, sdf[latencies[ip]]+ip*maxFR, 'ro')
        axis.text(postTime*0.85, (ip+0.5)*maxFR, imageNames[ip])  
    
    axis.set_yticks([0, round(maxFR+0.5)])
    axis.spines['left'].set_visible(False)
    formatFigure(plt.gcf(), axis, xLabel='time to flash (s)', yLabel='Spikes/s')
   
def find_latency(signal, baseline_end = 100, stdev_thresh = 3, min_points_above_thresh=30):
    thresh = stdev_thresh*np.std(signal[:baseline_end]) + np.mean(signal[:baseline_end])   
    over_std = np.where(signal>thresh)[0]

    if len(over_std)==0:
        return np.nan
    
    counter = 1
    cand = over_std[0]
    while any(signal[cand:cand+min_points_above_thresh]<thresh):
        cand = over_std[counter]
        counter += 1
        if counter==len(over_std):
            return np.nan
    
    return cand
       
def make_psth_all_flashes(spikes, frameTimes, core_data, preTime = 0.1, postTime = 0.5, sdfSigma=0.005):
    image_flash_times = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
    image_id = np.array(core_data['visual_stimuli']['image_name'])
    imageNames = np.unique(image_id)
    
    sdfs = []
    latencies = []
    for i,img in enumerate(imageNames):
        this_image_times = image_flash_times[image_id==img]         
        sdf, t = getSDF(spikes,this_image_times-preTime,preTime+postTime, sigma=sdfSigma)
        latency = find_latency(sdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
        latencies.append(latency)
        sdfs.append(sdf)
        
    return np.array(sdfs), np.array(latencies)
    
def compute_lifetime_sparseness(spikes, frameTimes, core_data, preTime = 0.1, postTime = 0.5, sdfSigma=0.005, latency=0.05):
    sdfs,_ = make_psth_all_flashes(spikes, frameTimes, core_data, preTime, postTime, sdfSigma)
    meanResps = np.mean(sdfs[:, int(1000*(preTime+latency)):int(1000*(preTime+latency+0.25))], 1)
    meanBaselines = np.mean(sdfs[:, 0:int(1000*preTime+latency/2)], 1)
    
    stimResps = meanResps-meanBaselines    
    
    sumsquared = np.sum(stimResps)**2
    sumofsquares = np.sum(stimResps**2)
    
    N = sdfs.shape[0]
    
    Sl = (1 - (sumsquared/sumofsquares)/N)/(1-1/N)
    
    return Sl
    
sl = []
regionsToConsider = ['VIS']
for u in probeSync.getOrderedUnits(units[pid]):
    spikes = units[pid][u]['times']
    slu = compute_lifetime_sparseness(spikes, frameTimes, core_data)     
    units[pid][u]['lifetime_sparseness'] = slu

    region = units[pid][u]['ccfRegion']    
    if region is not None and any([r in region for r in regionsToConsider]):
        sl.append(slu)

plt.figure()
plt.hist(sl)
    
    
# sdf for all hit and miss trials    
#preTime = 1.5
#postTime = 1.5
#sdfSigma = 0.02
#for pid in probes_to_run:
#    plt.close('all')
#    orderedUnits = probeSync.getOrderedUnits(units[pid])
#    for u in orderedUnits:
#        spikes = units[pid][u]['times']
#        fig = plt.figure(facecolor='w')
#        ax = plt.subplot(1,1,1)
#        ymax = 0
#        for resp,clr in zip((hit,miss),'rb'):
#            selectedTrials = resp & (~ignore)
#            changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
#            sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
#            ax.plot(t-preTime,sdf,clr)
#            ymax = max(ymax,sdf.max())
#        for side in ('right','top'):
#            ax.spines[side].set_visible(False)
#        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
#        ax.set_xlim([-preTime,postTime])
#        ax.set_ylim([0,1.02*ymax])
#        ax.set_xlabel('Time relative to image change (s)',fontsize=12)
#        ax.set_ylabel('Spike/s',fontsize=12)
#        ax.set_title('Probe '+pid+', Unit '+str(u),fontsize=12)
#        plt.tight_layout()
#    multipage(os.path.join(dataDir, 'behaviorPSTHs_combined_hits_misses_' + pid + '.pdf'))

def plot_psth_hits_vs_misses(spikes, frameTimes, trials, axis, preTime = 1.5, postTime = 1.5, sdfSigma=0.02, average_across_images=True):
    autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
    earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')    
    ignore = earlyResponse | autoRewarded
    miss = np.array(trials['response_type']=='MISS')
    hit = np.array(trials['response_type']=='HIT')
    falseAlarm = np.array(trials['response_type']=='FA')
    correctReject = np.array(trials['response_type']=='CR')
    ymax = 0
    if average_across_images:
        plotlines = []
        for resp,clr in zip((hit,miss, falseAlarm, correctReject),'bkrg'):
            selectedTrials = resp & (~ignore)
            changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
            sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
            plotline, = axis.plot(t-preTime,sdf,clr)
            plotlines.append(plotline)
            ymax = max(ymax,sdf.max())
        
        axis.set_xlim([-preTime,postTime])
        axis.set_ylim([0,1.02*ymax])
        axis.legend(tuple([plotline for plotline in plotlines]), ('hit', 'miss', 'false alarm', 'correct reject'), loc='best', prop={'size':8})
        formatFigure(plt.gcf(), axis, xLabel='Time to image change (s)', yLabel='Spikes/s')
    else:
        changeImage = np.array(trials['change_image_name'])
        imageNames = np.unique(changeImage)
        sdfs = [[] for respType in [hit, miss]]
        for i,img in enumerate(imageNames):
            for ir, resp in enumerate([hit, miss]):
                selectedTrials = resp & (changeImage==img) & (~ignore)
                changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
                sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
                sdfs[ir].append(sdf)
        
        maxFR = np.max(sdfs)
        clrs = 'bkrg'
        for im, respsdfs in enumerate(zip(*sdfs)):
#            axis.text(postTime*1.05, (im+0.5)*maxFR, imageNames[im])  
            for ir, respsdf in enumerate(respsdfs):            
                axis.plot(t-preTime, respsdf + im*maxFR, clrs[ir])
        
        
        for side in ('right','top'):
            axis.spines[side].set_visible(False)
        axis.tick_params(direction='out',top=False,right=False,labelsize=10)
        axis.set_xlim([-preTime,postTime])
#        axis.set_ylabel('Spikes/s',fontsize=12)
        axis.set_xlabel('Time relative to image change (s)',fontsize=12)
        axis.set_yticks([0, round(maxFR+0.5)])
        axis.spines['left'].set_visible(False)   

# sdf for hit and miss trials for each image
#for pid in probes_to_run:
#    plt.close('all')
#    orderedUnits = probeSync.getOrderedUnits(units[pid])
#    for u in orderedUnits:
#        spikes = units[pid][u]['times']
#        fig = plt.figure(facecolor='w',figsize=(8,10))
#        axes = []
#        ymax = 0
#        for i,img in enumerate(imageNames):
#            axes.append(plt.subplot(imageNames.size,1,i+1))
#            for resp,clr in zip((hit,miss),'rb'):
#                selectedTrials = resp & (changeImage==img) & (~ignore)
#                changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
#                sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
#                axes[-1].plot(t-preTime,sdf,clr)
#                ymax = max(ymax,sdf.max())
#        for ax,img in zip(axes,imageNames):
#            for side in ('right','top'):
#                ax.spines[side].set_visible(False)
#            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
#            ax.set_xlim([-preTime,postTime])
#            ax.set_ylim([0,1.02*ymax])
#            ax.set_ylabel(img,fontsize=12)
#            if ax!=axes[-1]:
#                ax.set_xticklabels([])
#        axes[-1].set_xlabel('Time relative to image change (s)',fontsize=12)
#        axes[0].set_title('Probe '+pid+', Unit '+str(u),fontsize=12)
#        plt.tight_layout()
#    multipage(os.path.join(dataDir, 'behaviorPSTHs_image_hits_misses_' + pid + '.pdf'))

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

    
##############################
#######   Analyze RF #########
#First get stimulus pickle file
    
def get_rf_trial_params(dataDir, stim_dict=None):

    if stim_dict is None:
        rfstim_pickle_file = glob.glob(os.path.join(dataDir, '*brain_observatory_stimulus.pkl'))[0] 
        stim_dict = pd.read_pickle(rfstim_pickle_file)
    
    pre_blank_frames = int(stim_dict['pre_blank_sec']*stim_dict['fps'])
    rfstim = stim_dict['stimuli'][0]
    
    return rfstim, pre_blank_frames

def plot_rf(spikes, frameTimes, trials, dataDir, axis, rfstim, pre_blank_frames, resp_latency=0.05):
    
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
    rf_frameTimes = frameTimes[first_rf_frame:]
    rf_trial_start_times = rf_frameTimes[np.array([f[0] for f in sweep_frames]).astype(np.int)]
    resp_latency = 0.05

    trial_spikes = find_spikes_per_trial(spikes, rf_trial_start_times + resp_latency, rf_trial_start_times+resp_latency+0.2)
    respMat = np.zeros([xpos.size, ypos.size, ori.size])
    for (x, y, o, tspikes) in zip(trial_xpos, trial_ypos, trial_ori, trial_spikes):
        respInd = tuple([np.where(xpos==x)[0][0], np.where(ypos==y)[0][0], np.where(ori==o)[0][0]])
        respMat[respInd] += tspikes
    
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]  
    im = axis.imshow(respMat[:, :, bestOri].T, interpolation='none', origin='lower')
    plt.colorbar(im)    
    tickLabels = [str(tick) for tick in np.unique(xpos)[::2]]
    axis.set_xticks(np.arange(0, len(np.unique(xpos)), 2))
    axis.set_yticks(np.arange(0, len(np.unique(xpos)), 2))
    axis.set_xticklabels(tickLabels)
    axis.set_yticklabels(tickLabels)
        

        



#respMats = []
#for u in probeSync.getOrderedUnits(units[pid]):
#    spikes = units[pid][u]['times']
#    respMat = np.zeros([psthSize, xpos.size, ypos.size, ori.size])
#    for trialType in np.unique(sweep_order):
#        this_trial_starts = rf_trial_start_times[sweep_order==trialType]
#        this_trial_psth = makePSTH(spikes, this_trial_starts-preTime, preTime+postTime, binSize)
#        this_trial_params = sweep_table[trialType]
#
#        xInd = np.where(xpos==this_trial_params[0][0])
#        yInd = np.where(ypos==this_trial_params[0][1])
#        oriInd = np.where(ori==this_trial_params[3])
#        
#        respMat[:, xInd, yInd, oriInd] = this_trial_psth[:, None, None]     
#        
#        
#    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]
#    fig, ax = plt.subplots(ypos.size, xpos.size)
#    fig.suptitle(u)
#    for x in np.arange(xpos.size):
#        for y in np.arange(ypos.size):
#            ax[y,x].plot(respMat[:, x, y, bestOri], 'k')
#            ax[y,x].set_ylim([0, respMat.max()])
#            ax[y,x].axis('off')
            

#################################################
###### Analyze running and licking ##############
def get_trial_by_time(times, trial_start_times, trial_end_times):
    trials = []
    for time in times:
        if trial_start_times[0]<=time<trial_end_times[-1]:
            trial = np.where((trial_start_times<=time)&(trial_end_times>time))[0][0]
        else:
            trial = -1
        trials.append(trial)
    
    return np.array(trials)

def find_run_transitions(run_signal, run_time, thresh = [1,5], smooth_kernel = 0.5, inter_run_interval = 2, min_run_duration = 3, sample_freq = 60):
    smooth_kernel = round(smooth_kernel*sample_freq)
    smooth_kernel = smooth_kernel if np.mod(smooth_kernel, 2) == 1 else smooth_kernel + 1 #must be an odd number for median filter
    run_speed_smooth =  scipy.signal.medfilt(run_signal, int(smooth_kernel))
    run_samples = np.where(run_speed_smooth>=thresh[1])[0]
    run_starts = run_samples[np.insert(np.diff(run_samples)>=inter_run_interval*sample_freq, 0, True)]
    
    adjusted_rs = []
    for rs in run_starts:
        last_stat_points = np.where(run_speed_smooth[:rs]<=thresh[0])[0]
        if len(last_stat_points)>0:
            adjusted = (last_stat_points[-1])
        else:
            adjusted = rs
        
        if np.median(run_speed_smooth[adjusted:adjusted+min_run_duration*sample_freq]) > thresh[1]:
            adjusted_rs.append(adjusted)
    
    adjusted_rs = np.array(adjusted_rs).astype(np.int)
    run_start_times = run_time[adjusted_rs]
    
    return run_start_times
     
def plot_lick_triggered_fr(spikes, syncDataset, frameTimes, trials, axis, min_inter_lick_time = 0.5, preTime=1, postTime=2):
    trial_start_frames = np.array(trials['startframe'])
    trial_end_frames = np.array(trials['endframe'])
    trial_start_times = frameTimes[trial_start_frames]
    trial_end_times = frameTimes[trial_end_frames]
    
    lick_times = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
    first_lick_times = lick_times[np.insert(np.diff(lick_times)>=min_inter_lick_time, 0, True)]
    first_lick_trials = get_trial_by_time(first_lick_times, trial_start_times, trial_end_times)
    
    hit = np.array(trials['response_type']=='HIT')
    earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')
    falseAlarm = np.array(trials['response_type']=='FA')
    hit_lick_times = first_lick_times[np.where(hit[first_lick_trials])[0]]
    bad_lick_times = first_lick_times[np.where(falseAlarm[first_lick_trials] | earlyResponse[first_lick_trials])[0]]
   
    hit_psth, t = getSDF(spikes,hit_lick_times-preTime,preTime+postTime)
    bad_psth, t  = getSDF(spikes,bad_lick_times-preTime,preTime+postTime)
    
    hit, = axis.plot(t-1,hit_psth, 'k')
    bad, = axis.plot(t-1, bad_psth, 'r')
    axis.legend((hit, bad), ('hit', 'aborted/FA'), loc='best', prop={'size':8})
    formatFigure(plt.gcf(), axis, xLabel='Time to lick (s)',  yLabel='Lick-Trig. FR (Hz)')
    axis.plot([0,0], axis.get_ylim(), 'k--')
    
    
def plot_run_triggered_fr(spikes, run_start_times, axis, preTime=1, postTime=2):
            
    run_psth, t = getSDF(spikes,run_start_times-preTime,preTime+postTime)
    axis.plot(t-1,run_psth, 'k')
    axis.plot([0,0], axis.get_ylim(), 'k--')
    formatFigure(plt.gcf(), axis, xLabel='Time to run (s)', yLabel='Run-Trig. FR (Hz)')
    
    
    
# saccade aligned sdfs
def plot_saccade_triggered_fr(spikes, eyeData, eyeFrameTimes, axis, preTime=2, postTime=2, sdfSigma=0.02, latThresh=5, minPtsAboveThresh=50):
    
    if eyeData is None:
        print('No eye data')
        return
    pupilArea = eyeData['pupilArea'][:]
    pupilX = eyeData['pupilX'][:]
    negSaccades = eyeData['negSaccades'][:]
    posSaccades = eyeData['posSaccades'][:]    
    
    latFilt = np.ones(minPtsAboveThresh)

    axis.plot([0,0],[0,1000],'k--')
    ymax = 0
    plotlines = []
    for j,(saccades,clr) in enumerate(zip((negSaccades,posSaccades),'rb')):
        saccadeTimes = eyeFrameTimes[saccades]
        sdf,t = getSDF(spikes,saccadeTimes-preTime,preTime+postTime,sigma=sdfSigma)
        plotline, = axis.plot(t-preTime,sdf,clr)
        plotlines.append(plotline)
        ymax = max(ymax,sdf.max())
        z = sdf-sdf[t<1].mean()
        z /= sdf[t<1].std()
        posLat = np.where(np.correlate(z>latThresh,latFilt,mode='valid')==minPtsAboveThresh)[0]
        posLat = posLat[0] if posLat.size>0 else None
        negLat = np.where(np.correlate(z<-latThresh,latFilt,mode='valid')==minPtsAboveThresh)[0]
        negLat = negLat[0] if negLat.size>0 else None
#            posLat = np.where(z[:np.argmax(z)]<latencyThresh)[0][-1]+1 if z.max()>latencyThresh else None
#            negLat = np.where(z[:np.argmin(z)]>-latencyThresh)[0][-1]+1 if z.min()<-latencyThresh else None
        if posLat is not None or negLat is not None:
            if posLat is None:
                latInd = negLat
            elif negLat is None:
                latInd = posLat
            else:
                latInd = min(posLat,negLat)
            axis.plot(t[latInd]-preTime,sdf[latInd],'o',mfc=clr,mec=clr,ms=10)

    for side in ('right','top'):
        axis.spines[side].set_visible(False)
    axis.tick_params(direction='out',top=False,right=False,labelsize=10)
    axis.set_xlim([-preTime,postTime])
    axis.set_ylim([0,1.02*ymax])
    axis.set_xlabel('Time relative to saccade (s)',fontsize=12)
    axis.set_ylabel('Spike/s',fontsize=12)
    axis.legend((plotlines[0], plotlines[1]), ('temporal', 'nasal'), loc='best', prop={'size':8})
    
   

    
        
##########################
#### MAKE SUMMARY PLOT####
        
import matplotlib.gridspec as gridspec

def all_unit_summary(probesToAnalyze, units, dataDir, runSpeed, runTime):
    plt.close('all')
    run_start_times = find_run_transitions(runSpeed, runTime)
    rfstim, pre_blank_frames = get_rf_trial_params(dataDir, None)
    
    for pid in probesToAnalyze:
        multipageObj = PdfPages(os.path.join(dataDir, 'SummaryPlots_' + pid + '.pdf'))
        orderedUnits = probeSync.getOrderedUnits(units[pid])
        for u in orderedUnits:
            plot_unit_summary(pid, u, units, run_start_times, rfstim, pre_blank_frames, multipageObj)
        
#        multipage(os.path.join(dataDir, 'summaryPlots_' + pid + '.pdf'))
#        plt.close('all')
        multipageObj.close()
        
def plot_unit_summary(pid, uid, units, run_start_times, rfstim, pre_blank_frames, multipageObj=None):
    spikes = units[pid][uid]['times']
    fig = plt.figure(facecolor='w', figsize=(16,12))
    if 'ccfRegion' in units[pid][uid] and units[pid][uid]['ccfRegion'] is not None:
        figtitle = 'Probe: ' + str(pid) + ', unit: ' + str(uid) + ' ' + units[pid][uid]['ccfRegion']
    else:
        figtitle = 'Probe: ' + str(pid) + ', unit: ' + str(uid)
        
    fig.suptitle(figtitle)
    
    gs = gridspec.GridSpec(8, 21)
    gs.update(top=0.95, bottom = 0.35, left=0.05, right=0.95, wspace=0.3)
    
    allflashax = plt.subplot(gs[:, :7])
    plot_psth_all_flashes(spikes, frameTimes, core_data, allflashax)
    
    allrespax = plt.subplot(gs[:, 8:14])
    plot_psth_hits_vs_misses(spikes, frameTimes, trials, allrespax, average_across_images=False)
    
    respax = plt.subplot(gs[:4, 15:])
    plot_psth_hits_vs_misses(spikes, frameTimes, trials, respax, preTime = 0.1, postTime = 0.5, sdfSigma=0.005, average_across_images=True)
    
    rfax = plt.subplot(gs[5:, 15:])
    plot_rf(spikes, frameTimes, trials, dataDir, rfax, rfstim, pre_blank_frames, resp_latency=0.05)
    
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=0.25, bottom = 0.05, left=0.05, right=0.95, wspace=0.3)
    
    lickax = plt.subplot(gs2[0, 0])
    plot_lick_triggered_fr(spikes, syncDataset, frameTimes, trials, lickax)
    
    runax = plt.subplot(gs2[0, 1])
    plot_run_triggered_fr(spikes, run_start_times, runax)
    
    saccadeax = plt.subplot(gs2[0,2])
    plot_saccade_triggered_fr(spikes, eyeData, eyeFrameTimes, saccadeax)
    if multipageObj is not None:
        fig.savefig(multipageObj, format='pdf')
    
    plt.close(fig)
    

#####################
##### Behavior summary stuff
min_inter_lick_time = 0.5
lick_times = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
first_lick_times = lick_times[np.insert(np.diff(lick_times)>=min_inter_lick_time, 0, True)]

autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')    
ignore = earlyResponse | autoRewarded
hit = np.array(trials['response_type']=='HIT')

#Histogram of lick times for hit trials
preTime = 1
postTime = 1.5
binsize = 0.01
binsLocations = np.arange(-preTime, postTime+binsize, binsize)
fig, ax = plt.subplots()
for licks,clr in zip((lick_times,first_lick_times),'kr'):
    selectedTrials = hit & (~ignore)
    changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
    psth = makePSTH(licks, changeTimes-preTime, preTime+postTime, binsize)
    ax.bar(np.linspace(-preTime, postTime-binsize, psth.size), psth, binsize, color=clr, alpha=0.5, edgecolor='none')



#######################
####### LFP STUFF
def find_nearest_timepoint(time, timeseries):
    return np.where(timeseries<time)[0][-1]

presTimes = frameTimes[np.array(core_data['visual_stimuli']['frame'])]   
flash_lfp = []
for pres in presTimes:
    index = find_nearest_timepoint(pres, alfp_time)
    flash_lfp.append(alfp[index-2500:index+2500, 220])

flash_lfp = np.array(flash_lfp)
plt.figure()
plt.plot(np.mean(flash_lfp, 0))    
    
pid = 'B'
lfpsig = lfp[pid][0]
lfptime = lfp[pid][1]

autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')    
ignore = earlyResponse | autoRewarded
miss = np.array(trials['response_type']=='MISS')
hit = np.array(trials['response_type']=='HIT')
falseAlarm = np.array(trials['response_type']=='FA')
correctReject = np.array(trials['response_type']=='CR')

cortical_channels = np.unique([units[pid][u]['peakChan'] for u in units[pid] if units[pid][u]['ccfRegion'] is not None and 'VIS' in units[pid][u]['ccfRegion']])
preTime = 5000
postTime = 5000
resp_sxx = []
for resp,clr in zip((hit,miss),'bkrg'):
    selectedTrials = resp & (~ignore)
    changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
#    lfp_changeTimes = np.array([find_nearest_timepoint(t, alfp_time) for t in changeTimes])
    lfp_changeTimes = np.searchsorted(alfp_time.T, changeTimes).flatten()
    indexer = lfp_changeTimes[:, None] + np.arange(-preTime, postTime)
    chan_sxx = []
    for chan in cortical_channels:
        print chan
        resp_lfp = alfp[indexer, chan]
        mean_sub_resp_lfp = resp_lfp - np.mean(resp_lfp, 0)[None,:]
        sxx_all = []
        for r in mean_sub_resp_lfp:
            f, t, sxx = scipy.signal.spectrogram(r, fs=2500, nperseg=2500, noverlap=2490)
            sxx_all.append(sxx)
        chan_sxx.append(np.mean(sxx_all, 0))
    #    resp_lfp = np.zeros(preTime+postTime)
    #    [np.sum([resp_lfp, alfp[l, peakchan]], axis=0, out=resp_lfp) for l in indexer]
    #    resp_lfp /= changeTimes.size
    resp_sxx.append(np.array(chan_sxx))

pretrial_power_hit = np.mean(resp_sxx[0][:, :, 375:400], axis=(0,2))
pretrial_power_miss = np.mean(resp_sxx[1][:, :, 375:400], axis=(0,2))
plt.figure()
plt.plot(f, pretrial_power_hit)
plt.plot(f, pretrial_power_miss, 'k')
plt.figure()
plt.plot(f, pretrial_power_hit/pretrial_power_miss)
    


trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameTimes[trial_start_frames]
trial_end_times = frameTimes[trial_end_frames]

#min_inter_lick_time = 0.05
#lick_times = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
#first_lick_times = lick_times[np.insert(np.diff(lick_times)>=min_inter_lick_time, 0, True)]
#first_lick_trials = get_trial_by_time(first_lick_times, trial_start_times, trial_end_times)

hit = np.array(trials['response_type']=='HIT')
miss = np.array(trials['response_type']=='MISS')
earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')
falseAlarm = np.array(trials['response_type']=='FA')
#hit_lick_times = first_lick_times[np.where(hit[first_lick_trials])[0]]
#bad_lick_times = first_lick_times[np.where(falseAlarm[first_lick_trials] | earlyResponse[first_lick_trials])[0]]

hit_lfps = []
miss_lfps = []
pxx_hits = []
pxx_misses = []

for chan in cortical_channels:
    chanstd = np.std(lfp[pid][0][:, chan])
    
    #exclude dead channels
    if chanstd > 200:
        hit_lfp = get_spike_triggered_lfp(trial_start_times[hit], chan, lfp[pid][0], lfp[pid][1], preTime=2, postTime=0, standardize=False)
        miss_lfp = get_spike_triggered_lfp(trial_start_times[miss], chan, lfp[pid][0], lfp[pid][1], preTime=2, postTime=0, standardize=False)
        
        hit_lfp = scipy.signal.medfilt(hit_lfp, 11)
        miss_lfp = scipy.signal.medfilt(miss_lfp, 11)
        
        f, pxx_hit = scipy.signal.welch(hit_lfp, fs = 2500, nperseg=1250)
        f, pxx_miss = scipy.signal.welch(miss_lfp, fs = 2500, nperseg=1250)
        
        hit_lfps.append(hit_lfp)
        miss_lfps.append(miss_lfp)
        pxx_hits.append(pxx_hit)
        pxx_misses.append(pxx_miss)

maxlfp = np.max(hit_lfps)
fig, ax = plt.subplots()
for ih, hlfp in enumerate(hit_lfps):
    ax.plot(hlfp+ih*maxlfp, 'k')
        

fig, ax = plt.subplots()
ax.plot(f, np.log(np.mean(pxx_hits, 0)/np.mean(pxx_misses,0)))
ax.set_xlim([0, 100])


hit_lfp_times = np.array([find_nearest_timepoint(t, alfp_time) for t in hit_lick_times])
hit_lfp = [alfp[lfp_changeTime-2500:lfp_changeTime+2500, 220] for lfp_changeTime in hit_lfp_times]

bad_lfp_times = np.array([find_nearest_timepoint(t, alfp_time) for t in bad_lick_times])
bad_lfp = [alfp[lfp_changeTime-2500:lfp_changeTime+2500, 220] for lfp_changeTime in bad_lfp_times]

hitspect = []
for time in hit_lfp_times:
    lfpseg = alfp[time-10000:time+10000, 221]
    f, t, sxx = scipy.signal.spectrogram(lfpseg, fs=2500, nperseg=5000, noverlap=4999)
    hitspect.append(sxx[:200])

plt.figure()
plt.pcolormesh(t, f[:200], np.mean(hitspect, 0))



plt.figure()
plt.plot(np.linspace(-1, 1, hit_lfp[0].size), np.mean(hit_lfp,0), 'b')
plt.plot(np.linspace(-1, 1, bad_lfp[0].size), np.mean(bad_lfp,0), 'r')

plt.figure()
for il, first_lick in enumerate(first_lick_times):
    plt.plot(lick_times-first_lick, il*np.ones(lick_times.size), '.')
    

# compute spike triggered average of lfp for one unit
def get_spike_triggered_lfp(spikes, peakchan, lfpsig, lfptime, preTime=1, postTime=1, max_spike_num=5000, standardize=True):
    spikes = spikes.flatten()
    if spikes.size > max_spike_num:
        spikes = np.random.choice(spikes, max_spike_num, False)    
    
    preTime *= 2500
    postTime *= 2500    
    
    u_lfpsig = np.copy(lfpsig[:, peakchan])
    if standardize:
        u_lfpsig = u_lfpsig - u_lfpsig.mean()
        u_lfpsig /= np.std(u_lfpsig)
    
    lfp_spike_times = np.searchsorted(lfptime.T, spikes).flatten()
    lfp_spike_times = lfp_spike_times[(lfp_spike_times>preTime) & (lfp_spike_times<lfpsig.shape[0]-postTime)]
    indexer = lfp_spike_times[:, None] + np.arange(-preTime, postTime)
    stlfp = np.zeros(preTime+postTime)
    [np.sum([stlfp, u_lfpsig[l]], axis=0, out=stlfp) for l in indexer]
    
    return stlfp/lfp_spike_times.size
    

spikes = units[pid][u]['times'].flatten()
peakchan = units[pid][u]['peakChan']

stlfp = get_spike_triggered_lfp(spikes, peakchan, lfpsig, lfptime)
plt.figure(u)
plt.plot(stlfp)

regionsToConsider = ['VIS']
engagement_end = 2500
for u in probeSync.getOrderedUnits(units[pid]):
    print u
    spikes = units[pid][u]['times']
    peakchan = units[pid][u]['peakChan']
    region = units[pid][u]['ccfRegion']
    if region is not None and any([r in region for r in regionsToConsider]):
        stlfp = get_spike_triggered_lfp(spikes[spikes<3600], peakchan, lfpsig, lfptime)
        units[pid][u]['spike_triggered_lfp'] = stlfp

        matched_spike_num = np.min([np.sum((spikes<engagement_end)&(spikes>lfptime[0])), np.sum((spikes>engagement_end) & (spikes<3600))])
        for period, name in zip(([lfptime[0], engagement_end], [engagement_end, 3600]), ('engaged_', 'nonengaged_')):
            period_spikes = np.random.choice(spikes[(spikes>period[0]) & (spikes<period[1])], matched_spike_num, False)
            stlfp = get_spike_triggered_lfp(period_spikes, peakchan, lfpsig, lfptime)
            units[pid][u][name+'spike_triggered_lfp'] = stlfp
            
for u in probeSync.getOrderedUnits(units[pid]):
    region = units[pid][u]['ccfRegion']
    if region is not None and any([r in region for r in regionsToConsider]):    
        plt.figure(u)
        plt.plot(units[pid][u]['engaged_spike_triggered_lfp'])
        plt.plot(units[pid][u]['nonengaged_spike_triggered_lfp'])

pratios = []
e_stlfps = []
ne_stlfps = []
for u in probeSync.getOrderedUnits(units[pid]):
    region = units[pid][u]['ccfRegion']
    if region is not None and any([r in region for r in regionsToConsider]):    
        fig, ax = plt.subplots()
        fig.suptitle(str(u))
        estlfp = scipy.signal.medfilt(units[pid][u]['engaged_spike_triggered_lfp'], 11)
        nestlfp = scipy.signal.medfilt(units[pid][u]['nonengaged_spike_triggered_lfp'], 11)

        e_stlfps.append(estlfp)
        ne_stlfps.append(nestlfp)        
        
        f, pxx_e = scipy.signal.welch(estlfp, fs = 2500, nperseg=1250)
        f, pxx_ne = scipy.signal.welch(nestlfp, fs = 2500, nperseg=1250)
        
        pratios.append(np.log(pxx_e/pxx_ne))
        ax.plot(f, np.log(pxx_e/pxx_ne))
        ax.set_xlim([0, 100])

fig, ax = plt.subplots()
ax.plot(f, np.mean(pratios,0),'k')
ax.set_xlim([0,100])

####################################
#### PCA (or some other technique) to understand response patterns
import clust
 
responseTensor = []
responseRegion = []
regionsToConsider = ['LP', 'LD', 'VIS']
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            spikes = units[pid][u]['times']
            sdf, _ = make_psth_all_flashes(spikes, frameTimes, core_data)
            responseTensor.append(sdf)
            responseRegion.append(region)
            
            
            
responseTensor = np.array(responseTensor)

nbytrial = np.reshape(responseTensor, [-1, responseTensor.shape[2]])
nbytrial_standardized = clust.standardizeData(nbytrial)            
#nbytrial_standardized = nbytrial - np.mean(nbytrial[:, :100], 1)[:, None]
#nbytrial_standardized = nbytrial/np.max(np.abs(nbytrial), 1)[:, None]
nbytrial_pca, nbytrial_eVal, nbytrial_eVec = clust.pca(nbytrial_standardized, True)   
timeComponents = nbytrial_eVec.T

plt.figure()
plt.plot(np.dot(nbytrial_eVec, nbytrial_pca[1]))
plt.plot(nbytrial[1])

time = np.linspace(-0.1, 0.5, timeComponents.shape[1])
fig, ax = plt.subplots(5, 1)
flip = [-1, 1, 1, 1, 1]
for i, (e, f) in enumerate(zip(timeComponents[:5], flip)):
    e *= f    
    ax[i].plot(time, e, 'k')
    ax[i].set_yticks([round(e.min(),2), round(e.max(), 2)])
    if i >3:
        formatFigure(fig, ax[i], xLabel='Time to flash (s)', yLabel='PC ' + str(i))
    else:
        formatFigure(fig, ax[i], yLabel='PC ' + str(i))
        ax[i].xaxis.set_visible(False)
        ax[i].spines['bottom'].set_visible(False)
        


ncattrial = np.reshape(responseTensor, [responseTensor.shape[0], -1])
ncattrial_standardized = clust.standardizeData(ncattrial)            
ncattrial_pca, ncattrial_eVal, ncattrial_eVec = clust.pca(ncattrial_standardized)   
timeComponents = ncattrial_eVec.T

fig, ax = plt.subplots(5, 1)
for i, e in enumerate(timeComponents[:5]):
    ax[i].plot(e)         
            
            
from sklearn.decomposition import FastICA, PCA
ica_nbytrial = FastICA(n_components=3)
nbytrial_ica = ica_nbytrial.fit_transform(ncattrial_standardized.T)

fig, ax = plt.subplots(5, 1)
for i, e in enumerate(nbytrial_ica.T):
    ax[i].plot(-e)     
            
pca = PCA(n_components=3)
nbytrial_pca = pca.fit_transform(ncattrial_standardized.T)     

fig, ax = plt.subplots(5, 1)
for i, e in enumerate(nbytrial_pca.T):
    ax[i].plot(e)   
    
    
    
    
    
    
early_late_ratios = []
sustained = []
latencies = []
sdfs_all = []
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            spikes = units[pid][u]['times']
            sdfs, latency = make_psth_all_flashes(spikes, frameTimes, core_data)            
            latencies.extend(latency)
            
            sdfs_all.extend(sdfs)
        
            for s in sdfs:
                early_late_ratios.append(np.mean(s[150:250])/np.mean(s[275:375]))
                sustained.append(np.mean(s[150:375])/np.max(s[150:375]))

latencies = np.array(latencies) - 100
sustained = np.array(sustained)
sdfs_all = np.array(sdfs_all)

good_resps = (~np.isnan(latencies) & (latencies<100))
fig, ax = plt.subplots()
ax.plot(latencies[good_resps], sustained[good_resps], 'ko')

formatFigure(fig, ax, xLabel='Latency (ms)', yLabel='Sustained Index')
            
            
            