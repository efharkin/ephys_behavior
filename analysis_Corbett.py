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
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from analysis_utils import *

probes_to_run = ('A', 'B', 'C')

    
#Make summary pdf of unit responses    


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
   
       
def make_psth_all_flashes(spikes, frameTimes, core_data, preTime = 0.1, postTime = 0.5, sdfSigma=0.005):
    image_flash_times = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
    image_id = np.array(obj.core_data['visual_stimuli']['image_name'])
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
    
#sl = []
#regionsToConsider = ['VIS']
#for u in probeSync.getOrderedUnits(units[pid]):
#    spikes = units[pid][u]['times']
#    slu = compute_lifetime_sparseness(spikes, frameTimes, core_data)     
#    units[pid][u]['lifetime_sparseness'] = slu
#
#    region = units[pid][u]['ccfRegion']    
#    if region is not None and any([r in region for r in regionsToConsider]):
#        sl.append(slu)
#
#plt.figure()
#plt.hist(sl)

def plot_spike_amplitudes(pid, uid, axis):
    
    spikes = units[pid][uid]['times']
    amplitudes = units[pid][uid]['amplitudes']    
    
    num_spikes_to_plot = 1000.
    if spikes.size>num_spikes_to_plot:
        num_spikes_to_skip = int(spikes.size/num_spikes_to_plot)
    else:
        num_spikes_to_skip = 1
        
    axis.plot(amplitudes[::num_spikes_to_skip], frameTimes[-1] - spikes[::num_spikes_to_skip], 'ko', alpha=0.2)
    
    last_behavior_time = frameTimes[trials['endframe'].values[-1]]    
    
    axis.set_ylim([0, int(frameTimes[-1])])
    axis.set_yticks([0, int(frameTimes[-1])])
    axis.set_yticklabels([int(frameTimes[-1]), 0])
    axis.plot([axis.get_xlim()[0], axis.get_xlim()[1]], [frameTimes[-1]-last_behavior_time, frameTimes[-1]-last_behavior_time], 'k--')
    formatFigure(plt.gcf(), axis, xLabel='Template Scale Factor', yLabel='Time (s)')
    axis.yaxis.labelpad = -20    

def plot_spike_template(pid, uid, gs=None):
    if gs is None:
        gs = gridspec.GridSpec(3,3)
    template = units[pid][uid]['template']
    tempax = plt.subplot(gs[:, 0])    
    tempax.imshow(template.T, cmap='gray', origin='lower')
    tempax.set_xlim([0, 80])
    tempax.set_xticks([0, 80])
    tempax.set_xticklabels([0, 2.67]) 
    tempax.set_xlabel('Time (ms)')
    tempax.set_ylabel('Channel')
    
    temp_channels = np.where(template>0)
    first_chan = temp_channels[1].min()
    last_chan = temp_channels[1].max()
    temp_inset = template[:, first_chan:last_chan]
    insetax = plt.subplot(gs[0, 1])
    insetax.patch.set_alpha(0.0)
    insetax.imshow(temp_inset.T, cmap='gray', origin='lower')    
    insetax.set_yticks([0, temp_inset.shape[1]])
    insetax.set_yticklabels([first_chan, last_chan])
    insetax.set_xlim([0, 80])
    insetax.set_xticks([0, 80])
    insetax.set_xticklabels([0, 2.67])    
    
    peak_channel_waveform = template[:, units[pid][uid]['peakChan']]
    peakchanax = plt.subplot(gs[1,1])
    peakchanax.plot(peak_channel_waveform, 'k')
    peakchanax.set_xlim([0, 80])
    peakchanax.set_xticks([0, 80])
    peakchanax.set_xticklabels([0, 2.67])     
    
    spike_times = units[pid][uid]['times']
    d = []                   # Distance between any two spike times
    n_sp = len(spike_times)  # Number of spikes in the input spike train
    
    bin_width=0.0005
    width=0.1
    i, j = 0, 0
    for t in spike_times:
        # For each spike we only consider those spikes times that are at most
        # at a 'width' time lag. This requires finding the indices
        # associated with the limiting spikes.
        while i < n_sp and spike_times[i] < t - width:
            i += 1
        while j < n_sp and spike_times[j] < t + width:
            j += 1
        # Once the relevant spikes are found, add the time differences
        # to the list
        d.extend(spike_times[i:j] - t)
        

    n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
    # Define the edges of the bins (including rightmost bin)
    b = np.linspace(-width, width, 2 * n_b, endpoint=True)
    [h, hb] = np.histogram(d, bins=b)
    h[np.ceil(len(h)/2).astype(int) - 1] = 0
                    
    acgax = plt.subplot(gs[2,1])
    acgax.bar(hb[:-1], h, bin_width)
    acgax.set_xlim([-width,width])
    acgax.spines['right'].set_visible(False)
    acgax.spines['top'].set_visible(False)
    acgax.tick_params(direction='out',top=False,right=False)
    acgax.set_yticks([0, h.max()])
    acgax.set_xlabel('Time (s)')
    acgax.set_ylabel('Spike count')
    
    

def plot_psth_hits_vs_misses(spikes, frameTimes, trials, axis, preTime = 1.5, postTime = 1.5, sdfSigma=0.02, average_across_images=True):
    autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
    earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')    
    ignore = earlyResponse | autoRewarded
    miss = np.array(trials['response_type']=='MISS')
    hit = np.array(trials['response_type']=='HIT')
    falseAlarm = np.array(trials['response_type']=='FA')
    correctReject = np.array(trials['response_type']=='CR')
    ymax = 0
    
    respsToAnalyze = [hit, miss]
    if average_across_images:
        plotlines = []
        for resp,clr in zip(respsToAnalyze,'bkrg'):
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
        
def all_unit_summary(probesToAnalyze, units, dataDir, runSpeed, runTime, name_tag = ''):
    plt.close('all')
    run_start_times = find_run_transitions(runSpeed, runTime)
    rfstim, pre_blank_frames = get_rf_trial_params(dataDir, None)
    
    for pid in probesToAnalyze:
        multipageObj = PdfPages(os.path.join(dataDir, 'SummaryPlots_' + pid + name_tag + '.pdf'))
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
    plot_psth_all_flashes(spikes, frameTimes, core_data, allflashax, preTime = 0.1, postTime = 0.75)
    
    allrespax = plt.subplot(gs[:, 8:14])
    plot_psth_hits_vs_misses(spikes, frameTimes, trials, allrespax, preTime = 1.5, postTime = 3, average_across_images=False)
    
    respax = plt.subplot(gs[:4, 15:])
    plot_psth_hits_vs_misses(spikes, frameTimes, trials, respax, preTime = 0.1, postTime = 2, sdfSigma=0.005, average_across_images=True)
    
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
    

def plot_unit_behavior_summary(pid, uid, multipageObj=None):
    spikes = units[pid][uid]['times']    
    
    fig = plt.figure(facecolor='w', figsize=(16,12))
    if 'ccfRegion' in units[pid][uid] and units[pid][uid]['ccfRegion'] is not None:
        figtitle = 'Probe: ' + str(pid) + ', unit: ' + str(uid) + ' ' + units[pid][uid]['ccfRegion']
    else:
        figtitle = 'Probe: ' + str(pid) + ', unit: ' + str(uid)
        
    fig.suptitle(figtitle)    
    
    gs_waveform = gridspec.GridSpec(3, 3)
    gs_waveform.update(top=0.95, bottom = 0.35, left=0.05, right=0.95, wspace=0.3)

    plot_spike_template(pid, uid, gs=gs_waveform)
    
    amp_ax = plt.subplot(gs_waveform[:, 2])
    plot_spike_amplitudes(pid, uid, amp_ax)
    
    gs_behavior = gridspec.GridSpec(1, 3)
    gs_behavior.update(top=0.25, bottom = 0.05, left=0.05, right=0.95, wspace=0.3)
    
    lickax = plt.subplot(gs_behavior[0, 0])
    plot_lick_triggered_fr(spikes, syncDataset, frameTimes, trials, lickax)
    
    runax = plt.subplot(gs_behavior[0, 1])
    plot_run_triggered_fr(spikes, run_start_times, runax)
    
    saccadeax = plt.subplot(gs_behavior[0,2])
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
preTime = 0.5
postTime = 1.5
binsize = 0.02
binsLocations = np.arange(-preTime, postTime+binsize, binsize)
fig, ax = plt.subplots()
for licks,clr in zip((lick_times,first_lick_times),'kr'):
    selectedTrials = hit & (~ignore)
    changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
    psth = makePSTH(licks, changeTimes-preTime, preTime+postTime, binsize)
    ax.bar(np.linspace(-preTime, postTime-binsize, psth.size), psth, binsize, color=clr, edgecolor='none')

formatFigure(fig, ax, xLabel='Time to change (s)', yLabel='Number of licks')

selectedTrials = miss & (~ignore)
changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
changeImages = np.array(trials['change_image_name'][selectedTrials])
initialImages = np.array(trials['initial_image_name'][selectedTrials])

changeCounts = []
initialCounts = []
for ir, respType in enumerate((hit, miss)):
    selectedTrials = respType & (~ignore)
    changeImages = np.array(trials['change_image_name'][selectedTrials])
    initialImages = np.array(trials['initial_image_name'][selectedTrials])

    changeCounts.append([np.sum(changeImages==im) for im in np.unique(changeImages)])
    initialCounts.append([np.sum(initialImages==im) for im in np.unique(initialImages)])

changeCounts = np.array(changeCounts)
initialCounts = np.array(initialCounts)
    
changeTotals = np.sum(changeCounts, 0).astype(np.float)
initialTotals = np.sum(initialCounts,0).astype(np.float)

change_image_hit_rates = changeCounts[0]/changeTotals

fig, ax = plt.subplots()
ax.plot(changeCounts[0]/changeTotals)
ax.plot(changeCounts[1]/changeTotals, 'k')

fig, ax = plt.subplots()
ax.plot(initialCounts[0]/initialTotals)
ax.plot(initialCounts[1]/initialTotals, 'k')

#######################
####### LFP STUFF
def find_nearest_timepoint(time, timeseries):
    return np.where(timeseries<time)[0][-1]
    
def notch_filter(signal, notch_freq=60, sample_freq=2500, bandwidth=2):
    nfb, nfa = scipy.signal.iirnotch(notch_freq/float(sample_freq/2), notch_freq/bandwidth)
    signal_filt = scipy.signal.filtfilt(nfb, nfa, signal)
    
    return signal_filt
    

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
    lfp_changeTimes = np.searchsorted(lfptime.T, changeTimes).flatten()
    indexer = lfp_changeTimes[:, None] + np.arange(-preTime, postTime)
    chan_sxx = []
    for chan in cortical_channels:
        print chan
        resp_lfp = lfpsig[indexer, chan]
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
    


#min_inter_lick_time = 0.05
#lick_times = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
#first_lick_times = lick_times[np.insert(np.diff(lick_times)>=min_inter_lick_time, 0, True)]
#first_lick_trials = get_trial_by_time(first_lick_times, trial_start_times, trial_end_times)

autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')    
ignore = earlyResponse | autoRewarded
miss = np.array(trials['response_type']=='MISS')
hit = np.array(trials['response_type']=='HIT')
falseAlarm = np.array(trials['response_type']=='FA')
correctReject = np.array(trials['response_type']=='CR')
#hit_lick_times = first_lick_times[np.where(hit[first_lick_trials])[0]]
#bad_lick_times = first_lick_times[np.where(falseAlarm[first_lick_trials] | earlyResponse[first_lick_trials])[0]]

hit_lfps = []
miss_lfps = []
pxx_hits = []
pxx_misses = []
preTime = 2
postTime = 2
for chan in cortical_channels:
    chanstd = np.std(lfp[pid][0][:, chan])
    
    #exclude dead channels
    if chanstd > 400:
        hit_changes = frameTimes[np.array(trials['change_frame'][hit & ~ignore]).astype(int)]
        miss_changes = frameTimes[np.array(trials['change_frame'][miss & ~ignore]).astype(int)]
        hit_lfp = get_spike_triggered_lfp(hit_changes, chan, lfp[pid][0], lfp[pid][1], preTime=preTime, postTime=postTime, standardize=False)
        miss_lfp = get_spike_triggered_lfp(miss_changes, chan, lfp[pid][0], lfp[pid][1], preTime=preTime, postTime=postTime, standardize=False)
        
        hit_lfp = scipy.signal.medfilt(hit_lfp, 11)
        miss_lfp = scipy.signal.medfilt(miss_lfp, 11)
        
        f, pxx_hit = scipy.signal.welch(hit_lfp, fs = 2500, nperseg=1250)
        f, pxx_miss = scipy.signal.welch(miss_lfp, fs = 2500, nperseg=1250)
        
        hit_lfps.append(hit_lfp)
        miss_lfps.append(miss_lfp)
        pxx_hits.append(pxx_hit)
        pxx_misses.append(pxx_miss)

time = np.linspace(-preTime, postTime, hit_lfps[0].size)

maxlfp = np.max(hit_lfps)
fig, ax = plt.subplots()
for ih, hlfp in enumerate(hit_lfps):
    ax.plot(hlfp+ih*maxlfp, 'k')

hit_lfps = np.array(hit_lfps)
miss_lfps = np.array(miss_lfps)
fig, ax = plt.subplots()
ax.plot(time, np.mean(hit_lfps, 0) - np.mean(hit_lfps[:, :int(preTime*2500)]), 'b')
ax.plot(time, np.mean(miss_lfps, 0) - np.mean(miss_lfps[:, :int(preTime*2500)]), 'k')        

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
#    peakchan = units[pid][u]['peakChan']
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
        plt.plot(units[pid][u]['engaged_spike_triggered_lfp'] - units[pid][u]['engaged_spike_triggered_lfp'].mean())
        plt.plot(units[pid][u]['nonengaged_spike_triggered_lfp'] - units[pid][u]['nonengaged_spike_triggered_lfp'].mean())

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
        
        estlfp_filt = notch_filter(estlfp)
        nestlfp_filt = notch_filter(nestlfp)

        e_stlfps.append(estlfp_filt)
        ne_stlfps.append(nestlfp_filt)        
        
        f, pxx_e = scipy.signal.welch(estlfp_filt, fs = 2500, nperseg=2500)
        f, pxx_ne = scipy.signal.welch(nestlfp_filt, fs = 2500, nperseg=2500)
        
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
regionsToConsider = ['LP', 'LD', 'VIS', 'cc']
preTime = 0.1
postTime = 0.5
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            spikes = units[pid][u]['times']
            sdf, _ = make_psth_all_flashes(spikes, frameTimes, core_data, preTime=preTime, postTime=postTime)
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

time = np.linspace(-preTime, postTime, timeComponents.shape[1])
fig, ax = plt.subplots(5, 1)
flip = [-1, -1, -1, -1, 1]
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





preTime = 0.1
postTime = 0.8

respLatency = 0.05
epochs = (1000*np.array([[preTime + respLatency, preTime + respLatency + 0.1], [preTime + 0.2, preTime + 0.3], [preTime+0.45, preTime+0.65]])).astype(np.int)
meanRates = []
uid = []
all_sdfs = []
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            uid.append(pid+str(u))
            spikes = units[pid][u]['times']
                    
            sdfs, latency = make_psth_all_flashes(spikes, frameTimes, core_data, preTime=preTime, postTime = postTime)            
            baseline = np.mean(sdfs[:, 0:epochs[0][0]])
            all_sdfs.append(sdfs)
            for s in sdfs:
                meanRate = []
                for epoch in epochs:
                    meanRate.append(np.max(s[epoch[0]:epoch[1]]) - baseline)
                meanRates.append(meanRate)

meanRates = np.array(meanRates)
all_sdfs = np.array(all_sdfs)

meanRates_byCell = np.reshape(meanRates, (-1, 8, 3))
rs = []
upper_inds = np.triu_indices(3,1)
for mr in meanRates_byCell:
    rs.append(np.corrcoef(mr.T)[upper_inds])

labels = ['onset vs late', 'onset vs post stim', 'late vs post stim']
for r, label in zip(rs.T, labels):   
    plt.figure(label)
    plt.hist(r)    


#####################################
###### classify image identity ######

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
obj = b
ignore = obj.ignore
frameTimes = obj.frameAppearTimes
trials = obj.trials
units = obj.units
selectOnRegion = False

probes_to_run = ['C']
clf = RandomForestClassifier(1000, min_samples_split=5)

#selectedTrials = miss & (~ignore)
selectedTrials = ~ignore
changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
changeImages = np.array(trials['change_image_name'][selectedTrials])
initialImages = np.array(trials['initial_image_name'][selectedTrials])

#preTime = 1
#postTime = 1
#all_sdfs = []
#uid = []
#for pid in probes_to_run:
#    for u in probeSync.getOrderedUnits(units[pid]):
#        region = units[pid][u]['ccfRegion']
#        if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
#            spikes = units[pid][u]['times']
#            uid.append(pid+str(u))
#            sdfs,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=0.02,sampInt=0.001,avg=False)
#            if len(all_sdfs) == 0:
#                all_sdfs = np.copy(sdfs)
#                
#            else:
#                all_sdfs = np.hstack([all_sdfs, sdfs])
#
#
#
#clf.fit(all_sdfs, changeImages)
#f_importance_miss = clf.feature_importances_
#
#plt.figure()
#plt.plot(np.mean(np.reshape(f_importance_miss, [59, -1]), 0))
#
#for i, fi in enumerate(np.reshape(f_importance_miss, [59, -1])):
#    plt.figure(uid[i])
#    plt.plot(fi)
    

# same as above using psths not sdfs
preTime = 0.5
postTime = 0.5
all_psths = []
uid = []
baselines = []
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
            spikes = units[pid][u]['times']
            uid.append(pid+str(u))
            psth = makePSTH(spikes, changeTimes-preTime, preTime+postTime, binSize=0.01, avg=False)
            if len(all_psths) == 0:
                all_psths = np.copy(psth)
#                baselines = np.copy(psth[:, 530:550])
                
            else:
                all_psths = np.hstack([all_psths, psth])
#                baselines = np.hstack([baselines, psth[:, 300:320]])

psth_train, psth_test, image_train, image_test = train_test_split(all_psths, changeImages, test_size=0.5)
clf.fit(psth_train, image_train)
score = clf.score(psth_test, image_test)
print(score)
f_psth_importance = clf.feature_importances_

plt.figure()
plt.plot(np.mean(np.reshape(f_psth_importance, [len(uid), -1]), 0))

#for i, fi in enumerate(np.reshape(f_psth_importance, [len(uid), -1])):
#    plt.figure(uid[i])
#    plt.plot(fi)


# classify on several postTime intervals to "simulate" cortical silencing experiment
scores = []
startTime = 0
endTime = 0.5
timeStep = 0.01
for postTime in np.arange(startTime, endTime, timeStep):
    preTime = 0.5
    all_psths = []
    uid = []
    baselines = []
    for pid in probes_to_run:
        for u in probeSync.getOrderedUnits(units[pid]):
            region = units[pid][u]['ccfRegion']
            if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                spikes = units[pid][u]['times']
                uid.append(pid+str(u))
                psth = makePSTH(spikes, changeTimes-preTime, preTime+postTime, binSize=0.01, avg=False)
                if len(all_psths) == 0:
                    all_psths = np.copy(psth)
                    baselines = np.copy(psth[:, 530:550])
                    
                else:
                    all_psths = np.hstack([all_psths, psth])
                    baselines = np.hstack([baselines, psth[:, 300:320]])
    
    psth_train, psth_test, image_train, image_test = train_test_split(all_psths, changeImages, test_size=0.2)
    clf.fit(psth_train, image_train)
    scores.append(clf.score(psth_test, image_test))
    
fig, ax = plt.subplots()
ax.plot(np.arange(startTime, endTime, timeStep), scores, 'ko-')
ax.vlines(0.08, 0, 1, 'k')
   
ap = np.reshape(all_psths, [len(all_psths), len(uid), -1])
ap_mean = np.mean(ap, axis=(0,1))
ap_mean /= ap_mean.max()
ax.plot(np.arange(startTime, endTime, timeStep)[1:], ap_mean[int(preTime*100):], 'k')
    
    
    
            
# same as above predicting image before change
preTime = 2
postTime = 2
all_psths = []
uid = []
baselines = []
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            spikes = units[pid][u]['times']
            uid.append(pid+str(u))
            psth = makePSTH(spikes, changeTimes-preTime, preTime+postTime, binSize=0.01, avg=False)
            if len(all_psths) == 0:
                all_psths = np.copy(psth)
                baselines = np.copy(psth[:, 180:200])
                
            else:
                all_psths = np.hstack([all_psths, psth])
                baselines = np.hstack([baselines, psth[:, 180:200]])


psth_train, psth_test, image_train, image_test = train_test_split(all_psths, changeImages, test_size=0.5)
clf.fit(psth_train, image_train)
score = clf.score(psth_test, image_test)
print(score)
f_psth_importance = clf.feature_importances_

plt.figure()
plt.plot(np.mean(np.reshape(f_psth_importance, [len(uid), -1]), 0))

for i, fi in enumerate(np.reshape(f_psth_importance, [len(uid), -1])):
    plt.figure(uid[i])
    plt.plot(fi)
            
            
            
            
            
preTime = 5
postTime = 5          
all_psths = [[], []]    
change_response_ratio = [[],[]]    
for pid in probes_to_run:
    for u in probeSync.getOrderedUnits(units[pid]):
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            spikes = units[pid][u]['times']
            uid.append(pid+str(u))

            for ir, respType in enumerate((hit, miss)):
                selectedTrials = respType & (~ignore)  
                changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
                psth = makePSTH(spikes, changeTimes-preTime, preTime+postTime, binSize=0.01)
                all_psths[ir].append(psth)
                change_response_ratio[ir].append(np.max(psth[505:530])/np.max(psth[430:455]))
all_psths = np.array(all_psths)
change_response_ratio = np.array(change_response_ratio)

fig, ax = plt.subplots()
ax.plot(np.mean(all_psths[0], 0), 'b')
ax.plot(np.mean(all_psths[1], 0), 'k')

fig, ax = plt.subplots()
ax.hist()

######Correlate pop response to hit rate##########

#get hit rates for each image
hit = obj.hit
miss = obj.miss
changeCounts = []
initialCounts = []
for ir, respType in enumerate((hit, miss)):
    selectedTrials = respType & (~ignore)
    changeImages = np.array(trials['change_image_name'][selectedTrials])
    initialImages = np.array(trials['initial_image_name'][selectedTrials])

    changeCounts.append([np.sum(changeImages==im) for im in np.unique(trials['change_image_name'])])
    initialCounts.append([np.sum(initialImages==im) for im in np.unique(trials['change_image_name'])])

changeCounts = np.array(changeCounts)
initialCounts = np.array(initialCounts)
    
changeTotals = np.sum(changeCounts, 0).astype(np.float)
initialTotals = np.sum(initialCounts,0).astype(np.float)

change_image_hit_rates = changeCounts[0]/changeTotals


#calculate pop response to each image
regionsToConsider = ['VIS', 'cc']
probes_to_run = ['C']
uid = []
image_sdfs = []   
preTime = 0.1
postTime = 0.5
allFlashes = True  #if True pop response will be calculated using all flashes of stimulus, if false only change flashes will be used
if allFlashes:
    for pid in probes_to_run:
        for u in probeSync.getOrderedUnits(units[pid]):
            region = units[pid][u]['ccfRegion']
            if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                spikes = units[pid][u]['times']
                uid.append(pid+str(u))
                sdfs, time = make_psth_all_flashes(spikes, frameTimes, obj.core_data, preTime = preTime, postTime = postTime, sdfSigma=0.005)
                image_sdfs.append(sdfs)
                
    meanImageResps = np.mean(image_sdfs, 0)            
    latency = np.array([find_latency(s, stdev_thresh=5) for s in meanImageResps]) - 1000*preTime

else:
    changeImages = np.array(trials['change_image_name'][~ignore])
    changeTimes = frameTimes[np.array(trials['change_frame'][~ignore]).astype(int)] 
    for pid in probes_to_run:
        for u in probeSync.getOrderedUnits(units[pid]):
            region = units[pid][u]['ccfRegion']
            if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                spikes = units[pid][u]['times']
                uid.append(pid+str(u))
                unit_sdf = []
                for ii, im in enumerate(np.unique(changeImages)):                    
                    theseTimes = changeTimes[changeImages==im]
                    sdf, time = getSDF(spikes, theseTimes-preTime, preTime+postTime, sigma=0.005)
                    unit_sdf.append(sdf)
                image_sdfs.append(unit_sdf)
                
    meanImageResps = np.mean(image_sdfs, 0)            
    latency = np.array([find_latency(s, stdev_thresh=5) for s in meanImageResps]) - 1000*preTime

fig, ax = plt.subplots()
fig_in, ax_in = plt.subplots()
for i, (s, l) in enumerate(zip(meanImageResps, latency.astype(np.int) + int(1000*preTime))):
    ax.plot(s[100:200])
    #ax.plot(l-100, s[100:200][l-100], 'ro')    
    ax.set_ylim([0, 30])
    
    ax_in.plot(s)
    #ax_in.plot(l, s[l], 'ro')    
    ax_in.set_ylim([0, 30])
    


#plot correlation between pop response and hit rate
early_response = np.mean(meanImageResps[:, 125:200], 1)
full_response = np.mean(meanImageResps[:, 130:400], 1)

fig, ax = plt.subplots()
ax.plot(early_response, change_image_hit_rates, 'ko')
formatFigure(fig, ax, xLabel='Early response magnitude (sp/s)', yLabel='Hit rate')

fig, ax = plt.subplots()
ax.plot(latency, change_image_hit_rates, 'ko')
formatFigure(fig, ax, xLabel='Response latency (ms)', yLabel='Hit rate')

fig, ax = plt.subplots()
ax.plot(full_response, change_image_hit_rates, 'ko')
formatFigure(fig, ax, xLabel='Mean response magnitude (sp/s)', yLabel='Hit rate')

fig, ax = plt.subplots()
for tp in np.arange(meanImageResps.shape[1]):
    corr = scipy.stats.linregress(np.mean(meanImageResps[:, :tp], 1), change_image_hit_rates)[2]
    ax.plot(tp, corr, 'ko')
    
    corr = scipy.stats.linregress(meanImageResps[:, tp], change_image_hit_rates)[2]
    ax.plot(tp, corr, 'ro')



#calculate pop response for each probe
probes_to_run = ['A', 'B', 'C', 'F']
uid = []
preTime = 1
postTime = 1
allFlashes = False  #if True pop response will be calculated using all flashes of stimulus, if false only change flashes will be used
meanProbeResponse = []
for pid in probes_to_run:
    image_sdfs = []   
    if allFlashes:
        for u in probeSync.getOrderedUnits(units[pid]):
            region = units[pid][u]['ccfRegion']
            if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                spikes = units[pid][u]['times']
                uid.append(pid+str(u))
                sdfs, time = make_psth_all_flashes(spikes, frameTimes, obj.core_data, preTime = preTime, postTime = postTime, sdfSigma=0.005)
                image_sdfs.append(sdfs)
                    
        meanImageResps = np.mean(image_sdfs, 0)            
        latency = np.array([find_latency(s, stdev_thresh=5) for s in meanImageResps]) - 1000*preTime
    
    else:
        changeImages = np.array(trials['change_image_name'][~ignore])
        changeTimes = frameTimes[np.array(trials['change_frame'][~ignore]).astype(int)] 
        for u in probeSync.getOrderedUnits(units[pid]):
            region = units[pid][u]['ccfRegion']
            if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                spikes = units[pid][u]['times']
                uid.append(pid+str(u))
                unit_sdf = []
                for ii, im in enumerate(np.unique(changeImages)):                    
                    theseTimes = changeTimes[changeImages==im]
                    sdf, time = getSDF(spikes, theseTimes-preTime, preTime+postTime, sigma=0.005)
                    unit_sdf.append(sdf)
                image_sdfs.append(unit_sdf)
                
        meanImageResps = np.mean(image_sdfs, 0)            
        latency = np.array([find_latency(s, stdev_thresh=5) for s in meanImageResps]) - 1000*preTime
        
        meanProbeResponse.append(np.mean(meanImageResps,0))        
        
        fig, ax = plt.subplots()
        fig.suptitle(pid)
        #fig_in, ax_in = plt.subplots()
        for i, (s, l) in enumerate(zip(meanImageResps, latency.astype(np.int) + int(1000*preTime))):
            ax.plot(s)
            #ax.plot(l-100, s[100:200][l-100], 'ro')    
            #ax.set_ylim([0, 30])
            
            #ax_in.plot(s)
            #ax_in.plot(l, s[l], 'ro')    
            #ax_in.set_ylim([0, 30])

meanProbeResponse = np.array(meanProbeResponse)
plt.figure()
colors = ['b', 'g', 'k', 'c', 'y', 'r']
for i, (c, r, p) in enumerate(zip(colors, meanProbeResponse, probes_to_run)):
    r -= r.min()
    r /= r.max()
    plt.plot(r, c)
    plt.text(1, 9-0.3*i, p, color=c)

regionsToConsider=['VIS', 'cc', 'LP', 'LGd']
regionsToConsider=['VIS', 'cc']
unitIDs = np.concatenate([[pid+'_'+str(u) for u in probeSync.getOrderedUnits(units[pid])] for pid in probes_to_run])
ccgs = []
regions = []
u1u2s = []
for ui, uid in enumerate(unitIDs):
    pid1, u1 = uid.split('_')
    u1 = int(u1)
    region1 = units[pid1][u1]['ccfRegion']
    if region1 is not None and any([r in region1 for r in regionsToConsider]):
        spikes = units[pid1][u1]['times']
        for uid2 in unitIDs[ui:]:
            pid2, u2 = uid2.split('_')
            u2 = int(u2)
            region2 = units[pid2][u2]['ccfRegion']
            if region2 is not None and any([r in region2 for r in regionsToConsider]):
                spikes2 = units[pid2][u2]['times']
                ccg, bins = get_ccg(spikes, spikes2)
                ccgs.append(ccg)
                regions.append(region1 + '_' + region2)
                u1u2s.append(uid + '/' + uid2)
    
    

np.save(os.path.join(dataDir, 'ccgs_visctx_thal.npy'), ccgs)
np.save(os.path.join(dataDir, 'ccg_regions.npy'), regions)
np.save(os.path.join(dataDir, 'ccg_unitIDs.npy'), u1u2s)



source = 'VISp'
target = 'VISp'
source_ccgs = []
source_u1u2 = []
targetIDs = []
for (region, u1u2, ccg) in zip(regions, u1u2s, ccgs):
    r1, r2 = region.split('_')
#    if (any([source in r for r in (r1,r2)])) and ((target is None) or any([target in r for r in (r1,r2)])): 
    if ((source in r1) and (target in r2)) or ((source in r2) and (target in r1)):
        u1, u2 = u1u2.split('/')
        if u1 != u2:
            if source in r1:
                source_ccgs.append(ccg)
                source_u1u2.append(u1u2)
                targetIDs.append(r2)
            elif source in r2:
                source_ccgs.append(ccg[::-1])
                u1, u2 = u1u2.split('/')
                source_u1u2.append(u2 + '/' + u1)
                targetIDs.append(r1)
        
source_ccgs = np.array(source_ccgs)
plt.figure()
plt.plot(np.mean(source_ccgs, 0))

#for source_unit in unitIDs:
for source_unit in inLP:
#source_unit = 'A_98'
    target = 'VISp'
    source_ccgs = []
    source_u1u2 = []
    targetIDs = []
    for (region, u1u2, ccg) in zip(regions, u1u2s, ccgs):
        r1, r2 = region.split('_')
        u1, u2 = u1u2.split('/')
        if u1 != u2:
            if (any([source_unit in u for u in (u1, u2)])) and ((target is None) or any([target in r for r in (r1,r2)])): 
                if source_unit in u1:
                    source_ccgs.append(ccg)
                    source_u1u2.append(u1u2)
                    targetIDs.append(r2)
                elif source_unit in u2:
                    source_ccgs.append(ccg[::-1])
                    source_u1u2.append(u2 + '/' + u1)
                    targetIDs.append(r1)
                
    source_ccgs = np.array(source_ccgs)
    fig, ax = plt.subplots()
#    minind = np.unravel_index(np.argmin(source_ccgs), source_ccgs.shape)
    ax.plot(np.mean(source_ccgs, 0))
    fig.suptitle(source_unit)




#### compare FS and RS pop response to change stimulus
regionsToConsider = ['VIS', 'cc']
probes_to_run = ['A', 'B', 'C']
preTime = 1
postTime = 1.5
allFlashes = False  #if True pop response will be calculated using all flashes of stimulus, if false only change flashes will be used
ptCriterion = lambda pt: pt<0.35

if allFlashes:
    for pid in probes_to_run:
        image_sdfs = []
        for u in probeSync.getOrderedUnits(units[pid]):
            region = units[pid][u]['ccfRegion']
            if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                spikes = units[pid][u]['times']
                uid.append(pid+str(u))
                sdfs, time = make_psth_all_flashes(spikes, frameTimes, obj.core_data, preTime = preTime, postTime = postTime, sdfSigma=0.005)
                image_sdfs.append(sdfs)
                
        meanImageResps = np.mean(image_sdfs, 0)            
        latency = np.array([find_latency(s, stdev_thresh=5) for s in meanImageResps]) - 1000*preTime

else:
    changeImages = np.array(trials['change_image_name'][~ignore])
    changeTimes = frameTimes[np.array(trials['change_frame'][~ignore]).astype(int)] 
    for pid in probes_to_run:
        image_sdfs = []
        for u in probeSync.getOrderedUnits(units[pid]):
            ptot = units[pid][u]['peakToTrough']
            if ptCriterion(ptot) and units[pid][u]['ccfRegion'] is not 'hipp':
                print(pid + ': ' + str(u))
                region = units[pid][u]['ccfRegion']            
                if not selectOnRegion or (region is not None and any([r in region for r in regionsToConsider])):
                    spikes = units[pid][u]['times']
                    uid.append(pid+str(u))
                    unit_sdf = []
                    for ii, im in enumerate(np.unique(changeImages)):                    
                        theseTimes = changeTimes[changeImages==im]
                        sdf, time = getSDF(spikes, theseTimes-preTime, preTime+postTime, sigma=0.005)
                        unit_sdf.append(sdf)
                    image_sdfs.append(unit_sdf)
        
        image_sdfs = np.array(image_sdfs)
        image_sdfs_across_images = np.mean(image_sdfs, 1)
        figim, axim = plt.subplots()
        figim.suptitle(pid)
        axim.imshow(image_sdfs_across_images, aspect='auto', cmap='plasma')
        print('Num cells in ' + pid + ': ' + str(len(image_sdfs)))
        meanImageResps = np.mean(image_sdfs, 0)            
        latency = np.array([find_latency(s, stdev_thresh=5) for s in meanImageResps]) - 1000*preTime
        fig, ax = plt.subplots()
        fig.suptitle(pid)
        ax.plot(meanImageResps.T)

















