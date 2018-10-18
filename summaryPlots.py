# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:30:18 2018

@author: svc_ccg
"""

from __future__ import division
import os
import probeSync, analysis_utils
import numpy as np
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
    fig = plt.figure(facecolor='w', figsize=(16,10))
    figtitle = 'Probe ' + str(pid) + ', unit ' + str(uid)
    if 'ccfRegion' in units[pid][uid] and units[pid][uid]['ccfRegion'] is not None:
        figtitle += ' , ' + units[pid][uid]['ccfRegion']
    fig.suptitle(figtitle)
    
    gs = gridspec.GridSpec(8, 8)
    gs.update(top=0.95, bottom = 0.35, left=0.05, right=0.95, wspace=0.3)
    
    imgAxes = [plt.subplot(gs[i,0]) for i in range(8)]
    plot_images(imgAxes)
    
    rfAxes = [plt.subplot(gs[i,1]) for i in range(8)]
    plot_rf(spikes, rfAxes, resp_latency=0.05)
    
    allflashax = [plt.subplot(gs[i,2:4]) for i in range(8)]
    plot_psth_all_flashes(spikes, allflashax)
    
    hitmissax = [plt.subplot(gs[i,4:]) for i in range(8)]
    plot_psth_hits_vs_misses(spikes, hitmissax)
    
    if multipageObj is not None:
        fig.savefig(multipageObj, format='pdf')
    
    plt.close(fig)


def plot_images(axes):
    for ax,img,imname in zip(axes,imagesDownsampled,imageNames):
        ax.imshow(img,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(imname,fontsize=10)


def plot_rf(spikes, axes=None, resp_latency=0.05):
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
    
    trial_spikes = analysis_utils.find_spikes_per_trial(spikes, rf_trial_start_times+resp_latency, rf_trial_start_times+resp_latency+0.2)
    respMat = np.zeros([ypos.size, xpos.size, ori.size])
    for (y, x, o, tspikes) in zip(trial_ypos, trial_xpos, trial_ori, trial_spikes):
        respInd = tuple([np.where(ypos==y)[0][0], np.where(xpos==x)[0][0], np.where(ori==o)[0][0]])
        respMat[respInd] += tspikes
    bestOri = np.unravel_index(np.argmax(respMat), respMat.shape)[-1]  
    
    gridSpacingDeg = xpos[1]-xpos[0]
    gridSpacingPix = int(round(imageDownsamplePixPerDeg*gridSpacingDeg))
    r = np.repeat(np.repeat(respMat[:,:,bestOri],gridSpacingPix,axis=0),gridSpacingPix,axis=1)
    rmap = np.zeros(imagesDownsampled[0].shape)
    y0,x0 = (int(rmap.shape[i]/2-r.shape[i]/2) for i in (0,1))
    rmap[y0:y0+r.shape[0],x0:x0+r.shape[1]] = r
    
    for ax,img,imname in zip(axes,imagesDownsampled,imageNames):
        ax.imshow(rmap,cmap='hot')
#        ax.imshow(img,cmap='gray',alpha=0.9)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_psth_all_flashes(spikes, axes=None, preTime = 0.05, postTime = 0.55, sdfSigma=0.005):
    image_flash_times = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
    image_id = np.array(core_data['visual_stimuli']['image_name'])
    
    sdfs = []
    latencies = []
    for i,img in enumerate(imageNames):
        this_image_times = image_flash_times[image_id==img]
        sdf, t = analysis_utils.getSDF(spikes,this_image_times-preTime,preTime+postTime, sigma=sdfSigma)
        latency = analysis_utils.find_latency(sdf[:int(1000*(preTime+0.25+0.05))], int(preTime*1000), 5)
        latencies.append(latency)
        sdfs.append(sdf)
        
    ymax = round(max([s.max() for s in sdfs])+0.5)
    for ax,sdf,lat in zip(axes,sdfs,latencies):
        ax.add_patch(patches.Rectangle([0,0],width=stimDur.mean(),height=ymax,color='0.9',alpha=0.5))
        ax.plot(t-preTime, sdf, 'k')
        if not np.isnan(lat):
            ax.plot(lat/1000-preTime, sdf[lat], 'ro')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim([0, ymax])
        ax.set_xlim([-preTime,postTime])
        ax.set_yticks([0,ymax])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Time to flash (s)',fontsize=12)


def plot_psth_hits_vs_misses(spikes, axes=None, preTime=1.55, postTime=4.5, sdfSigma=0.02):
    hitSDFs = []
    missSDFs = []
    for resp,s in zip((hit, miss),(hitSDFs,missSDFs)):
        for img in imageNames:
            selectedTrials = resp & (changeImage==img) & (~ignore)
            changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
            sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
            s.append(sdf)
    
    ymax = round(max([max(h.max(),m.max()) for h,m in zip(hitSDFs,missSDFs)])+0.5)
    stimdur = stimDur.mean()
    stimint = stimdur+grayDur[0,0]
    stimStarts = np.concatenate((np.arange(-stimint,-preTime,-stimint),np.arange(0,postTime,stimint)))
    for ax,h,m in zip(axes,hitSDFs,missSDFs):
        for s in stimStarts:
            ax.add_patch(patches.Rectangle([s,0],width=stimdur,height=ymax,color='0.9',alpha=0.5))
        ax.plot(t-preTime, h, 'b')
        ax.plot(t-preTime, m, 'k')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_ylim([0, ymax])
        ax.set_xlim([-preTime,postTime])
        ax.set_yticks([0,ymax])
        if ax is not axes[0]:
            ax.set_yticklabels([])
        if ax is not axes[-1]:
            ax.set_xticklabels([])
    axes[-1].set_xlabel('Time to change (s)',fontsize=12) 

