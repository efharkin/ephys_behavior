# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os
import probeSync
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def makePSTH(spikes,startTimes,windowDur,binSize=0.1):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros((len(startTimes),bins.size-1))    
    for i,start in enumerate(startTimes):
        counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,bins)[0]
    return counts.mean(axis=0)/binSize
    
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


preTime = 1.5
postTime = 1.5
sdfSigma = 0.02

probesToAnalyze = ['A','B','C']
unitsToAnalyze = []

# sdf for all hit and miss trials
for pid in probesToAnalyze:
    orderedUnits = probeSync.getOrderedUnits(units[pid]) if len(unitsToAnalyze)<1 else unitsToAnalyze
    for u in orderedUnits:
        spikes = units[pid][u]['times']
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ymax = 0
        for resp,clr in zip((hit,miss),'rb'):
            selectedTrials = resp & (~ignore)
            changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
            sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
            ax.plot(t-preTime,sdf,clr)
            ymax = max(ymax,sdf.max())
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,1.02*ymax])
        ax.set_xlabel('Time relative to image change (s)',fontsize=12)
        ax.set_ylabel('Spike/s',fontsize=12)
        ax.set_title('Probe '+pid+', Unit '+str(u),fontsize=12)
        plt.tight_layout()


# sdf for hit and miss trials for each image
for pid in probesToAnalyze:
    orderedUnits = probeSync.getOrderedUnits(units[pid]) if len(unitsToAnalyze)<1 else unitsToAnalyze
    for u in orderedUnits:
        spikes = units[pid][u]['times']
        fig = plt.figure(facecolor='w',figsize=(8,10))
        axes = []
        ymax = 0
        for i,img in enumerate(imageNames):
            axes.append(plt.subplot(imageNames.size,1,i+1))
            for resp,clr in zip((hit,miss),'rb'):
                selectedTrials = resp & (changeImage==img) & (~ignore)
                changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
                sdf,t = getSDF(spikes,changeTimes-preTime,preTime+postTime,sigma=sdfSigma)
                axes[-1].plot(t-preTime,sdf,clr)
                ymax = max(ymax,sdf.max())
        for ax,img in zip(axes,imageNames):
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xlim([-preTime,postTime])
            ax.set_ylim([0,1.02*ymax])
            ax.set_ylabel(img,fontsize=12)
            if ax!=axes[-1]:
                ax.set_xticklabels([])
        axes[-1].set_xlabel('Time relative to image change (s)',fontsize=12)
        axes[0].set_title('Probe '+pid+', Unit '+str(u),fontsize=12)
        plt.tight_layout()
        
        
# saccade aligned sdfs
for pid in probesToAnalyze:
    orderedUnits = probeSync.getOrderedUnits(units[pid]) if len(unitsToAnalyze)<1 else unitsToAnalyze
    for u in orderedUnits:
        spikes = units[pid][u]['times']
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.plot([0,0],[0,1000],'k--')
        ymax = 0
        for saccades,clr in zip((negSaccades,posSaccades),'rb'):
            saccadeTimes = eyeFrameTimes[saccades]
            sdf,t = getSDF(spikes,saccadeTimes-preTime,preTime+postTime,sigma=sdfSigma)
            ax.plot(t-preTime,sdf,clr)
            ymax = max(ymax,sdf.max())
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0,1.02*ymax])
        ax.set_xlabel('Time relative to saccade (s)',fontsize=12)
        ax.set_ylabel('Spike/s',fontsize=12)
        ax.set_title('Probe '+pid+', Unit '+str(u),fontsize=12)
        plt.tight_layout()
    multipage(os.path.join(dataDir, 'saccadeSDFs_' + pid + '.pdf'))
    plt.close('all')


