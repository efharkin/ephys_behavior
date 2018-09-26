# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os
import probeSync
import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib import patches
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
preTime = 2
postTime = 2
sdfSigma = 0.02
latency = []
latThresh = 5
minPtsAboveThresh = 50
latFilt = np.ones(minPtsAboveThresh)
peakResp = []
for pid in probesToAnalyze:
    orderedUnits = probeSync.getOrderedUnits(units[pid]) if len(unitsToAnalyze)<1 else unitsToAnalyze
    latency.append(np.full((orderedUnits.size,2),np.nan))
    peakResp.append(latency[-1].copy())    
    for i,u in enumerate(orderedUnits):
        spikes = units[pid][u]['times']
        fig = plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        ax.plot([0,0],[0,1000],'k--')
        ymax = 0
        for j,(saccades,clr) in enumerate(zip((negSaccades,posSaccades),'rb')):
            saccadeTimes = eyeFrameTimes[saccades]
            sdf,t = getSDF(spikes,saccadeTimes-preTime,preTime+postTime,sigma=sdfSigma)
            ax.plot(t-preTime,sdf,clr)
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
                latency[-1][i,j] = t[latInd]-preTime
                ax.plot(t[latInd]-preTime,sdf[latInd],'o',mfc=clr,mec=clr,ms=10)
            peakResp[-1][i,j] = z.max() if z.max()>-z.min() else z.min()
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
    

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.add_patch(patches.Rectangle([-10,-10],width=20,height=20,color='0.8'))
ax.plot([-1000,1000],[-1000,1000],'k--')
amax = 0
for peak in peakResp:
    ax.plot(peak[:,0],peak[:,1],'o',mec='k',mfc='none',ms=8,mew=2)
    amax = max(amax,np.abs(peak).max())
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlabel('Temporal saccade max z score',fontsize=12)
ax.set_ylabel('Nasal saccade max z score',fontsize=12)
amax *= 1.02
ax.set_xlim([-amax,amax])
ax.set_ylim([-amax,amax])
ax.set_aspect('equal')
plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for lat,peak in zip(latency,peakResp):
    for j,clr in enumerate('rb'):
        ax.plot(lat[:,j],peak[:,j],'o',mec=clr,mfc='none',ms=8,mew=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlabel('Saccade resp latency (s)',fontsize=12)
ax.set_ylabel('Max z score',fontsize=12)
plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
binWidth = 0.05
bins = np.arange(-preTime,postTime,binWidth)
for lat,peak,clr in zip(np.concatenate(latency).T,np.concatenate(peakResp).T,'rb'):
    ax.plot(bins[:-1]+binWidth/2,np.histogram(lat[(~np.isnan(lat)) & (peak>10)],bins)[0],clr,linewidth=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.set_xlim([-1,1])
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlabel('Saccade resp latency (s)',fontsize=12)
ax.set_ylabel('Number of units',fontsize=12)
plt.tight_layout()


allSaccadeTimes = eyeFrameTimes[np.sort(np.concatenate((negSaccades,posSaccades)))]
imageFlashTimes = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
lat = allSaccadeTimes[allSaccadeTimes<imageFlashTimes[-1],None]-imageFlashTimes
lat[lat<0] = np.nan
lat = np.nanmin(lat[~np.isnan(lat).all(axis=1)],axis=1)
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
binWidth = 0.05
bins = np.arange(0,0.75+binWidth,binWidth)
h = np.histogram(lat,bins)[0].astype(float)/lat.shape[0]
ax.plot(bins[:-1]+binWidth/2,h,'k',linewidth=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,0.75])
ax.set_ylim([0,1.02*h.max()])
ax.set_xlabel('Time from image flash to saccade (s)',fontsize=12)
ax.set_ylabel('Fraction of saccades',fontsize=12)
plt.tight_layout()


imageChangeTimes = frameTimes[np.array(trials['change_frame'][~earlyResponse]).astype(int)]
lat = allSaccadeTimes[allSaccadeTimes<imageChangeTimes[-1],None]-imageChangeTimes
lat[lat<0] = np.nan
lat = np.nanmin(lat[~np.isnan(lat).all(axis=1)],axis=1)
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
binWidth = 0.5
bins = np.arange(0,lat.max()+binWidth,binWidth)
h = np.histogram(lat,bins)[0].astype(float)/lat.shape[0]
ax.plot(bins[:-1]+binWidth/2,h,'k',linewidth=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,lat.max()])
ax.set_ylim([0,1.02*h.max()])
ax.set_xlabel('Time from image change to saccade (s)',fontsize=12)
ax.set_ylabel('Fraction of saccades',fontsize=12)
plt.tight_layout()



# pupil area
pupilAreaFilt = scipy.signal.medfilt(pupilArea,7)

frameRate = 60.0
preTime = postTime = 10
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for resp,clr in zip((hit,miss,falseAlarm,correctReject),'krgb'):
    changeTimes = frameTimes[np.array(trials['change_frame'][~ignore & resp]).astype(int)]
    alignedPupilArea = np.zeros((changeTimes.size,int(frameRate*(preTime+postTime))))
    for i,t in enumerate(changeTimes):
        ind = np.argmin(np.abs(eyeFrameTimes-t))
        alignedPupilArea[i] = pupilAreaFilt[int(ind-frameRate*preTime):int(ind+frameRate*postTime)]
    ax.plot(np.arange(0,preTime+postTime,1/frameRate)-preTime,np.nanmean(alignedPupilArea,axis=0),clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlabel('Time relative to image change',fontsize=12)
ax.set_ylabel('Pupil Area',fontsize=12)
plt.tight_layout()



# random forest model

from sklearn.ensemble import RandomForestRegressor

binWidth = 0.1
bins = np.arange(0,spikes[-1]+1,binWidth)
spikes = units['A'][9]['times']
fr = np.histogram(spikes,bins)[0].astype(float)/binWidth
psr = np.histogram(eyeFrameTimes[posSaccades],bins)[0].astype(float)/binWidth
psrmat = np.stack([np.roll(psr, shift) for shift in np.arange(-10, 11)]).T
nsr = np.histogram(eyeFrameTimes[negSaccades],bins)[0].astype(float)/binWidth
nsrmat = np.stack([np.roll(nsr, shift) for shift in np.arange(-10, 11)]).T

srmat = np.concatenate((psrmat,nsrmat),axis=1)

rf = RandomForestRegressor(n_estimators=100)

rf.fit(srmat,fr)


