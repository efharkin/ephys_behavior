# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os, cv2
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
        
        
# behavior summary
startFrame = int(trials['startframe'][0])
startTime = frameTimes[startFrame]
endFrame = int(trials['endframe'][hit.size-1])
endTime = frameTimes[endFrame]   

fig = plt.figure(facecolor='w',figsize=(18,10))
ax = plt.subplot(4,1,1)
selectedTrials = ~earlyResponse
changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
for trialIndex,t in zip(np.where(selectedTrials)[0],changeTimes):
    licks = frameTimes[np.array(trials['lick_frames'][trialIndex]).astype(int)]-t
    ax.plot(t-startTime+np.zeros(licks.size),licks,'o',mec='0.5',mfc='none',ms=3)
    reward = frameTimes[np.array(trials['reward_frames'][trialIndex]).astype(int)]-t
    m = 's' if autoRewarded[trialIndex] else 'o'
    ax.plot(t-startTime+np.zeros(reward.size),reward,m,mec='0.5',mfc='0.5',ms=3)
for resp,clr in zip((hit,miss,falseAlarm,correctReject),'bkrg'):
    ax.plot(changeTimes[resp[selectedTrials]],-0.125+np.zeros(resp.sum()),'s',mec=clr,mfc='none',ms=3)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,endTime-startTime])
ax.set_ylim([-0.25,postTime])
ax.set_ylabel('Time to image change (s)',fontsize=12)

ax = plt.subplot(4,1,2)
ax.plot(runTime,runSpeed,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,endTime-startTime])
ax.set_ylabel('Speed',fontsize=12)

ax = plt.subplot(4,1,3)
for resp,clr,lbl in zip((hit,miss,falseAlarm,correctReject),'bkrg',('hit','miss','false alarm','correct reject')):
    ax.plot(changeTimes-startTime,np.cumsum(resp[selectedTrials]),clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,endTime-startTime])
ax.set_ylabel('Count',fontsize=12)
ax.legend()

ax = plt.subplot(4,1,4)
window = 300*60
interval = 60*60
dframes = np.arange(int(startFrame+window),int(endFrame),int(interval))
hitProb = np.full(dframes.size,np.nan)
falseAlarmProb = np.full(dframes.size,np.nan)
d = np.full(dframes.size,np.nan)
for i,f in enumerate(dframes):
    h,m,fa,cr = [np.sum((trials['change_frame'][r & (~ignore)]>=f-window) & (trials['change_frame'][r & (~ignore)]<f)) for r in (hit,miss,falseAlarm,correctReject)]
    hitProb[i] = h/(h+m)
    if hitProb[i]==1:
        hitProb[i] = 1-0.5/(h+m)
    elif hitProb[i]==0:
        hitProb[i] = 0.5/(h+m)
    falseAlarmProb[i] = fa/(fa+cr)
    if falseAlarmProb[i]==1:
        falseAlarmProb[i] = 1-0.5/(fa+cr)
    elif falseAlarmProb[i]==0:
        falseAlarmProb[i] = 0.5/(fa+cr)
    d[i] = scipy.stats.norm.ppf(hitProb[i])-scipy.stats.norm.ppf(falseAlarmProb[i])
ax.plot(frameTimes[dframes]-startTime,hitProb,'b',label='hit')
ax.plot(frameTimes[dframes]-startTime,falseAlarmProb,'k',label='false alarm')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlim([0,endTime-startTime])
ax.set_ylim([0,1])
ax.set_ylabel('Probability',fontsize=12)
ax.set_xlabel('Time (s)',fontsize=12)
ax.legend()

plt.tight_layout()



# licks aligned to non-change flashes
flashFrames = np.array(core_data['visual_stimuli']['frame'])
flashFrames = np.setdiff1d(np.array(core_data['visual_stimuli']['frame']),trials['change_frame']+1)
flashTimes = frameTimes[flashFrames]
lickTimes = probeSync.get_sync_line_data(syncDataset,'lick_sensor')[0]

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for row,t in enumerate(flashTimes):
    ax.vlines(lickTimes[(lickTimes>=t-preTime) & (lickTimes<=t+postTime)]-t,row+0.6,row+1.4,color='k')





# sdf for all hit and miss trials
probesToAnalyze = ['A','B','C']
unitsToAnalyze = []

for pid in probesToAnalyze:
    orderedUnits = probeSync.getOrderedUnits(units[pid]) if len(unitsToAnalyze)<1 else unitsToAnalyze
    for u in orderedUnits:
        spikes = units[pid][u]['times']
        
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


# raster aligned to each image change
for pid in probesToAnalyze:
    pdf = PdfPages(os.path.join(dataDir,'imageChangeRaster_'+ pid+'.pdf'))
    orderedUnits = probeSync.getOrderedUnits(units[pid]) if len(unitsToAnalyze)<1 else unitsToAnalyze
    for u in orderedUnits:
        spikes = units[pid][u]['times']
        fig = plt.figure(facecolor='w',figsize=(8,10))
        for i,img in enumerate(imageNames):
            ax = plt.subplot(imageNames.size,1,i+1)
            selectedTrials = (changeImage==img) & (~ignore)
            changeTimes = frameTimes[np.array(trials['change_frame'][selectedTrials]).astype(int)]
            for row,(trialIndex,t) in enumerate(zip(np.where(selectedTrials)[0],changeTimes)):
                licks = frameTimes[np.array(trials['lick_frames'][trialIndex]).astype(int)]-t
                ax.plot(licks,row+np.ones(licks.size),'o',mec='0.5',mfc='none',ms=2)
                reward = frameTimes[np.array(trials['reward_frames'][trialIndex]).astype(int)]-t
                ax.plot(reward,row+np.ones(reward.size),'o',mec='r',mfc='r',ms=2)
                ax.vlines(spikes[(spikes>=t-preTime) & (spikes<=t+postTime)]-t,row+0.6,row+1.4,color='k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False,labelsize=10)
            ax.set_xlim([-preTime,postTime])
            ax.set_ylim([0,row+2])
            ax.set_yticks([1,row+1])
            ax.set_ylabel(img,fontsize=12)
            if i==0:
                ax.set_title('Probe '+pid+', Unit '+str(u)+', '+units[pid][u]['ccfRegion'],fontsize=12)
            if i==imageNames.size-1:
                ax.set_xlabel('Time relative to image change (s)',fontsize=12)
            else:
                ax.set_xticklabels([])
        plt.tight_layout()
        fig.savefig(pdf,format='pdf')
        plt.close(fig)
    pdf.close()
        
        
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
#        fig = plt.figure(facecolor='w')
#        ax = plt.subplot(1,1,1)
#        ax.plot([0,0],[0,1000],'k--')
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
#        for side in ('right','top'):
#            ax.spines[side].set_visible(False)
#        ax.tick_params(direction='out',top=False,right=False,labelsize=10)
#        ax.set_xlim([-preTime,postTime])
#        ax.set_ylim([0,1.02*ymax])
#        ax.set_xlabel('Time relative to saccade (s)',fontsize=12)
#        ax.set_ylabel('Spike/s',fontsize=12)
#        ax.set_title('Probe '+pid+', Unit '+str(u),fontsize=12)
#        plt.tight_layout()
#    multipage(os.path.join(dataDir, 'saccadeSDFs_' + pid + '.pdf'))
#    plt.close('all')
    

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
binWidth = 0.05
bins = np.arange(-preTime,postTime,binWidth)
for lat,peak,clr in zip(np.concatenate(latency).T,np.concatenate(peakResp).T,'rb'):
    ax.plot(bins[:-1]+binWidth/2,np.histogram(lat[(~np.isnan(lat)) & (peak>10)],bins)[0],clr,linewidth=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.set_xlim([-1,1])
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlabel('Saccade response latency (s)',fontsize=12)
ax.set_ylabel('Number of units',fontsize=12)
ax.legend(('temporal','nasal'))
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
for resp,clr in zip((hit,miss,falseAlarm,correctReject),'bkrg'):
    changeTimes = frameTimes[np.array(trials['change_frame'][~ignore & resp]).astype(int)]
    alignedPupilArea = np.zeros((changeTimes.size,int(frameRate*(preTime+postTime))))
    for i,t in enumerate(changeTimes):
        ind = np.argmin(np.abs(eyeFrameTimes-t))
        alignedPupilArea[i] = pupilAreaFilt[int(ind-frameRate*preTime):int(ind+frameRate*postTime)]
    ax.plot(np.arange(0,preTime+postTime,1/frameRate)-preTime,np.nanmean(alignedPupilArea,axis=0),clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=10)
ax.set_xlabel('Time relative to image change (s)',fontsize=12)
ax.set_ylabel('Pupil Area (pixels^2)',fontsize=12)
ax.legend(('hit','miss','false alarm','correct reject'))
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


# receptive fields
def find_spikes_per_trial(spikes, trial_starts, trial_ends):
    spike_counts = np.zeros(len(trial_starts))
    for i, (ts, te) in enumerate(zip(trial_starts, trial_ends)):
        spike_counts[i] = ((spikes>=ts) & (spikes<te)).sum()  
    return spike_counts
    
# get rf mapping stim info
rfstim_pickle_file = glob.glob(os.path.join(dataDir, '*brain_observatory_stimulus.pkl'))[0] 
stim_dict = pd.read_pickle(rfstim_pickle_file)
    
pre_blank_frames = int(stim_dict['pre_blank_sec']*stim_dict['fps'])
rfstim = stim_dict['stimuli'][0]

# get image info\
images = core_data['image_set']['images']
imageNames = [i['image_name'] for i in core_data['image_set']['image_attributes']]

monSizePix = stim_dict['monitor']['sizepix']
monHeightCm = monSizePix[1]/monSizePix[0]*stim_dict['monitor']['widthcm']
monDistCm = stim_dict['monitor']['distancecm']

imagePixPerDeg = images[0].shape[0]/np.degrees(2*np.arctan(0.5*monHeightCm/monDistCm))

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

spikes = units['A'][221]['times']
trial_spikes = find_spikes_per_trial(spikes, rf_trial_start_times+resp_latency, rf_trial_start_times+resp_latency+0.2)
respMat = np.zeros([ypos.size, xpos.size, ori.size])
for (y, x, o, tspikes) in zip(trial_xpos, trial_ypos, trial_ori, trial_spikes):
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


img = np.stack((images[0],)*3+(alpha,),axis=2)

ax = plt.subplot(1,1,1)
ax.patch.set_alpha(0.0)
ax.imshow(img)


im = axis.imshow(respMat[:, :, bestOri].T, interpolation='none', origin='lower')
plt.colorbar(im)    
tickLabels = [str(tick) for tick in np.unique(xpos)[::2]]
axis.set_xticks(np.arange(0, len(np.unique(xpos)), 2))
axis.set_yticks(np.arange(0, len(np.unique(xpos)), 2))
axis.set_xticklabels(tickLabels)
axis.set_yticklabels(tickLabels)



# cell x image peak response heat map
image_flash_times = frameTimes[np.array(core_data['visual_stimuli']['frame'])]
image_id = np.array(core_data['visual_stimuli']['image_name'])

regionsToConsider = ('VIS','cc')

sdfSigma = 0.005
preTime = 0.25
postTime = 0.5
#resp = []
for pid in probeIDs:
    orderedUnits = probeSync.getOrderedUnits(units[pid])
    for u in orderedUnits:
        region = units[pid][u]['ccfRegion']
        if region is not None and any([r in region for r in regionsToConsider]):
            r = np.zeros(len(imageNames))
            spikes = units[pid][u]['times']
            for i,img in enumerate(sorted(imageNames)):
                this_image_times = image_flash_times[image_id==img]
                sdf, t = analysis_utils.getSDF(spikes,this_image_times-preTime,preTime+postTime,sigma=sdfSigma)
                r[i] = sdf[t>preTime].max()-sdf[t<preTime].mean()
            resp.append(r)

respMat = np.stack(resp)
normResp = respMat/respMat.max(axis=1)[:,None]
sortResp = normResp[np.argsort([np.argmax(r) for r in respMat])]

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.imshow(sortResp,cmap='magma',interpolation='none')
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel('Image',fontsize=10)
ax.set_ylabel('Unit',fontsize=10)
plt.tight_layout()

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(np.sum(sortResp==1,axis=0)/sortResp.shape[0],'k',linewidth=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([-0.5,7.5])
ax.set_ylim([0,0.2])
ax.set_xticks(np.arange(len(imageNames)))
ax.set_xticklabels(sorted(imageNames))
ax.set_ylabel('Fraction of Cells Preferred',fontsize=14)
plt.tight_layout()
            
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(sortResp.mean(axis=0),'k',linewidth=2)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=14)
ax.set_xlim([-0.5,7.5])
ax.set_ylim([0,0.6])
ax.set_xticks(np.arange(len(imageNames)))
ax.set_xticklabels(sorted(imageNames))
ax.set_ylabel('Population Mean Response',fontsize=14)
plt.tight_layout()


#
ax = plt.subplot(1,1,1)
for img in imageNames:
    licklat = trials['response_latency'][hit & (changeImage==img) & (~ignore)]
    ax.plot(licklat.size,np.mean(licklat),'ko')




#############################

from __future__ import division
import os
import getData
import probeSync
import analysis_utils
import numpy as np
import matplotlib.pyplot as plt


def getChangeSDFs(obj,sdfSigma=0.001):
    
    preTime = 0.25
    postTime = 0.75
    
    changeFrames = np.array(obj.trials['change_frame']).astype(int)+1 #add one to correct for change frame indexing problem
    flashFrames = np.array(obj.core_data['visual_stimuli']['frame'])
    
    sdfs = {probe: {state: {resp: {epoch: [] for epoch in ('change','preChange')} for resp in ('hit','miss','all')} for state in ('active','passive')} for probe in obj.probes_to_analyze}
    
    for probe in obj.probes_to_analyze:
        units = probeSync.getOrderedUnits(obj.units[probe])
        for state in sdfs[probe]:
            if state=='active' or len(obj.passive_pickle_file)>0:
                for resp in (sdfs[probe][state]):
                    trials = ~obj.ignore
                    if resp!='all':
                        trials = trials & getattr(obj,resp)
                    frameTimes =obj.frameAppearTimes if state=='active' else obj.passiveFrameAppearTimes
                    changeTimes = frameTimes[changeFrames[trials]]
                    flashTimes = frameTimes[flashFrames]
                    preChangeTimes = flashTimes[np.searchsorted(flashTimes,changeTimes)-1]
                    for u in units:
                        spikes = obj.units[probe][u]['times']
                        for epoch,startTimes in zip(('change','preChange'),(changeTimes,preChangeTimes)):
                            s = analysis_utils.getSDF(spikes,startTimes-preTime,preTime+postTime,sigma=sdfSigma)[0]
                            sdfs[probe][state][resp][epoch].append(s)                    
    return sdfs


def getUnitRegions(obj):
    regions = {}
    for probe in obj.probes_to_analyze:
        regions[probe] = []
        units = probeSync.getOrderedUnits(obj.units[probe])
        for u in units:
            r = obj.probeCCF[probe]['ISIRegion'] if obj.units[probe][u]['inCortex'] else obj.units[probe][u]['ccfRegion']
            regions[probe].append(r)
    return regions


baseDir = 'Z:\\'
localDir = r'C:\Users\svc_ccg\Desktop\Analysis\Probe'

mouseInfo = (
             ('409096',('03212019',),('ABCD',)),
             ('417882',('03262019','03272019'),('ABCEF','ABCF')),
             ('408528',('04042019','04052019'),('ABCDEF',)*2),
             ('408527',('04102019','04112019'),('BCDEF',)*2),
             ('421323',('04252019','04262019'),('ABCDEF',)*2),
             ('422856',('04302019','05012019'),('ABCDEF','ABCF')),
             ('423749',('05162019','05172019'),('ABCDEF',)*2),
            )


alldata = {}
for mouseID,ephysDates,probeIDs in mouseInfo:
    for date,probes in zip(ephysDates,probeIDs):
        expName = date+'_'+mouseID
        print(expName)
        dataDir = baseDir+expName
        obj = getData.behaviorEphys(dataDir,probes,probeGen='3b')
        hdf5Dir = os.path.join(localDir,expName+'.hdf5')
        
        # load and save data
#        obj.loadFromRawData()
#        obj.saveHDF5(hdf5Dir)
        
        # or fetch and analyze data
        obj.loadFromHDF5(hdf5Dir)
        
        alldata[expName] = {}
        alldata[expName]['sdfs'] = getChangeSDFs(obj)
        alldata[expName]['regions'] = getUnitRegions(obj)
        


def findLatency(data,baseWin,respWin,thresh=3,minPtsAbove=10):
    ptsAbove = np.where(np.correlate(data[respWin]>data[baseWin].std()*thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
    latency = ptsAbove[0] if len(ptsAbove)>0 else np.nan
    return latency


def calcChangeMod(preChangeSDFs,changeSDFs,baseWin,respWin):
    diff = changeSDFs-preChangeSDFs
    changeMod = np.log2(diff[:,respWin].max(axis=1)/preChangeSDFs[:,respWin].max(axis=1))
    changeMod[np.isinf(changeMod)] = np.nan
    meanMod = np.nanmean(changeMod) # 2**np.nanmean(changeMod)
    semMod = np.nanstd(changeMod)/(changeMod.size**0.5) # (np.log(2)*np.nanstd(changeMod)*meanMod)/(changeMod.size**0.5)
    changeLat = findLatency(diff.mean(axis=0),baseWin,respWin)
    return meanMod, semMod, changeLat


Adays = ('04042019','04102019','04252019','04302019','05162019')
Bdays = ('04052019','04112019','04262019','05172019')

data = dict(alldata)
for key in dict(data):
    if key[:8] not in Adays+Bdays:
        del data[key]
        
baseWin = slice(0,250)
respWin = slice(250,500)

pre,change = [[np.array([s for exp in data for probe in data[exp]['sdfs'] for s in data[exp]['sdfs'][probe][state]['all'][epoch]]) for state in ('active','passive')] for epoch in ('preChange','change')]
activePre,passivePre,activeChange,passiveChange = [sdfs-sdfs[:,baseWin].mean(axis=1)[:,None] for sdfs in pre+change]
hasResp = activeChange[:,respWin].max(axis=1) > 5*activeChange[:,baseWin].std(axis=1)

regions = np.array([r for exp in data for probe in data[exp]['regions'] for r in data[exp]['regions'][probe]])    
#regionNames = sorted(list(set(regions)))
regionNames = (
               ('V1',('VISp',)),
               ('LM',('VISl',)),
               ('AL',('VISal',)),
               ('RL',('VISrl',)),
               ('PM',('VISpm',)),
               ('AM',('VISam',)),
               ('LP',('LP',)),
               ('SCd',('SCig','SCig-b')),
               ('APN',('APN',)),
               ('MRN',('MRN',)),
               ('hipp',('CA1','CA3','DG-mo','DG-po','DG-sg','HPF'))
              )
regionNames = regionNames[:6]

nUnits = []
fig1 = plt.figure(figsize=(8,8))
ax1 = fig1.add_subplot(2,1,1)
ax2 = fig1.add_subplot(2,1,2)
for ind,(region,regionLabels) in enumerate(regionNames):
    inRegion = np.in1d(regions,regionLabels) & hasResp
    nUnits.append(inRegion.sum())
    
    (activeChangeMean,activeChangeSem,activeChangeLat),(passiveChangeMean,passiveChangeSem,passiveChangeLat),(diffChangeMean,diffChangeSem,diffChangeLat) = \
    [calcChangeMod(pre[inRegion],change[inRegion],baseWin,respWin) for pre,change in zip((activePre,passivePre,passiveChange),(activeChange,passiveChange,activeChange))]
    
    activeLat,passiveLat = [findLatency(sdfs[inRegion].mean(axis=0),baseWin,respWin) for sdfs in (activeChange,passiveChange)]
    
    for m,s,c in zip((activeChangeMean,passiveChangeMean,diffChangeMean),(activeChangeSem,passiveChangeSem,diffChangeSem),'rbk'):
        if c!='k':
            ax1.plot(ind,m,c+'o')
            ax1.plot([ind,ind],[m-s,m+s],c)
    ax2.plot(ind,activeLat,'o',mec='r',mfc='none')
    ax2.plot(ind,passiveLat,'o',mec='b',mfc='none')
    ax2.plot(ind,activeChangeLat,'o',mec='r',mfc='r')
    ax2.plot(ind,passiveChangeLat,'o',mec='b',mfc='b')
#    ax2.plot(ind,diffChangeLat,'o',mec='k',mfc='k')
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(2,1,1)
    ax.plot(activeChange[inRegion].mean(axis=0),color=[1,0,0])
    ax.plot(activePre[inRegion].mean(axis=0),color=[1,0.7,0.7])
    ax.plot((activeChange-activePre)[inRegion].mean(axis=0),color=[0.5,0.5,0.5])
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([250,600])
    ax.set_xticks([250,350,450,550])
    ax.set_xticklabels([0,100,200,300,400])
    ax.set_ylabel('Spikes/s')
    ax.set_title(region+' Active')
    
    ylim = plt.get(ax,'ylim')
    ax = fig.add_subplot(2,1,2)
    ax.plot(passiveChange[inRegion].mean(axis=0),color=[0,0,1])
    ax.plot(passivePre[inRegion].mean(axis=0),color=[0.7,0.7,1])
    ax.plot((passiveChange-passivePre)[inRegion].mean(axis=0),color=[0.5,0.5,0.5])
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([250,600])
    ax.set_ylim(ylim)
    ax.set_xticks([250,350,450,550])
    ax.set_xticklabels([0,100,200,300,400])
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Spikes/s')
    ax.set_title(region+' Passive')

for a in (ax1,ax2):
    for side in ('right','top'):
        a.spines[side].set_visible(False)
    a.tick_params(direction='out',top=False,right=False,labelsize=14)
    a.set_xlim([-0.5,len(regionNames)-0.5])
    a.set_xticks(np.arange(len(regionNames)))
ax1.set_xticklabels([])
ax2.set_xticklabels([r[0]+'\nn='+str(n) for r,n in zip(regionNames,nUnits)],fontsize=16)
ax1.set_ylabel('Change Mod',fontsize=16)
ax2.set_ylabel('Latency (ms)',fontsize=16)




from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score


model = RandomForestRegressor(n_estimators=100)
X = np.concatenate([alldata['04042019_408528']['A']['active']['hit'][epoch] for epoch in ('change','preChange')])
y = np.zeros(X.shape[0])
y[:int(X.shape[0]/2)] = 1
score = cross_val_score(model,X,y,cv=2)










