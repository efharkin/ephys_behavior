# -*- coding: utf-8 -*-
"""
Created on Mon Oct 08 14:24:20 2018

@author: svc_ccg
"""

from __future__ import division
import os
import glob
import datetime
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe


def calculateHitRate(hits,misses):
    n = hits+misses
    hitRate = hits/n
    if hitRate==0:
        hitRate = 0.5/n
    elif hitRate==1:
        hitRate = 1-0.5/n
    return hitRate

def calculateDprime(hits,misses,falseAlarms,correctRejects):
    hitRate = calculateHitRate(hits,misses)
    falseAlarmRate = calculateHitRate(falseAlarms,correctRejects)
    z = [scipy.stats.norm.ppf(r) for r in (hitRate,falseAlarmRate)]
    return z[0]-z[1]


pickleDir = r'\\EphysRoom342\Data\behavior pickle files'

mouseInfo = (('385533',('09072018','09102018','09112018','09172018')),
             ('390339',('09192018','09202018','09212018')),
             ('394873',('10042018','10052018')),
             ('403472',('10312018','11012018')),
             ('403468',('11142018','11152018')),
             ('412624',('11292018','11302018')),
             ('416656',('03122019','03132019','03142019')),
             ('409096',('03212019',)),
             ('417882',('03262019','03272019')),
             ('408528',('04042019','04052019')),
             ('408527',('04102019','04112019')),
            )

trainingDay = []
isImages = []
isRig = []
isEphys = []
rewardsEarned = []
dprimeOverall = []
dprimeEngaged = []
probEngaged = []
frameRate = 60.0
windowFrames = 60*frameRate
for mouseID,ephysDates in mouseInfo: 
    ephysDateTimes = [datetime.datetime.strptime(d,'%m%d%Y') for d in ephysDates] if ephysDates is not None else (None,)
    rewardsEarned.append([])
    dprimeOverall.append([])
    dprimeEngaged.append([])
    probEngaged.append([])
    trainingDate = []
    trainingStage = []
    rigID = []
    for pklFile in  glob.glob(os.path.join(pickleDir,mouseID,'*.pkl')):
        try:
            core_data = data_to_change_detection_core(pd.read_pickle(pklFile))
            trials = create_extended_dataframe(
                trials=core_data['trials'],
                metadata=core_data['metadata'],
                licks=core_data['licks'],
                time=core_data['time'])
        except:
            print('could not import '+pklFile)
            continue
        
        autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
        earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')
        ignore = earlyResponse | autoRewarded
        miss = np.array(trials['response_type']=='MISS')
        hit = np.array(trials['response_type']=='HIT')
        falseAlarm = np.array(trials['response_type']=='FA')
        correctReject = np.array(trials['response_type']=='CR')
        
        rewardsEarned[-1].append(hit.sum())
        dprimeOverall[-1].append(calculateDprime(hit.sum(),miss.sum(),falseAlarm.sum(),correctReject.sum()))
        
        startFrame = int(trials['startframe'][0])
        endFrame = int(np.array(trials['endframe'])[-1])
        changeFrames = np.array(trials['change_frame'])
        hitFrames = np.zeros(endFrame,dtype=bool)
        hitFrames[changeFrames[hit].astype(int)] = True
        binSize = int(frameRate*60)
        halfBin = int(binSize/2)
        engagedThresh = 2
        rewardRate = np.zeros(hitFrames.size,dtype=int)
        rewardRate[halfBin:halfBin+hitFrames.size-binSize+1] = np.correlate(hitFrames,np.ones(binSize))
        rewardRate[:halfBin] = rewardRate[halfBin]
        rewardRate[-halfBin:] = rewardRate[-halfBin]
        engagedTrials = rewardRate[changeFrames[~ignore].astype(int)]>engagedThresh
        dprimeEngaged[-1].append(calculateDprime(*(r[~ignore][engagedTrials].sum() for r in (hit,miss,falseAlarm,correctReject))))
        probEngaged[-1].append(np.sum(rewardRate>engagedThresh)/rewardRate.size)
            
        trainingDate.append(datetime.datetime.strptime(os.path.basename(pklFile)[:6],'%y%m%d'))
        trainingStage.append(core_data['metadata']['stage'])
        rigID.append(core_data['metadata']['rig_id'])
        
    trainingDay.append(np.array([(d-min(trainingDate)).days+1 for d in trainingDate]))
    isImages.append(np.array(['images' in s for s in trainingStage]))
    isRig.append(np.array(['NP' in r for r in rigID]))
    isEphys.append(np.array([d in ephysDateTimes for d in trainingDate]))


params = (rewardsEarned,dprimeEngaged,probEngaged)
labels = ('Rewards Earned','d prime','prob. engaged')
for ind,(mouseID,ephysDates) in enumerate(mouseInfo):     
    fig = plt.figure(facecolor='w')
    for i,(prm,lbl,ymax) in enumerate(zip(params,labels,(None,None,None))):
        ax = plt.subplot(len(params),1,i+1)
        for j,(d,p) in enumerate(zip(trainingDay[ind],prm[ind])):
            mec = 'r' if isEphys[ind][j] else 'k'
            mfc = mec if isRig[ind][j] else 'none'
            mrk = 'o' if isImages[ind][j] else 's'
            ax.plot(d,p,mrk,mec=mec,mfc=mfc,ms=8)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,max(trainingDay[ind])+1])
        ylimMax = np.nanmax(prm[ind]) if ymax is None else ymax
        ax.set_ylim([0,1.05*ylimMax])
        ax.set_ylabel(lbl,fontsize=14)
        if i==0:
            ax.set_title(mouseID,fontsize=14)
        if i==len(params)-1:
            ax.set_xlabel('Day',fontsize=14)
    plt.tight_layout()


labels = ('NSB','Rig1','RigLast','Ephys1','Ephys2')
numRewards = []
dpr = []
engaged = []
for day,rig,ephys,rewards,d,eng in zip(trainingDay,isRig,isEphys,rewardsEarned,dprimeEngaged,probEngaged):
    numRewards.append([])
    dpr.append([])
    engaged.append([])
    sortOrder = np.argsort(day)
    rig,ephys,rewards,d,eng = [np.array(a)[sortOrder] for a in (rig,ephys,rewards,d,eng)]
    if not all(rig):
        lastNSBDay = np.where(~rig)[0][-1]
        numRewards[-1].append(rewards[lastNSBDay])
        dpr[-1].append(d[lastNSBDay])
        engaged[-1].append(eng[lastNSBDay])
        firstRigDay = np.where(rig)[0][0]
        numRewards[-1].append(rewards[firstRigDay])
        dpr[-1].append(d[firstRigDay])
        engaged[-1].append(eng[firstRigDay])
    else:
        numRewards[-1].extend([np.nan]*2)
        dpr[-1].extend(([np.nan]*2))
        engaged[-1].extend(([np.nan]*2))
    ephysInd = np.where(ephys)[0]
    lastNonEphysDay = ephysInd[0]-1
    numRewards[-1].append(rewards[lastNonEphysDay])
    dpr[-1].append(d[lastNonEphysDay])
    engaged[-1].append(eng[lastNonEphysDay])    
    ephysDays = ephysInd[:2]
    numRewards[-1].extend(rewards[ephysDays])
    dpr[-1].extend(d[ephysDays])
    engaged[-1].extend(eng[ephysDays])
    if len(ephysDays)<2:
        numRewards[-1].append(np.nan)
        dpr[-1].append(np.nan)
        engaged[-1].append(np.nan)

params = (numRewards,dpr,engaged)
fig = plt.figure(facecolor='w')
for i,(prm,ylab) in enumerate(zip(params,('Rewards Earned','d prime','prob. engaged'))): 
    ax = plt.subplot(len(params),1,i+1)
    for p,rig in zip(prm,isRig):
        mkr = 'o' if not all(rig) else 's'
        ax.plot(p,'k'+mkr+'-',mfc='none',ms=10)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([-0.25,4.25])
    ax.set_ylim([0,1.05*np.nanmax(prm)])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylab,fontsize=12)

fig = plt.figure(facecolor='w')
for i,(prm,ylab) in enumerate(zip(params,('Rewards Earned','d prime','prob. engaged'))):
    prm = np.array([p for p,rig in zip(prm,isRig) if not all(rig)])
    meanPrm = np.nanmean(prm,axis=0)
    n = np.sum(~np.isnan(prm),axis=0)
    print(n)
    stdPrm = np.nanstd(prm,axis=0)
#    semPrm = stdPrm/n**0.5
    ax = plt.subplot(len(params),1,i+1)
    ax.plot(meanPrm,'ko',mfc='none',ms=10)
    for x,(m,s) in enumerate(zip(meanPrm,stdPrm)):
        ax.plot([x]*2,m+np.array([-s,s]),'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([-0.25,4.25])
    ax.set_ylim([0,1.05*np.nanmax(meanPrm+stdPrm)])
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylab,fontsize=12)

