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
import matplotlib.pyplot as plt
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.translator.core import create_extended_dataframe

pickleDir = r'\\EphysRoom342\Data\behavior pickle files'

mouseInfo = (('385533',('09072018','09102018','09112018','09172018')),
             ('390339',('09192018','09202018','09212018')),
             ('394873',('10042018','10052018')),
             ('403472',('10312018','11012018')),
             ('403468',('11142018','11152018')),
             ('412624',('11292018','11302018')))

trainingDay = []
isImages = []
isRig = []
isEphys = []
rewardsEarned = []
for mouseID,ephysDates in mouseInfo[:2]: 
    ephysDateTimes = [datetime.datetime.strptime(d,'%m%d%Y') for d in ephysDates] if ephysDates is not None else (None,)
    frameRate = 60
    window = 300*frameRate
    interval = 60*frameRate
    rewardThresh = 2/60
    rewardsEarned.append([])
    lastEngaged = []
    probEngaged = []
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
        
        startFrame = int(trials['startframe'][0])
        endFrame = int(trials['endframe'][hit.size-1])
        winFrames = np.arange(int(startFrame+window),int(endFrame),int(interval))
        hitProb = np.full(winFrames.size,np.nan)
        hitRate = hitProb.copy()
        falseAlarmProb = hitProb.copy()
        for i,f in enumerate(winFrames):
            h,m,fa,cr = [np.sum((trials['change_frame'][r & (~ignore)]>=f-window) & (trials['change_frame'][r & (~ignore)]<f)) for r in (hit,miss,falseAlarm,correctReject)]
            hitRate[i] = h/window*frameRate
            hitProb[i] = h/(h+m) if h>0 else 0
            falseAlarmProb[i] = fa/(fa+cr) if fa>0 else 0
        
        rewardsEarned[-1].append(np.sum(hit & (~ignore)))
        if np.any(hitRate>rewardThresh):
            lastEngaged.append(interval/frameRate*(1+np.where(hitRate>rewardThresh)[0][-1]))
            probEngaged.append(np.sum(hitRate>rewardThresh)/hitRate.size)
        else:
            lastEngaged.append(0)
            probEngaged.append(0)
            
        trainingDate.append(datetime.datetime.strptime(os.path.basename(pklFile)[:6],'%y%m%d'))
        trainingStage.append(core_data['metadata']['stage'])
        rigID.append(core_data['metadata']['rig_id'])
        
    trainingDay.append(np.array([(d-min(trainingDate)).days+1 for d in trainingDate]))
    isImages.append(np.array(['images' in s for s in trainingStage]))
    isRig.append(np.array(['NP' in r for r in rigID]))
    isEphys.append(np.array([d in ephysDateTimes for d in trainingDate]))
    
    fig = plt.figure(facecolor='w')
    params = (rewardsEarned[-1],)
    for i,(prm,ymax,lbl) in enumerate(zip(params,(None,),('Rewards Earned',))):
        ax = plt.subplot(len(params),1,i+1)
        for j,(d,p) in enumerate(zip(trainingDay[-1],prm)):
            mec = 'r' if isEphys[-1][j] else 'k'
            mfc = mec if isRig[-1][j] else 'none'
            mrk = 'o' if isImages[-1][j] else 's'
            ax.plot(d,p,mrk,mec=mec,mfc=mfc,ms=8)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False,labelsize=12)
        ax.set_xlim([0,max(trainingDay[-1])+1])
        ylimMax = max(prm) if ymax is None else ymax
        ax.set_ylim([0,1.05*ylimMax])
        ax.set_ylabel(lbl,fontsize=14)
        if i==0:
            ax.set_title(mouseID,fontsize=14)
        if i==len(params)-1:
            ax.set_xlabel('Day',fontsize=14)
    plt.tight_layout()


labels = ('NSB','Rig1','RigLast','Ephys1','Ephys2')
numRewards = []
for day,rig,ephys,rewards in zip(trainingDay,isRig,isEphys,rewardsEarned):
    numRewards.append([])
    sortOrder = np.argsort(day)
    rig,ephys,rewards = [np.array(a)[sortOrder] for a in (rig,ephys,rewards)]
    if not all(rig):
        numRewards[-1].append(rewards[np.where(~rig)[0][-1]])
        numRewards[-1].append(rewards[np.where(rig)[0][0]])
    else:
        numRewards[-1].extend([np.nan]*2)
    ephysInd = np.where(ephys)[0]
    numRewards[-1].append(rewards[ephysInd[0]-1])
    numRewards[-1].extend(rewards[ephysInd[:2]])
    
    
fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
for r,rig in zip(numRewards,isRig):
    mkr = 'o' if not all(rig) else 's'
    ax.plot(np.arange(len(r)),r,'k'+mkr+'-',mfc='none',ms=10)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([-0.25,4.25])
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_ylim([0,180])
ax.set_ylabel('Rewards Earned',fontsize=12)

