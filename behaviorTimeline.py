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


pickleDir = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\Behavior\behavior pickle files'

mouseID = '385533'
ephysDates = ('09072018','09102018','09112018','09172018')

mouseID = '390339'
ephysDates = ('09192018','09202018','09212018')

mouseID = '394873'
ephysDates = ('10042018','10052018')

mouseID = '404788'
ephysDates = None

mouseID = '403472'
ephysDates = None


ephysDates = [datetime.datetime.strptime(d,'%m%d%Y') for d in ephysDates] if ephysDates is not None else (None,)
frameRate = 60
window = 300*frameRate
interval = 60*frameRate
rewardThresh = 2/60
rewardsEarned = []
lastEngaged = []
probEngaged = []
trainingDate = []
trainingStage = []
for pklFile in  glob.glob(os.path.join(pickleDir,mouseID,'*.pkl')):
    try:
        core_data = data_to_change_detection_core(pd.read_pickle(pklFile))
        trials = create_extended_dataframe(
            trials=core_data['trials'],
            metadata=core_data['metadata'],
            licks=core_data['licks'],
            time=core_data['time'])
    except:
        print('count not import '+pklFile)
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
    
    rewardsEarned.append(np.sum(hit & (~ignore)))
    if np.any(hitRate>rewardThresh):
        lastEngaged.append(interval/frameRate*(1+np.where(hitRate>rewardThresh)[0][-1]))
        probEngaged.append(np.sum(hitRate>rewardThresh)/hitRate.size)
    else:
        lastEngaged.append(0)
        probEngaged.append(0)
        
    trainingDate.append(datetime.datetime.strptime(os.path.basename(pklFile)[:6],'%y%m%d'))
    trainingStage.append(trials['stage'])
    
t = np.array([(d-min(trainingDate)).days+1 for d in trainingDate])
isImages = np.array(['images' in s[0] for s in trainingStage])
isEphys = np.array([d in ephysDates for d in trainingDate])

fig = plt.figure(facecolor='w',figsize=(8,10))
for i,(prm,ymax,lbl) in enumerate(zip((rewardsEarned,lastEngaged,probEngaged),(None,3660,1),('Rewards Earned','Last Engaged (s)','Probability Engaged'))):
    ax = plt.subplot(3,1,i+1)
    for ind,clr,mrk in zip(((~isImages) & (~isEphys),(~isImages & isEphys),(isImages & (~isEphys)),(isImages & isEphys)),'krkr','ssoo'):
        ax.plot(t[ind],np.array(prm)[ind],mrk,mec=clr,mfc=clr,ms=8)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
    ax.set_xlim([0,max(t)+1])
    ylimMax = max(prm) if ymax is None else ymax
    ax.set_ylim([0,1.05*ylimMax])
    ax.set_ylabel(lbl,fontsize=14)
    if i==0:
        ax.set_title(mouseID,fontsize=14)
    if i==2:
        ax.set_xlabel('Training Day',fontsize=14)
plt.tight_layout()
    
    
    

