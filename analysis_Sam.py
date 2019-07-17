# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os
import getData
import probeSync
import analysis_utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


baseDir = 'Z:\\'
localDir = r'C:\Users\svc_ccg\Desktop\Analysis\Probe'

mouseInfo = (
             ('409096',('03212019',),('ABCD',),'A',(False,)),
             ('417882',('03262019','03272019'),('ABCEF','ABCF'),'AB',(False,False)),
             ('408528',('04042019','04052019'),('ABCDEF',)*2,'AB',(True,True)),
             ('408527',('04102019','04112019'),('BCDEF',)*2,'AB',(True,True)),
             ('421323',('04252019','04262019'),('ABCDEF',)*2,'AB',(True,True)),
             ('422856',('04302019',),('ABCDEF',),'A',(True,)),
             ('423749',('05162019','05172019'),('ABCDEF',)*2,'AB',(True,True)),
            )


def getDataDict(makeHDF5=False,miceToAnalyze='all',probesToAnalyze='all',imageSetsToAnalyze='all',mustHavePassive=False,sdfParams={}):
    data = {}
    for mouseID,ephysDates,probeIDs,imageSet,passiveSession in mouseInfo:
        if miceToAnalyze!='all' and mouseID not in miceToAnalyze:
            continue
        for date,probes,imgset,passive in zip(ephysDates,probeIDs,imageSet,passiveSession):
            if probesToAnalyze!='all':
                probes = probesToAnalyze
            if imageSetsToAnalyze!='all' and imgset not in imageSetsToAnalyze:
                continue
            if mustHavePassive and not passive:
                continue
            
            expName = date+'_'+mouseID
            print(expName)
            dataDir = baseDir+expName
            obj = getData.behaviorEphys(dataDir,probes,probeGen='3b')
            hdf5Dir = os.path.join(localDir,expName+'.hdf5')
            
            if makeHDF5:
                obj.loadFromRawData()
                obj.saveHDF5(hdf5Dir)
            else:
                obj.loadFromHDF5(hdf5Dir)
            
            data[expName] = {}
            data[expName]['sdfs'] = getSDFs(obj,probes,**sdfParams)
            data[expName]['regions'] = getUnitRegions(obj)
            data[expName]['isi'] = {p: obj.probeCCF[p]['ISIRegion'] for p in probes}
    return data


def getSDFs(obj,probes='all',behaviorStates=('active','passive'),responses=('hit','miss','all'),epochs=('change','preChange'),preTime=0.25,postTime=0.75,sampInt=0.001,sdfFilt='exp',sdfSigma=0.005,avg=True,psth=False):
    
    if probes=='all':
        probes = obj.probes_to_analyze
    
    changeFrames = np.array(obj.trials['change_frame']).astype(int)+1 #add one to correct for change frame indexing problem
    flashFrames = np.array(obj.core_data['visual_stimuli']['frame'])
    
    sdfs = {probe: {state: {resp: {epoch: {image: [] for image in obj.imageNames+['all']} for epoch in epochs} for resp in responses} for state in behaviorStates} for probe in probes}
    
    for probe in probes:
        units = probeSync.getOrderedUnits(obj.units[probe])
        for state in sdfs[probe]:
            if state=='active' or len(obj.passive_pickle_file)>0:
                for resp in (sdfs[probe][state]):
                    if resp=='all':
                        trials = (~obj.ignore) & (obj.hit | obj.miss)
                    else:
                        trials = (~obj.ignore) & getattr(obj,resp)
                    frameTimes =obj.frameAppearTimes if state=='active' else obj.passiveFrameAppearTimes
                    changeTimes = frameTimes[changeFrames[trials]]
                    if 'preChange' in epochs:
                        flashTimes = frameTimes[flashFrames]
                        preChangeTimes = flashTimes[np.searchsorted(flashTimes,changeTimes)-1]
                    for u in units:
                        spikes = obj.units[probe][u]['times']
                        for epoch in epochs:
                            trialTimes = changeTimes if epoch=='change' else preChangeTimes
                            for image in obj.imageNames+['all']:
                                t = trialTimes if image=='all' else trialTimes[obj.changeImage[trials]==image]
                                if psth:
                                    s = analysis_utils.makePSTH(spikes,t-preTime,preTime+postTime,binSize=sampInt,avg=avg)
                                else:
                                    s = analysis_utils.getSDF(spikes,t-preTime,preTime+postTime,sampInt=sampInt,filt=sdfFilt,sigma=sdfSigma,avg=avg)[0]
                                sdfs[probe][state][resp][epoch][image].append(s)                    
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




# change mod and latency analysis
    
def findLatency(data,baseWin,respWin,thresh=3,minPtsAbove=30):
    latency = []
    for d in data:
#        ptsAbove = np.where(np.correlate(d[respWin]>d[baseWin].std()*thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        ptsAbove = np.where(np.correlate(d[respWin]>0.5,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        if len(ptsAbove)>0:
            latency.append(ptsAbove[0])
        else:
            latency.append(np.nan)
    return np.array(latency)


def calcChangeMod(preChangeSDFs,changeSDFs,baseWin,respWin):
    diff = changeSDFs-preChangeSDFs
    changeMod = np.log2(diff[:,respWin].mean(axis=1)/preChangeSDFs[:,respWin].mean(axis=1))
    changeMod[np.isinf(changeMod)] = np.nan
    meanMod = 2**np.nanmean(changeMod)
    semMod = (np.log(2)*np.nanstd(changeMod)*meanMod)/(changeMod.size**0.5)
    changeLat = findLatency(diff,baseWin,respWin)
    return meanMod, semMod, changeLat


data = getDataDict(sdfParams={'responses':['all']})
        
baseWin = slice(0,250)
respWin = slice(250,500)

pre,change = [[np.array([s for exp in data for probe in data[exp]['sdfs'] for s in data[exp]['sdfs'][probe][state]['all'][epoch]]) for state in ('active','passive')] for epoch in ('preChange','change')]
hasSpikesActive,hasSpikesPassive = [sdfs.mean(axis=1) > 0.1 for sdfs in change]
baseRate = [sdfs[:,baseWin].mean(axis=1) for sdfs in pre+change]
activePre,passivePre,activeChange,passiveChange = [sdfs-sdfs[:,baseWin].mean(axis=1)[:,None] for sdfs in pre+change]
hasResp = hasSpikesActive & hasSpikesPassive & (activeChange[:,respWin].max(axis=1) > 5*activeChange[:,baseWin].std(axis=1))

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
figs = [plt.figure(figsize=(12,6)) for _ in range(5)]
axes = [fig.add_subplot(1,1,1) for fig in figs]
for ind,(region,regionLabels) in enumerate(regionNames):
    inRegion = np.in1d(regions,regionLabels) & hasResp
    nUnits.append(inRegion.sum())
    
    # plot baseline and response spike rates
    for sdfs,base,clr in zip((activePre,passivePre,activeChange,passiveChange),baseRate,([1,0.7,0.7],[0.7,0.7,1],'r','b')):
        meanResp = sdfs[inRegion,respWin].mean(axis=1)
        peakResp = sdfs[inRegion,respWin].max(axis=1)
        for r,ax in zip((base[inRegion],meanResp,peakResp),axes[:3]):
            m = r.mean()
            s = r.std()/(r.size**0.5)
            ax.plot(ind,m,'o',mec=clr,mfc=clr)
            ax.plot([ind,ind],[m-s,m+s],color=clr)
    
    # plot mean change mod and latencies
    (activeChangeMean,activeChangeSem,activeChangeLat),(passiveChangeMean,passiveChangeSem,passiveChangeLat),(diffChangeMean,diffChangeSem,diffChangeLat),(diffPreMean,diffPreSem,diffPreLat) = \
    [calcChangeMod(pre[inRegion],change[inRegion],baseWin,respWin) for pre,change in zip((activePre,passivePre,passiveChange,passivePre),(activeChange,passiveChange,activeChange,activePre))]
    
    activeLat,passiveLat = [findLatency(sdfs[inRegion],baseWin,respWin) for sdfs in (activeChange,passiveChange)]
    
    for m,s,ec,fc in zip((activeChangeMean,passiveChangeMean,diffChangeMean,diffPreMean),(activeChangeSem,passiveChangeSem,diffChangeSem,diffPreSem),'rbkk',['r','b','k','none']):
        axes[-2].plot(ind,m,'o',mec=ec,mfc=fc)
        axes[-2].plot([ind,ind],[m-s,m+s],ec)
            
    for lat,ec,fc in zip((activeLat,passiveLat,activeChangeLat,passiveChangeLat,diffChangeLat),'rbrbk',('none','none','r','b','k')):
        m = np.nanmedian(lat)
        s = np.nanstd(lat)/(lat.size**0.5)
        axes[-1].plot(ind,m,'o',mec=ec,mfc=fc)
        axes[-1].plot([ind,ind],[m-s,m+s],ec)
    
    # plot pre and post change responses and their difference
    fig = plt.figure(figsize=(8,8))
    ylim = None
    for i,(pre,change,clr,lbl) in enumerate(zip((activePre,passivePre),(activeChange,passiveChange),([1,0,0],[0,0,1]),('Active','Passive'))):
        ax = fig.add_subplot(2,1,i+1)
        ax.plot(change[inRegion].mean(axis=0),color=clr)
        clrlight = np.array(clr).astype(float)
        clrlight[clrlight==0] = 0.7
        ax.plot(pre[inRegion].mean(axis=0),color=clrlight)
        ax.plot((change-pre)[inRegion].mean(axis=0),color=[0.5,0.5,0.5])
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([250,600])
        ax.set_xticks([250,350,450,550])
        ax.set_xticklabels([0,100,200,300,400])
        if ylim is None:
            ylim = plt.get(ax,'ylim')
        else:
            ax.set_ylim(ylim)
        ax.set_ylabel('Spikes/s')
        ax.set_title(region+' '+lbl)

for ax,ylbl in zip(axes,('Baseline (spikes/s)','Mean Resp (spikes/s)','Peak Resp (spikes/s)','Change Mod','Latency (ms)')):
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=14)
    ax.set_xlim([-0.5,len(regionNames)-0.5])
    ax.set_xticks(np.arange(len(regionNames)))
    ax.set_xticklabels([r[0]+'\nn='+str(n) for r,n in zip(regionNames,nUnits)],fontsize=16)
    ax.set_ylabel(ylbl,fontsize=16)



# decoding analysis

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


data = getDataDict(miceToAnalyze='all',imageSetsToAnalyze='A',mustHavePassive=False,sdfParams={'responses':['all'],'preTime':0,'postTime':0.25,'avg':False})

regionLabels = ('VISp','VISl','VISal','VISrl','VISpm','VISam')
for exp in data:
    print(exp)
    for probe in data[exp]['regions']:
        n = []
        for region in regionLabels:
            inRegion = np.in1d(data[exp]['regions'][probe],region)
            unitMeanSDFs = np.array([s.mean(axis=0) for s in data[exp]['sdfs'][probe]['active']['all']['change']['all']])
            hasSpikes = unitMeanSDFs.mean(axis=1)>0.1
            n.append(np.sum(inRegion & hasSpikes))
        print(probe,n)
        

nUnits = 20
nRepeats = 3
truncInterval = 5
respTrunc = np.arange(truncInterval,201,truncInterval)

model = RandomForestClassifier(n_estimators=100)
result = {region: {state: {'exps':[],'changeScore':[],'imageScore':[]} for state in ('active','passive')} for region in regionLabels}
for expInd,exp in enumerate(data):
    print('experiment '+str(expInd+1)+' of '+str(len(data.keys())))
    for probeInd,probe in enumerate(data[exp]['sdfs']):
        print('probe '+str(probeInd)+' of '+str(len(data[exp]['sdfs'].keys())))
        region = data[exp]['isi'][probe]
        if region in regionLabels:
            inRegion = np.in1d(data[exp]['regions'][probe],region)
            unitMeanSDFs = np.array([s.mean(axis=0) for s in data[exp]['sdfs'][probe]['active']['all']['change']['all']])
            hasSpikes = unitMeanSDFs.mean(axis=1)>0.1
            if inRegion.sum()>nUnits:
                units = np.where(inRegion & hasSpikes)[0]
                unitSamples = [np.random.choice(units,nUnits) for _ in range(nRepeats)]
                for state in result[region]:
                    if state in data[exp]['sdfs'][probe] and len(data[exp]['sdfs'][probe][state]['all']['change']['all'])>0:
                        changeScore = np.zeros((nRepeats,respTrunc.size))
                        imageScore = np.zeros_like(changeScore)
                        for i,u in enumerate(unitSamples):
                            for j,end in enumerate(respTrunc):
                                # decode image change
                                change,pre = [np.array(data[exp]['sdfs'][probe][state]['all'][epoch]['all'])[u][:,:,0:end].transpose((1,0,2)) for epoch in ('change','preChange')]
                                change = change.reshape((change.shape[0],-1))
                                pre = pre.reshape((pre.shape[0],-1))
                                X = np.concatenate((change,pre))
                                y = np.zeros(X.shape[0])
                                y[:int(X.shape[0]/2)] = 1
                                changeScore[i,j] = cross_val_score(model,X,y,cv=3).mean()
                                # decode image identity
                                imgSDFs = [np.array(data[exp]['sdfs'][probe][state]['all']['change'][img])[u][:,:,0:end].transpose((1,0,2)) for img in data[exp]['sdfs'][probe][state]['all']['change'] if img!='all']
                                X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
                                y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
                                imageScore[i,j] = cross_val_score(model,X,y,cv=3).mean()
                        result[region][state]['exps'].append(exp)
                        result[region][state]['changeScore'].append(changeScore.mean(axis=0))
                        result[region][state]['imageScore'].append(imageScore.mean(axis=0))



for score,ymin in zip(('changeScore','imageScore'),[0,0.45]):
    plt.figure(facecolor='w',figsize=(10,10))
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,region in enumerate(regionLabels):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            for s in result[region][state][score]:
                ax.plot(respTrunc,s,'k')
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0.5,0.75,1])
            ax.set_xlim([0,200])
            ax.set_ylim([ymin,1])
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])
            if j==0:
                ax.set_title(region)
            else:
                ax.set_yticklabels([])
            if i==0 and j==0:
                ax.set_ylabel('Decoder Accuracy')
    ax.set_xlabel('Time (ms)')












