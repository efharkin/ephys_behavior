# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os
import h5py
import fileIO
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


def getPopData(objToHDF5=False,popDataToHDF5=True,miceToAnalyze='all',probesToAnalyze='all',imageSetsToAnalyze='all',mustHavePassive=False,sdfParams={}):
    if popDataToHDF5:
        popHDF5Path = os.path.join(localDir,'popData.hdf5')
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
            hdf5Path = os.path.join(localDir,expName+'.hdf5')
            
            if objToHDF5:
                obj.loadFromRawData()
                obj.saveHDF5(hdf5Path)
            else:
                obj.loadFromHDF5(hdf5Path)
            
            if popDataToHDF5:
                trials = ~(obj.earlyResponse | obj.autoRewarded)
                resp = np.array([None for _ in trials],dtype='object')
                resp[obj.hit] = 'hit'
                resp[obj.miss] = 'miss'
                resp[obj.falseAlarm] = 'falseAlarm'
                resp[obj.correctReject] = 'correctReject'
                
                data = {expName:{}}
                data[expName]['sdfs'] = getSDFs(obj,probes=probes,**sdfParams)
                data[expName]['regions'] = getUnitRegions(obj,probes=probes)
                data[expName]['isi'] = {probe: obj.probeCCF[probe]['ISIRegion'] for probe in probes}
                data[expName]['changeImage'] = obj.changeImage[trials]
                data[expName]['response'] = resp[trials]
                # add preChange image identity, time between changes, receptive field info

                fileIO.objToHDF5(obj=None,saveDict=data,filePath=popHDF5Path)


def getSDFs(obj,probes='all',behaviorStates=('active','passive'),epochs=('change','preChange'),preTime=0.25,postTime=0.75,sampInt=0.001,sdfFilt='exp',sdfSigma=0.005,avg=False,psth=False):
    
    if probes=='all':
        probes = obj.probes_to_analyze
    
    trials = ~(obj.earlyResponse | obj.autoRewarded)
    changeFrames = np.array(obj.trials['change_frame']).astype(int)+1 #add one to correct for change frame indexing problem
    flashFrames = np.array(obj.core_data['visual_stimuli']['frame'])
    
    sdfs = {probe: {state: {epoch: [] for epoch in epochs} for state in behaviorStates} for probe in probes}
    
    for probe in probes:
        units = probeSync.getOrderedUnits(obj.units[probe])
        for state in sdfs[probe]:
            if state=='active' or len(obj.passive_pickle_file)>0:  
                frameTimes =obj.frameAppearTimes if state=='active' else obj.passiveFrameAppearTimes
                changeTimes = frameTimes[changeFrames[trials]]
                if 'preChange' in epochs:
                    flashTimes = frameTimes[flashFrames]
                    preChangeTimes = flashTimes[np.searchsorted(flashTimes,changeTimes)-1]
                for u in units:
                    spikes = obj.units[probe][u]['times']
                    for epoch in epochs:
                        t = changeTimes if epoch=='change' else preChangeTimes
                        if psth:
                            s = analysis_utils.makePSTH(spikes,t-preTime,preTime+postTime,binSize=sampInt,avg=avg)
                        else:
                            s = analysis_utils.getSDF(spikes,t-preTime,preTime+postTime,sampInt=sampInt,filt=sdfFilt,sigma=sdfSigma,avg=avg)[0]
                        sdfs[probe][state][epoch].append(s)                    
    return sdfs


def getUnitRegions(obj,probes='all'):
    
    if probes=='all':
        probes = obj.probes_to_analyze
    regions = {}
    for probe in probes:
        regions[probe] = []
        units = probeSync.getOrderedUnits(obj.units[probe])
        for u in units:
            r = obj.probeCCF[probe]['ISIRegion'] if obj.units[probe][u]['inCortex'] else obj.units[probe][u]['ccfRegion']
            regions[probe].append(r)
    return regions




# change mod and latency analysis
    
def findLatency(data,baseWin=None,respWin=None,method='rel',thresh=3,minPtsAbove=30):
    latency = []
    if baseWin is not None:
        data = data-data[:,baseWin].mean(axis=1)[:,None]
    if respWin is None:
        respWin = slice(0,data.shape[1])
    for d in data:
        if method=='abs':
            ptsAbove = np.where(np.correlate(d[respWin]>thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
        else:
            ptsAbove = np.where(np.correlate(d[respWin]>d[baseWin].std()*thresh,np.ones(minPtsAbove),mode='valid')==minPtsAbove)[0]
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
from sklearn.model_selection import cross_val_score, cross_val_predict


def findResponsiveUnits(sdfs,baseWin,respWin):
    unitMeanSDFs = np.array([s.mean(axis=0) for s in sdfs])
    hasSpikes = unitMeanSDFs.mean(axis=1)>0.1
    unitMeanSDFs -= unitMeanSDFs[:,baseWin].mean(axis=1)[:,None]
    hasResp = unitMeanSDFs[:,respWin].max(axis=1) > 5*unitMeanSDFs[:,baseWin].std(axis=1)
    return hasSpikes,hasResp


data = h5py.File(os.path.join(localDir,'popData.hdf5'))

regionLabels = ('VISp','VISl','VISal','VISrl','VISpm','VISam')

baseWin = slice(0,250)
respWin = slice(250,500)

for exp in data:
    print(exp)
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    for probe in data[exp]['regions']:
        n = []
        for region in regionLabels:
            inRegion = data[exp]['regions'][probe][:]==region
            if any(inRegion):
                sdfs = data[exp]['sdfs'][probe]['active']['change'][inRegion,:,baseWin.start:respWin.stop][:,trials]
                hasSpikes,hasResp = findResponsiveUnits(sdfs,baseWin,respWin)
                n.append(np.sum(hasSpikes & hasResp))
            else:
                n.append(0)
        print(probe,n)
        

nUnits = [30] #[5,10,20,30,40,50]
nRepeats = 3
nCrossVal = 3

truncInterval = 5
lastTrunc = 200
truncTimes = np.arange(truncInterval,lastTrunc+1,truncInterval)

assert((len(nUnits)>=1 and len(truncTimes)==1) or (len(nUnits)==1 and len(truncTimes)>=1))
model = RandomForestClassifier(n_estimators=100)
result = {exp: {probe: {state: {'changeScore':[],'changePredict':[],'imageScore':[],'respLatency':[]} for state in ('active','passive')} for probe in data[exp]['sdfs']} for exp in data}
for expInd,exp in enumerate(data):
    print('experiment '+str(expInd+1)+' of '+str(len(data.keys())))
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    changeImage = data[exp]['changeImage'][trials]
    imageNames = np.unique(changeImage)
    for probeInd,probe in enumerate(data[exp]['sdfs']):
        print('probe '+str(probeInd+1)+' of '+str(len(data[exp]['sdfs'].keys())))
        region = data[exp]['isi'][probe].value
        result[exp][probe]['region'] = region
        if region in regionLabels:
            inRegion = data[exp]['regions'][probe][:]==region
            if any(inRegion):
                hasSpikesActive,hasRespActive = findResponsiveUnits(data[exp]['sdfs'][probe]['active']['change'][inRegion,:,baseWin.start:respWin.stop][:,trials],baseWin,respWin)
                useUnits = hasSpikesActive & hasRespActive
                hasPassive = len(data[exp]['sdfs'][probe]['passive']['change'])>0
                if hasPassive:
                    hasSpikesPassive,hasRespPassive = findResponsiveUnits(data[exp]['sdfs'][probe]['passive']['change'][inRegion,:,baseWin.start:respWin.stop][:,trials],baseWin,respWin)
                    useUnits = useUnits & hasSpikesPassive
                units = np.where(useUnits)[0]
                for n in nUnits:
                    if len(units)>=n:
                        unitSamples = [np.random.choice(units,size=n,replace=False) for _ in range(nRepeats)]
                        for state in ('active','passive'):
                            if state=='active' or hasPassive:
                                changeScore = np.zeros((nRepeats,len(truncTimes)))
                                changePredict = []
                                imageScore = np.zeros_like(changeScore)
                                imagePredict = []
                                respLatency = []
                                changeSDFs,preChangeSDFs = [data[exp]['sdfs'][probe][state][epoch][inRegion,:,respWin].transpose((1,0,2))[trials] for epoch in ('change','preChange')]
                                for i,unitSamp in enumerate(unitSamples):
                                    for j,trunc in enumerate(truncTimes):
                                        # decode image change
                                        X = np.concatenate([s[:,unitSamp,:trunc].reshape((s.shape[0],-1)) for s in (changeSDFs,preChangeSDFs)])
                                        y = np.zeros(X.shape[0])
                                        y[:int(X.shape[0]/2)] = 1
                                        changeScore[i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                        if trunc==lastTrunc:
                                            # get model prediction probability for full length sdfs
                                            changePredict.append(cross_val_predict(model,X,y,cv=nCrossVal,method='predict_proba')[:trials.sum(),1])
                                        # decode image identity
                                        imgSDFs = [changeSDFs[:,unitSamp,:trunc][changeImage==img] for img in imageNames]
                                        X = np.concatenate([s.reshape((s.shape[0],-1)) for s in imgSDFs])
                                        y = np.concatenate([np.zeros(s.shape[0])+imgNum for imgNum,s in enumerate(imgSDFs)])
                                        imageScore[i,j] = cross_val_score(model,X,y,cv=nCrossVal).mean()
                                    # calculate population response latency for unit sample
                                    respLatency.append(findLatency(data[exp]['sdfs'][probe][state]['change'][inRegion,:,baseWin.start:respWin.stop][unitSamp][:,trials].mean(axis=(0,1))[None,:],baseWin,respWin)[0])
                                result[exp][probe][state]['changeScore'].append(changeScore.mean(axis=0))
                                result[exp][probe][state]['changePredict'].append(np.mean(changePredict,axis=0))
                                result[exp][probe][state]['imageScore'].append(imageScore.mean(axis=0))
                                result[exp][probe][state]['respLatency'].append(np.nanmean(respLatency))
                            

# plot scores vs number of units
plt.figure(facecolor='w',figsize=(10,10))
gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
allScores = {score: [] for score in ('changeScore','imageScore')}
for i,region in enumerate(regionLabels):
    for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
        ax = plt.subplot(gs[i,j])
        expScores = []
        for exp in result:
            for probe in result[exp]:
                if result[exp][probe]['region']==region:
                    scr = [s[0] for s in result[exp][probe]['active'][score]]
                    scr += [np.nan]*(len(nUnits)-len(scr))
                    expScores.append(scr)
                    allScores[score].append(scr)
                    ax.plot(nUnits,scr,'k')
        ax.plot(nUnits,np.nanmean(expScores,axis=0),'r',linewidth=2)
        for side in ('right','top'):
                ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks(np.arange(0,100,10))
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels([0,'',0.5,'',1])
        ax.set_xlim([0,max(nUnits)])
        ax.set_ylim([ymin,1])
        if i<len(regionLabels)-1:
            ax.set_xticklabels([])  
        if i==0:
            if j==0:
                ax.set_title(region+', '+score[:score.find('S')])
            else:
                ax.set_title(score[:score.find('S')])
        elif j==0:
            ax.set_title(region)
        if i==0 and j==0:
            ax.set_ylabel('Decoder Accuracy')
ax.set_xlabel('Number of Units')

plt.figure()
for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
    ax = plt.subplot(1,2,j+1)
    ax.plot(nUnits,np.nanmean(allScores[score],axis=0),'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(0,100,10))
    ax.set_yticks([0,0.25,0.5,0.75,1])
    ax.set_yticklabels([0,'',0.5,'',1])
    ax.set_xlim([0,max(nUnits)])
    ax.set_ylim([ymin,1])
    ax.set_title(score)
    ax.set_xlabel('Number of Units')
    if j==0:
        ax.set_ylabel('Decoder Accuracy')


# plot scores for each probe
for score,ymin in zip(('changeScore','imageScore'),[0.45,0]):
    fig = plt.figure(facecolor='w',figsize=(10,10))
    gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
    for i,region in enumerate(regionLabels):
        for j,state in enumerate(('active','passive')):
            ax = plt.subplot(gs[i,j])
            for exp in result:
                for probe in result[exp]:
                    if result[exp][probe]['region']==region:
                        for s in result[exp][probe][state][score]:
                            ax.plot(truncTimes,s,'k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xticks([0,50,100,150,200])
            ax.set_yticks([0,0.25,0.5,0.75,1])
            ax.set_yticklabels([0,'',0.5,'',1])
            ax.set_xlim([0,200])
            ax.set_ylim([ymin,1])
            if i<len(regionLabels)-1:
                ax.set_xticklabels([])
            if j>0:
                ax.set_yticklabels([])    
            if i==0:
                if j==0:
                    ax.set_title(region+', '+state)
                else:
                    ax.set_title(state)
            elif j==0:
                ax.set_title(region)
            if i==0 and j==0:
                ax.set_ylabel('Decoder Accuracy')
    ax.set_xlabel('Time (ms)')
    fig.text(0.5,0.95,score[:score.find('S')],fontsize=14,horizontalalignment='center')
    
# plot avg score for each area
regionColors = matplotlib.cm.jet(np.linspace(0,1,len(regionLabels)))
plt.figure(facecolor='w',figsize=(10,8))
gs = matplotlib.gridspec.GridSpec(2,2)
for i,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
    for j,state in enumerate(('active','passive')):
        ax = plt.subplot(gs[i,j])
        for region,clr in zip(regionLabels,regionColors):
            regionScore = []
            for exp in result:
                for probe in result[exp]:
                    if result[exp][probe]['region']==region:
                        regionScore.append(result[exp][probe][state][score])
            n = len(regionScore)
            if n>0:
                m = regionScore.mean(axis=0)
                ax.plot(truncTimes,m,color=clr,label=region+'(n='+str(n)+')')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels([0,'',0.5,'',1])
        ax.set_xlim([0,200])
        ax.set_ylim([ymin,1])
        if i==0:
            ax.set_xticklabels([])
            ax.set_title(state)
        else:
            ax.set_xlabel('Time (ms)')
        if j==0:
            ax.set_ylabel('Decoder Accuracy ('+score[:score.find('S')]+')')
        else:
            ax.set_yticklabels([])
        if i==1 and j==1:
            ax.legend()

# compare avg change and image scores for each area
plt.figure(facecolor='w',figsize=(10,10))
gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
for i,region in enumerate(regionLabels):
    for j,state in enumerate(('active','passive')):
        ax = plt.subplot(gs[i,j])
        for score,clr in zip(('changeScore','imageScore'),('k','0.5')):
            regionScore = []
            for exp in result:
                for probe in result[exp]:
                    if result[exp][probe]['region']==region:
                        regionScore.append(result[exp][probe][state][score])
            n = len(regionScore)
            if n>0:
                m = regionScore.mean(axis=0)
                ax.plot(truncTimes,m,color=clr,label=score[:score.find('S')])
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels([0,'',0.5,'',1])
        ax.set_xlim([0,200])
        ax.set_ylim([0,1])
        if i<len(regionLabels)-1:
            ax.set_xticklabels([])
        if j>0:
            ax.set_yticklabels([])    
        if i==0:
            if j==0:
                ax.set_title(region+', '+state)
            else:
                ax.set_title(state)
        elif j==0:
            ax.set_title(region)
        if i==0 and j==0:
            ax.set_ylabel('Decoder Accuracy')
        if i==len(regionLabels)-1 and j==1:
            ax.legend()
ax.set_xlabel('Time (ms)')

# plot active vs passive for each area and score
plt.figure(facecolor='w',figsize=(10,10))
gs = matplotlib.gridspec.GridSpec(len(regionLabels),2)
for i,region in enumerate(regionLabels):
    for j,(score,ymin) in enumerate(zip(('changeScore','imageScore'),(0.45,0))):
        ax = plt.subplot(gs[i,j])
        for state,clr in zip(('active','passive'),'rb'):
            regionScore = []
            for exp in result:
                for probe in result[exp]:
                    if result[exp][probe]['region']==region:
                        regionScore.append(result[exp][probe][state][score])
            n = len(regionScore)
            if n>0:
                m = regionScore.mean(axis=0)
                ax.plot(truncTimes,m,color=clr,label=state)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xticks([0,50,100,150,200])
        ax.set_yticks([0,0.25,0.5,0.75,1])
        ax.set_yticklabels([0,'',0.5,'',1])
        ax.set_xlim([0,200])
        ax.set_ylim([ymin,1])
        if i<len(regionLabels)-1:
            ax.set_xticklabels([])  
        if i==0:
            if j==0:
                ax.set_title(region+', '+score[:score.find('S')])
            else:
                ax.set_title(score[:score.find('S')])
        elif j==0:
            ax.set_title(region)
        if i==0 and j==0:
            ax.set_ylabel('Decoder Accuracy')
        if i==len(regionLabels)-1 and j==1:
            ax.legend()
ax.set_xlabel('Time (ms)')

# plot visual response, change decoding, and image decoding latencies
plt.figure(facecolor='w',figsize=(10,10))
latency = {'region': {state: {'resp':[],'change':[],'image':[]}}}
for region in regionLabels:
    for exp in result:
        for probe in result[exp]:
            if result[exp][probe]['region']==region:
                for state in ('active','passive'):
                    latency[region][state]['resp'].append(result[exp][probe][state]['respLatency'])
                    for score,decodeThresh in zip(('changeScore','imageScore'),(0.5,0.125)):
                        intpScore = np.interp(truncTimes,np.arange(truncTimes[0],truncTimes[-1]+1),result[exp][probe][state][score])
                        latency[region][state][score[:score.find('S')]].append(findLatency(intpScore,method='abs',thresh=decodeThresh))

plt.figure()
gs = matplotlib.gridspec.GridSpec(3,2)
label = {'resp':'Visual Response Latency','change':'Change Decoding Latency','image':'Image Decoding Latency'}
for i,(x,y) in enumerate((('resp','change'),('resp','image'),('change','image'))):
    for j,state in enumerate(('active','passive')):
        ax = plt.subplot(gs[i,j])
        for region,clr in zip(regionLabels,regionColors):
            ax.plot(latency[region][state][x],latency[region][state][y],'o',color=clr)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel(label[x])
        ax.set_ylabel(label[y])
                        


#
for exp in data:
    response = data[exp]['response'][:]
    trials = (response=='hit') | (response=='miss')
    behavior = np.zeros(trials.sum())
    behavior[response[trials]=='hit']=1
    
    state = 'active'
    for region in regionLabels:
        if exp in result:
            for probe in result[exp]:
                if result[exp][probe]=='region':
                    predictProb = result[exp][probe]['active']['changePredict']
            predict = predictProb>0.5
    
    ax = plt.subplot(1,1,1)
    ax.plot(behavior,'k')
    ax.plot(predict,'b')
    ax.plot(predictProb,'r')
    
    plt.plot(predictProb,behavior,'o')
    
    predictProb[behavior==1].mean()
    
    prediceProb[behavior<1].mean()
    
    np.corrcoef(predictProb,behavior)


