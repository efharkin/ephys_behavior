# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 12:42:18 2019

@author: svc_ccg
"""
import getData
import glob, os
import numpy as np
from analysis_utils import getSDF, calculate_lifetime_sparseness
from matplotlib import pyplot as plt
import probeSync
import summaryPlots
import pandas as pd
from probeData import formatFigure
import scipy.stats

def findFano(sdfs, usepeak=False, responseStart=270, responseEnd=520):
    if type(sdfs) is list:
        sdfs = np.array(sdfs)
    responseWindow = slice(responseStart, responseEnd)
    try:
        if usepeak:
            resps = sdfs[:, responseWindow].max(axis=1)
        else:
            resps = sdfs[:, responseWindow].mean(axis=1)
        
        return resps.var()/resps.mean()
    
    except:
        return np.nan
    

def determineResponsive(psth, responseSlice=slice(270,520), baselineSlice=slice(0,250), stdthreshold=5):
    ''' psth should be trialtype X mean response'''
    responsive = [(p[responseSlice].max() - p[baselineSlice].mean())>stdthreshold*p[baselineSlice].std() for p in psth]
    
#    responsive = [scipy.stats.wilcoxon(p[baselineSlice], p[responseSlice])[1] for p in psth]
    
#    return any([r < (pthresh/psth.shape[0]) for r in responsive])
    return any(responsive)

experiments = ['04042019_408528', '04052019_408528', '04102019_408527','04112019_408527', 
    '04252019_421323', '04262019_421323', '04302019_422856', '05162019_423749', '05172019_423749']

im_set = ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'A', 'B']
hdf5dir = r"C:\Users\svc_ccg\Desktop\Data\analysis"
regionDict = {r: [] for r in ['active_changePSTH', 'active_preChangePSTH', 'active_preChangeSparseness', 'active_changeSparseness', 
              'active_preChangeFano', 'active_changeFano', 'passive_changePSTH', 'passive_preChangePSTH', 'passive_changeSparseness',
              'passive_preChangeSparseness', 'passive_preChangeFano', 'passive_changeFano',
              'region', 'experiment', 'imageSet', 'unitID', 'probeID', 'totalSpikes', 'rfpvalue']}

preTime = 0.25
postTime = 0.5        
responseWindow = slice(270,520)
baseline = slice(0, 250)
for exp, imset in zip(experiments[1:], im_set[1:]):
    print(exp)
    b = getData.behaviorEphys('Z:\\' + exp)
    
    h5FilePath = glob.glob(os.path.join(hdf5dir, exp+'*'))[0]
    b.loadFromHDF5(h5FilePath) 
    
    selectedTrials = (b.hit | b.miss)&(~b.ignore)
    changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one to correct for change frame indexing problem
    image_flash_times = b.frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
    
    image_id = np.array(b.core_data['visual_stimuli']['image_name'])
    preChangeIndices = np.searchsorted(image_flash_times, changeTimes)-1
    preChangeTimes = image_flash_times[preChangeIndices]
    preChangeIDs = image_id[preChangeIndices]
    changeIDs = image_id[preChangeIndices+1] #double check to make sure this worked
    
    regionsOfInterest = ['VISam', 'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl']

    for pid in b.probes_to_analyze:
        for u in probeSync.getOrderedUnits(b.units[pid]):
            spikes = b.units[pid][u]['times']
            
            if np.sum(spikes<3600)<900 or np.sum(spikes>b.passiveFrameAppearTimes[-1]-3600)<900:
                continue
            
            #total spikes over hour of behavior
            regionDict['totalSpikes'].append(np.sum(spikes<3600))
            
            #p value for RF
            rfmat = summaryPlots.plot_rf(b, spikes, plot=False, returnMat=True)
            regionDict['rfpvalue'].append(rfmat[1].max())
            
            for frameAppearTimes, state in zip((b.frameAppearTimes, b.passiveFrameAppearTimes), ('active', 'passive')):
                flash_times = frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
                changeTimes = frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1]
                preChangeTimes = flash_times[preChangeIndices]
                
                changeFano = []
                preChangeFano = []
                changesdfs = []
                presdfs = []            
                for im in np.unique(preChangeIDs):
                    imPrechangeTimes = preChangeTimes[preChangeIDs==im]
                    imChangeTimes = changeTimes[changeIDs==im]
    
                    imPrechangeSDF, time = getSDF(spikes, imPrechangeTimes-preTime, preTime+postTime, avg=False, filt='exp', sigma=0.02)
                    imChangeSDF, time = getSDF(spikes, imChangeTimes-preTime, preTime+postTime, avg=False, filt='exp', sigma=0.02)
    
                    changesdfs.append(imChangeSDF)
                    presdfs.append(imPrechangeSDF)
#                    changeFano.append(findFano(imChangeSDF - np.mean(imChangeSDF[:, baseline], axis=1)[:, None]))
#                    preChangeFano.append(findFano(imPrechangeSDF - np.mean(imPrechangeSDF[:, baseline], axis=1)[:, None]))
                    changeFano.append(findFano(imChangeSDF))
                    preChangeFano.append(findFano(imPrechangeSDF))
                
                changeMean = np.array([s.mean(axis=0) for s in changesdfs])
                preChangeMean = np.array([s.mean(axis=0) for s in presdfs])
                changeMean_sub = changeMean - np.mean(changeMean[:, baseline], axis=1)[:, None]
                preChangeMean_sub = preChangeMean - np.mean(preChangeMean[:, baseline], axis=1)[:, None]
                    
                regionDict[state + '_preChangePSTH'].append(preChangeMean)
                regionDict[state +'_changePSTH'].append(changeMean)
                regionDict[state + '_preChangeFano'].append(preChangeFano)
                regionDict[state +'_changeFano'].append(changeFano)
                regionDict[state + '_preChangeSparseness'].append(calculate_lifetime_sparseness(preChangeMean_sub[:, responseWindow].max(axis=1)))
                regionDict[state + '_changeSparseness'].append(calculate_lifetime_sparseness(changeMean_sub[:, responseWindow].max(axis=1)))
                
            
            region = b.probeCCF[pid]['ISIRegion'] if b.units[pid][u]['inCortex'] else b.units[pid][u]['ccfRegion']
            regionDict['region'].append(region)
            regionDict['experiment'].append(exp)
            regionDict['unitID'].append(u)
            regionDict['probeID'].append(pid)
            regionDict['imageSet'].append(imset)
            

regiondf = pd.DataFrame.from_dict(regionDict)



#compare baseline and resp amp across active/passive epochs
for state in ['active', 'passive']:
    for psth in ['changePSTH', 'preChangePSTH']:
        psths = regiondf[state + '_' + psth].values
        pmeans = np.array([p.mean(axis=0) for p in psths])
        baselines = pmeans[:, baseline].mean(axis=1)
        respAmp = pmeans[:, responseWindow].max(axis=1) - baselines
        
        regiondf[state + '_' + psth + '_baseline'] = baselines
        regiondf[state + '_' + psth + '_responseAmplitude'] = respAmp
        

fig, ax = plt.subplots(2)
for ir, r in enumerate(np.unique(regionDict['region'])):
    for state, color in zip(['active', 'passive'], ['r', 'b']):
        for psth, alpha in zip(['changePSTH', 'preChangePSTH'], [1, 0.5]):
            baselines = regiondf.loc[regiondf['region']==r, state + '_' + psth + '_baseline']    
            respAmp = regiondf.loc[regiondf['region']==r, state + '_' + psth + '_responseAmplitude']
            
            ax[0].plot(ir, baselines.mean(), color+'o', alpha=alpha)
            ax[1].plot(ir, respAmp.mean(), color+'o', alpha=alpha)

ax[0].set_xticks(np.arange(len(np.unique(regionDict['region']))))
ax[1].set_xticks(np.arange(len(np.unique(regionDict['region']))))
ax[0].set_xticklabels(np.unique(regionDict['region']), rotation=90)
ax[1].set_xticklabels(np.unique(regionDict['region']), rotation=90)

[a.set_xlim([0.5,len(np.unique(regionDict['region']))]) for a in ax]
[formatFigure(fig, a, yLabel=label) for a,label in zip(ax, ['baseline FR (Hz)', 'Resp. Amp. (Hz)'])]


#Active vs Passive pop response across areas
regionsOfInterest = ['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']
time = np.arange(-250, 500)
fig, ax = plt.subplots(1, len(regionsOfInterest))
for r, a in zip(regionsOfInterest, ax):
    
    df = regiondf.loc[(regiondf['region']==r)]
    
    #find cells with sig change response
    psths = df['active_changePSTH'].values
    pmeans = np.array([p.mean(axis=0) for p in psths])
    responsive = pmeans[:, responseWindow].max(axis=1) > 5*pmeans[:, baseline].std(axis=1)
    
    for state, color in zip(['active', 'passive'], ['r', 'b']):
        
#        for psth, alpha in zip(['changePSTH', 'preChangePSTH'], [1, 0.5]):
        for psth, alpha in zip(['changePSTH'], [1]):
            psths = df[state + '_' + psth].values
            pmeans = np.array([p.mean(axis=0) for p in psths[responsive]])
            
            allmean = pmeans.mean(axis=0)
            allsem = pmeans.std(axis=0)/pmeans.shape[0]**0.5
            
            a.plot(time, allmean, color)
            a.fill_between(time, allmean+allsem, allmean-allsem, color=color, alpha=0.5)
    
    a.set_title(r)
    a.text(-100, 19, str(np.sum(responsive)))
    formatFigure(fig, a)

ymax = np.max([a.get_ylim()[1] for a in ax])
[a.set_ylim([0, ymax]) for a in ax]
[a.axes.get_yaxis().set_visible(False) for a in ax[1:]]

    

#sparseness and reliability active vs passive
regionsOfInterest = ['VISp', 'VISl', 'VISal', 'VISrl', 'VISpm', 'VISam']
#regionsOfInterest = np.unique(regionDict['region'])
time = np.arange(-250, 500)
fig, ax = plt.subplots(2)
figsp, axsp = plt.subplots()
respSTDthresh = 10
for ir, r in enumerate(regionsOfInterest):
    
    df = regiondf.loc[(regiondf['region']==r)&(regiondf['rfpvalue']==100)]
    
    #find cells with sig change response
    psths = df['active_changePSTH'].values
    pmeans = np.array([p.mean(axis=0) for p in psths])
    pmeans_sub = pmeans - pmeans[:, baseline].mean(axis=1)[:, None]
    responsive = [determineResponsive(p, stdthreshold=respSTDthresh) for p in psths]
    
    
#    rfig, rax = plt.subplots()
#    rfig.suptitle(r)
#    ppeaks = [p[:, responseWindow].max(axis=1) for p in psths[responsive]]
#    rax.imshow(ppeaks, cmap='plasma', aspect='auto')
    
    
    for state, color in zip(['active', 'passive'], ['r', 'b']):
        for resp, alpha in zip(['change', 'preChange'], [1, 0.5]):
            sparseness = df[state+'_'+resp+'Sparseness'].values[responsive]
            fano = df[state+'_'+resp+'Fano'].values[responsive]
            fano = np.array([np.nanmean(f) for f in fano])
            
            ax[0].plot(ir, np.nanmean(sparseness), color+'o', alpha=alpha)
            ax[1].plot(ir, np.nanmean(fano), color+'o', alpha=alpha)
            
    for imageSet, color in zip(['A', 'B'], ['k', 'g']):
        imdf = df.loc[df['imageSet']==imageSet]
        psths = imdf['active_changePSTH'].values
        pmeans = np.array([p.mean(axis=0) for p in psths])
        pmeans_sub = pmeans - pmeans[:, baseline].mean(axis=1)[:, None]
        responsivesp = [determineResponsive(p, stdthreshold=respSTDthresh) for p in psths]
        print(r+imageSet+ ': ' + str(np.sum(responsivesp)))
        for resp, alpha in zip(['change', 'preChange'], [1, 0.5]):
            sparseness = imdf['active_'+resp+'Sparseness'].values[responsivesp]
            axsp.plot(ir, np.nanmean(sparseness), color+'o', alpha=alpha)

formatFigure(figsp, axsp)
axsp.set_xticks(np.arange(len(regionsOfInterest)))
axsp.set_xticklabels(regionsOfInterest, rotation=90) 

ax[0].set_xticks(np.arange(len(regionsOfInterest)))
ax[1].set_xticks(np.arange(len(regionsOfInterest)))
ax[0].set_xticklabels(regionsOfInterest, rotation=90)
ax[1].set_xticklabels(regionsOfInterest, rotation=90)

#[a.set_xlim([0.5,len(np.unique(regionDict['region']))]) for a in ax]
[formatFigure(fig, a, yLabel=label) for a,label in zip(ax, ['Lifetime Sparseness', 'Fano Factor'])]  
    
    


