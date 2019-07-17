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

def findFano(sdfs, usepeak=True, responseStart=270, responseEnd=520):
    if type(sdfs) is list:
        sdfs = np.array(sdfs)
    responseWindow = slice(responseStart, responseEnd)
    if usepeak:
        resps = sdfs[:, responseWindow].max(axis=1)
    else:
        resps = sdfs[:, responseWindow].mean(axis=1)
        
    return resps.std()**2/resps.mean()
    
    

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
for exp, imset in zip([experiments[0]], [im_set[0]]):
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
                    changeFano.append(findFano(imChangeSDF - np.mean(imChangeSDF[:, baseline], axis=1)[:, None]))
                    preChangeFano.append(findFano(imPrechangeSDF - np.mean(imPrechangeSDF[:, baseline], axis=1)[:, None]))
                
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

