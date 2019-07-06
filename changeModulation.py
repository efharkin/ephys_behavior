# -*- coding: utf-8 -*-
"""
Created on Mon Jul 01 12:48:27 2019

@author: svc_ccg
"""
import getData
import glob, os
import numpy as np
from analysis_utils import *
from matplotlib import pyplot as plt

experiments = ['03212019_409096', '03262019_417882', '03272019_417882', '04042019_408528', '04052019_408528', '04102019_408527','04112019_408527', 
    '04252019_421323', '04262019_421323', '04302019_422856', '05162019_423749', '05172019_423749']

hdf5dir = r"C:\Users\svc_ccg\Desktop\Data\analysis"
rds = []
for exp in experiments:
    print(exp)
    b = getData.behaviorEphys('Z:\\' + exp)
    
    h5FilePath = glob.glob(os.path.join(hdf5dir, exp+'*'))[0]
    b.loadFromHDF5(h5FilePath) 
    
    selectedTrials = (b.hit | b.miss)&(~b.ignore)
    changeTimes = b.frameAppearTimes[np.array(b.trials['change_frame'][selectedTrials]).astype(int)+1] #add one to correct for change frame indexing problem
    image_flash_times = b.frameAppearTimes[np.array(b.core_data['visual_stimuli']['frame'])]
    preChangeIndices = np.searchsorted(image_flash_times, changeTimes)-1
    preChangeTimes = image_flash_times[preChangeIndices]
        
    
    regionsOfInterest = ['VISam', 'VISpm', 'VISp', 'VISl', 'VISal', 'VISrl']
    
    preTime = 0.25
    postTime = 0.5
    regionDict = {r: {a:[] for a in ['changePSTH', 'preChangePSTH']} for r in regionsOfInterest}
    regionDict['experiment'] = exp
    for region in regionsOfInterest:
        pids, us = b.getUnitsByArea(region)
        
        for pid, u in zip(pids,us):
            spikes = b.units[pid][u]['times']
            
            if np.sum(spikes<3600)<3600: #exclude cells that fire below 1Hz during task
                continue
            
            changesdf, time = getSDF(spikes, changeTimes-preTime, preTime+postTime, sigma=0.001)            
            regionDict[region]['changePSTH'].append(changesdf)

            prechangesdf, time = getSDF(spikes, preChangeTimes-preTime, preTime+postTime, sigma=0.001)            
            regionDict[region]['preChangePSTH'].append(prechangesdf)
    
    rds.append(regionDict)

plt.figure()
allregionDict = {r: {a:[] for a in ['changePSTH', 'preChangePSTH']} for r in regionsOfInterest}   
for region in regionsOfInterest:
    meanChangeModPerExperiment = []
    for ir, r in enumerate(rds):
        allregionDict[region]['changePSTH'].extend(r[region]['changePSTH'])
        allregionDict[region]['preChangePSTH'].extend(r[region]['preChangePSTH'])    
#        
#        print(region + ': ' + str(len(r[region]['changePSTH'])))
#        if len(r[region]['changePSTH'])>0:
#            change = np.array(r[region]['changePSTH'])
#            prechange = np.array(r[region]['preChangePSTH'])    
#            
#            change_sub = change - np.mean(change[:, :200], axis=1)[:, None]
#            change_norm = change_sub/np.max(change_sub, axis=1)[:, None]
#            
#            prechange_sub = prechange - np.mean(prechange[:, :200], axis=1)[:, None]
#            prechange_norm = prechange_sub/np.max(change_sub, axis=1)[:, None]  
#            
#            diff = change_sub - prechange_sub
#            changeMod = np.log2(diff.max(axis=1)/prechange_sub.max(axis=1))
#            
#            meanChangeModPerExperiment.append(changeMod.mean())
#            print('change mod: ' + str(changeMod.mean()))
#            
#            changeMean = np.mean(change_sub, axis=0)
#            preChangeMean = np.mean(prechange_sub, axis=0)
#            diffMean = np.mean(diff, axis=0)
#            plt.figure(region + ' ' + str(ir))
#            plt.plot(changeMean)
#            plt.plot(preChangeMean)
#            plt.plot(diffMean)
#        
#    print('region ' + region + ' MEAN: ' + str(np.mean(meanChangeModPerExperiment)))
            

    change = np.array(allregionDict[region]['changePSTH'])
    prechange = np.array(allregionDict[region]['preChangePSTH'])    
    
    change_sub = change - np.mean(change[:, :250], axis=1)[:, None]
    change_norm = change_sub/np.max(change_sub, axis=1)[:, None]
    
    prechange_sub = prechange - np.mean(prechange[:, :250], axis=1)[:, None]
    prechange_norm = prechange_sub/np.max(change_sub, axis=1)[:, None]  
    
    diff = change_sub - prechange_sub
    changeMod = np.log2(diff[:, 280:530].max(axis=1)/prechange_sub[:, 280:530].max(axis=1))
#    plt.figure(region)
#    plt.hist(changeMod, bins=np.arange(-0.5, 0.5, 0.05))
    
#    plt.figure(region + 'changeMod vs Response amp')
#    plt.plot(prechange_sub.max(axis=1), changeMod, 'ko', alpha=0.5)
    print(region + ': ' + str(np.median(changeMod[~np.isnan(changeMod)])))
    print('n = ' + str(np.sum(~np.isnan(changeMod))))
    
    
    changeMean = np.mean(change_sub, axis=0)
    preChangeMean = np.mean(prechange_sub, axis=0)
    diffMean = np.mean(diff, axis=0)

            
#    plt.figure(region)
    plt.plot(changeMean)
#    plt.plot(preChangeMean)
#    plt.plot(diffMean)
    plt.legend(regionsOfInterest)
    
    
for ir, r in enumerate(rds):
    fig, axes = plt.subplots(3,2)
    fig.suptitle(experiments[ir])
    for ir, region in enumerate(regionsOfInterest):   
        axrow = ir%3
        axcol = int(ir>=3)
        ax = axes[axrow, axcol]
        if len(r[region]['changePSTH'])>0:
            change = np.array(r[region]['changePSTH'])
            prechange = np.array(r[region]['preChangePSTH'])    
            
            change_sub = change - np.mean(change[:, :250], axis=1)[:, None]
            prechange_sub = prechange - np.mean(prechange[:, :250], axis=1)[:, None]
            
            diff = change_sub - prechange_sub
            changeMod = np.log2(diff.max(axis=1)/prechange_sub.max(axis=1))
            
            changeMean = np.mean(change_sub, axis=0)
            preChangeMean = np.mean(prechange_sub, axis=0)
            diffMean = np.mean(diff, axis=0)
            ax.set_title(region)
            ax.plot(changeMean)
            ax.plot(preChangeMean)
            ax.plot(diffMean)
            ax.text(100, 5, 'n= ' + str(len(r[region]['changePSTH'])))