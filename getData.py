# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os, glob, h5py, nrrd, cv2
from xml.dom import minidom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from probeData import formatFigure
from sync import sync
import probeSync
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
import scipy
from analysis_utils import find_run_transitions


dataDir = "\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\Behavior\\09202018_390339"
sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = sync.Dataset(sync_file)


# get probe data
probeIDs = ('A', 'B', 'C')

units = {str(pid): probeSync.getUnitData(dataDir,syncDataset, pid) for pid in probeIDs}


# get unit CCF positions
probeCCFFile = glob.glob(os.path.join(dataDir,'probePosCCF*.xlsx'))
if len(probeCCFFile)>0:
    probeCCF = pd.read_excel(probeCCFFile[0])
    ccfDir = '\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\CCF'            
    annotationStructures = minidom.parse(os.path.join(ccfDir,'annotationStructures.xml'))
    annotationData = nrrd.read(os.path.join(ccfDir,'annotation_25.nrrd'))[0].transpose((1,2,0))
    tipLength = 201
    for pid in probeIDs:
        entry,tip = [np.array(probeCCF[pid+' '+loc]) for loc in ('entry','tip')]
        for u in units[pid]:
            distFromTip = tipLength+units[pid][u]['position'][1]
            dx,dy,dz = [entry[i]-tip[i] for i in range(3)]
            trackLength = (dx**2+dy**2+dz**2)**0.5
            units[pid][u]['ccf'] = tip+np.array([distFromTip*d/trackLength for d in (dx,dy,dz)])
            units[pid][u]['ccfID'] = annotationData[tuple(int(units[pid][u]['ccf'][c]/25) for c in (1,0,2))]
            units[pid][u]['ccfRegion'] = None
            for ind,structID in enumerate(annotationStructures.getElementsByTagName('id')):
                if int(structID.childNodes[0].nodeValue)==units[pid][u]['ccfID']:
                    units[pid][u]['ccfRegion'] = annotationStructures.getElementsByTagName('structure')[ind].childNodes[7].childNodes[0].nodeValue[1:-1]
                    break
else:
    for pid in probeIDs:
        for u in units[pid]:
            for key in ('ccf','ccfID','ccfRegion'):
                units[pid][u][key] = None


# get behavior data
pkl_file = glob.glob(os.path.join(dataDir,'*[0-9].pkl'))[0]
behaviordata = pd.read_pickle(pkl_file)
core_data = data_to_change_detection_core(behaviordata)
trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'])
    
#make_daily_figure(trials)


# Get frame times from sync file
frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'stim_vsync')
frameTimes = frameFalling

# get running data
runTime = frameTimes[core_data['running'].frame]
runSpeed = core_data['running'].speed


# get eye tracking data
eyeFrameTimes = probeSync.get_sync_line_data(syncDataset,'cam2_exposure')[1]

#camPath = glob.glob(os.path.join(dataDir,'cameras','*-1.h5'))[0]
#camData = h5py.File(camPath)
#frameIntervals = camData['frame_intervals'][:]

eyeDataPath = glob.glob(os.path.join(dataDir,'cameras','*_eyetrack_analysis.hdf5'))
if len(eyeDataPath)>0:
    eyeData = h5py.File(eyeDataPath[0])
    pupilArea = eyeData['pupilArea'][:]
    pupilX = eyeData['pupilX'][:]
    negSaccades = eyeData['negSaccades'][:]
    posSaccades = eyeData['posSaccades'][:]
else:
    eyeData = None
    

# align trials to sync
trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameTimes[trial_start_frames]
trial_end_times = frameTimes[trial_end_frames]

# trial info
autoRewarded = np.array(trials['auto_rewarded']).astype(bool)
earlyResponse = np.array(trials['response_type']=='EARLY_RESPONSE')
ignore = earlyResponse | autoRewarded
miss = np.array(trials['response_type']=='MISS')
hit = np.array(trials['response_type']=='HIT')
falseAlarm = np.array(trials['response_type']=='FA')
correctReject = np.array(trials['response_type']=='CR')
initialImage = np.array(trials['initial_image_name'])
changeImage = np.array(trials['change_image_name'])


# get rf mapping stim info
images = core_data['image_set']['images']
newSize = tuple(int(s/10) for s in images[0].shape[::-1])
imagesDownsampled = [cv2.resize(img,newSize,interpolation=cv2.INTER_AREA) for img in images]
imageNames = [i['image_name'] for i in core_data['image_set']['image_attributes']]
rfstim_pickle_file = glob.glob(os.path.join(dataDir, '*brain_observatory_stimulus.pkl'))
if len(rfstim_pickle_file)>0:
    rf_stim_dict = pd.read_pickle(rfstim_pickle_file[0])
    rf_pre_blank_frames = int(rf_stim_dict['pre_blank_sec']*rf_stim_dict['fps'])
    rfstim = rf_stim_dict['stimuli'][0]
    monSizePix = rf_stim_dict['monitor']['sizepix']
    monHeightCm = monSizePix[1]/monSizePix[0]*rf_stim_dict['monitor']['widthcm']
    monDistCm = rf_stim_dict['monitor']['distancecm']
    monHeightDeg = np.degrees(2*np.arctan(0.5*monHeightCm/monDistCm))
    imagePixPerDeg = images[0].shape[0]/monHeightDeg 
    imageDownsamplePixPerDeg = imagesDownsampled[0].shape[0]/monHeightDeg


# get run start times
run_start_times = find_run_transitions(runSpeed, runTime)

# get lfp data
lfp = {str(pid): probeSync.getLFPdata(dataDir, pid, syncDataset) for pid in probeIDs}

#for pid in probeIDs:
#    plfp = lfp[pid][0]   
#    gammapower = []
#    thetapower = []
#    for i in np.arange(384):
#        f, pxx = scipy.signal.welch(plfp[:10000, i], fs=2500, nperseg=5000)
#        gammafreq = [30<ff<55 for ff in f]
#        gamma = np.mean(pxx[gammafreq])
#        gammapower.append(gamma)
#        
#        thetafreq = [5<ff<15 for ff in f]
#        theta = np.mean(pxx[thetafreq])
#        thetapower.append(theta)
#    
#    fig, ax = plt.subplots()
#    fig.suptitle(pid)
#    ax.plot(gammapower/max(gammapower), 'k')    
#    ax.plot(thetapower/max(thetapower), 'g')
#
#    unitchannels = [units[pid][u]['peakChan'] for u in probeSync.getOrderedUnits(units[pid])]
#    ax.plot(max(unitchannels), 1, 'ro')



