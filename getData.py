# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os, glob, h5py, nrrd
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


dataDir = "\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\Behavior\\09172018_385533"
sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = sync.Dataset(sync_file)


# get probe data
probeIDs = ('A', 'C', 'B')
units = {str(pid): probeSync.getUnitData(dataDir,syncDataset, pid) for pid in probeIDs}


# get unit CCF positions
probeCCFFile = glob.glob(os.path.join(dataDir,'histology','probePosCCF_*.xlsx'))[0]
probeCCF = pd.read_excel(probeCCFFile,sheetname=dataDir[-15:-7])

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


# get behavior data
pkl_file = glob.glob(os.path.join(dataDir,'*[0-9].pkl'))[0]
behaviordata = pd.read_pickle(pkl_file)
core_data = data_to_change_detection_core(behaviordata)
trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'])
    
make_daily_figure(trials)


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
imageNames = np.unique(changeImage)
