# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

import os
import glob
from sync import sync
import probeSync
import behavSync
import numpy as np
import matplotlib.pyplot as plt
from probeData import formatFigure
from visual_behavior.visualization.extended_trials.daily import make_daily_figure


dataDir = "\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\Behavior\\09102018_385533"
sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = sync.Dataset(sync_file)

probeIDs = ('C',)
units = {str(pid): probeSync.getUnitData(dataDir,syncDataset, pid) for pid in probeIDs}

pkl_file = glob.glob(os.path.join(dataDir,'*[0-9].pkl'))[0]
trials, core_data, frameRising, frameFalling, runTime, runSpeed = behavSync.getBehavData(pkl_file,syncDataset)

make_daily_figure(trials)

#align trials to sync
trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameRising[trial_start_frames]
trial_end_times = frameFalling[trial_end_frames]

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
