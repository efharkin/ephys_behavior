# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:38:46 2018

@author: svc_ccg
"""

import probeSync
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
import pandas as pd


def getBehavData(pkl_file,syncDataset):
    #Get frame times from sync file
    frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'stim_vsync')
    
    #Get trial data pkl behavior file
    behaviordata = pd.read_pickle(pkl_file)
    core_data = data_to_change_detection_core(behaviordata)
    trials = create_extended_dataframe(
        trials=core_data['trials'],
        metadata=core_data['metadata'],
        licks=core_data['licks'],
        time=core_data['time'])
    
    runTime = frameRising[core_data['running'].frame]
    runSpeed = core_data['running'].speed
    
    return trials, frameRising, frameFalling, runTime, runSpeed