# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 14:11:06 2019

@author: svc_ccg
"""

from __future__ import division
import h5py, os
import fileIO
from sync import sync
import probeSync
import numpy as np
import matplotlib.pyplot as plt


def addDetectedLicksToAnnotationFile(lickDataFile,detectedLickFrames):
    lickData = h5py.File(lickDataFile,'r')
    newFile = h5py.File(lickDataFile[:-5]+'_withDetected'+'.hdf5','w',libver='latest')
    for key in lickData.attrs.keys():
        newFile.attrs.create(key,lickData.attrs[key])
    for key in lickData.keys():
        if key!='negSaccades':
            newFile.create_dataset(key,data=lickData[key][:],compression='gzip',compression_opts=1)
    newFile.create_dataset('negSaccades',data=detectedLickFrames,compression='gzip',compression_opts=1)
    lickData.close()
    newFile.close()

def getFirstLicks(lickTimes,minBoutInterval=0.5):
    lickIntervals = np.diff(lickTimes)
    return lickTimes[np.concatenate(([0],np.where(lickIntervals>minBoutInterval)[0]+1))]


syncFile = fileIO.getFile('choose sync file')
defaultDir = os.path.dirname(syncFile)
syncDataset = sync.Dataset(syncFile)

camFrameTimes = probeSync.get_sync_line_data(syncDataset,'cam1_exposure')[0]

detectedLickTimes = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
detectedLickTimes = detectedLickTimes[(detectedLickTimes>camFrameTimes[0]) & (detectedLickTimes<camFrameTimes[-1])]
detectedLickFrames = np.searchsorted(camFrameTimes,detectedLickTimes)

#camFile = fileIO.getFile('choose camera metadata file',defaultDir)
#camData = h5py.File(camFile,'r')
#frameIntervals = camData['frame_intervals'][:]
#camData.close()

lickDataFile = fileIO.getFile('choose lick data file',defaultDir)
lickData = h5py.File(lickDataFile,'r')
roiIntensity = lickData['pupilArea'][:]
annotatedLickFrames = lickData['posSaccades'][:]
annotatedLickTimes = camFrameTimes[annotatedLickFrames]
lickData.close()


firstLicks = getFirstLicks(annotatedLickTimes)
detectedFirstLicks = getFirstLicks(detectedLickTimes)


lickComparisonTolerance = 0.1

falseNegatives = np.absolute(firstLicks-detectedFirstLicks[:,None]).min(axis=0)>lickComparisonTolerance


falsePositives = np.absolute(detectedFirstLicks-firstLicks[:,None]).min(axis=0)>lickComparisonTolerance




