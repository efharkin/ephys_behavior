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
import pandas as pd
import matplotlib.pyplot as plt
from simlib import discretizer, tektronics, nidaq


# process analog signal
analogDataFile = fileIO.getFile() 
defaultDir = os.path.dirname(analogDataFile)

h = 0.000265 # arduino time step

analogData = pd.read_csv(analogDataFile,skiprows=2)
sampRate = 2500
data = np.array(analogData['Dev2/ai0'])
t = np.arange(0,data.size/sampRate,1/sampRate)

time,uo = discretizer.interpolate_data_to_timestep(t,data,h)

#start,stop = np.where((time>35) & (time<39))[0][[0,-1]]
#pointsToAnalyze = slice(start,stop)
pointsToAnalyze = slice(0,time.size)

time = time[pointsToAnalyze]
uo = uo[pointsToAnalyze]
filtered = []

# Band-Pass Butterworth Filter (2nd order) - Transfer Function
wo = 130.0 * (2 * np.pi)     # Filter Center Frequency (rad/s)
Q = 5                      # Band-Pass Quality Scalar
Ho = 2.0                   # Filter Gain Scalar
num = np.array([Ho*(wo/Q), 0])
den = np.array([1, wo/Q, wo*wo])
filtered.append(discretizer.apply_discrete_filter(num, den, h, uo))

# Rectify Signal
filtered.append(discretizer.apply_rectifier(filtered[0]))

# 2nd Order Tracking
K = 675
wn = 60 * (2 * np.pi)
lamda = 3
num = [K*wn]
den = [1, 2*lamda*wn, wn*wn]
filtered.append(discretizer.apply_discrete_filter(num, den, h, filtered[1]))

# derivative
numPointsToBaseline = 150
numPointsToAverage = 10
d = np.array(filtered[2])
dprime = d.copy()
dprime[:numPointsToBaseline] = 0
for i in range(numPointsToBaseline,d.size):
    baseline = d[i-numPointsToBaseline:i-numPointsToBaseline+numPointsToAverage].mean()
    dprime[i] = d[i-numPointsToAverage+1:i+1].mean()-baseline
filtered.append(dprime)

# Logic Gate
K = 1
hi_trigger = 0.04
lo_trigger = -0.01
filtered.append(discretizer.apply_schmidt_trigger(K, hi_trigger, lo_trigger, filtered[3]))


#for d,lbl in zip([uo]+filtered,['Analog Data','Bandpass','Rectify','Tracking','Derivative','Schmidt']):
#    fig = plt.figure(facecolor='w',figsize=(18,6))
#    ax = plt.subplot(1,1,1)
#    ax.plot(time,d,'k')
#    for side in ('right','top'):
#        ax.spines[side].set_visible(False)
#    ax.tick_params(direction='out',top=False,right=False,labelsize=12)
##    ax.set_xlim([44.2,46])
##    ax.set_ylim([0,3])
#    ax.set_xlabel('Time (s)',fontsize=14)
#    ax.set_ylabel(lbl,fontsize=14)
#    plt.tight_layout()


# get frame times and detected licks from sync file
syncFile = fileIO.getFile('choose sync file',defaultDir)
syncDataset = sync.Dataset(syncFile)

camFrameTimes = probeSync.get_sync_line_data(syncDataset,'cam1_exposure')[0]

detectedLickTimes = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
detectedLickTimes = detectedLickTimes[(detectedLickTimes>camFrameTimes[0]) & (detectedLickTimes<camFrameTimes[-1])]
detectedLickFrames = np.searchsorted(camFrameTimes,detectedLickTimes)

#camFile = fileIO.getFile('choose camera metadata file',defaultDir)
#camData = h5py.File(camFile,'r')
#frameIntervals = camData['frame_intervals'][:]
#camData.close()


# resample analog signals to frames
camExp = np.array(analogData[' Dev2/ai1'])
firstAnalogFrame = np.where(camExp[1:]>2.5)[0][0]
camFrameSamples = firstAnalogFrame+np.concatenate(([0],np.round(np.cumsum(np.diff(camFrameTimes))*sampRate).astype(int)))
tracking,deriv,detected = (discretizer.interpolate_data_to_timestep(time,filtered[i],1/sampRate)[1][camFrameSamples] for i in (-3,-2,-1))
detected = np.clip(detected,0.001,None,detected)


# make lick data file
lickDataFile = h5py.File(os.path.join(defaultDir,'lickData.hdf5'),'w',libver='latest')
paramData = (('frameTimes',np.full(camFrameTimes.size,np.nan)),
             ('reflectCenter',np.full((camFrameTimes.size,2),np.nan)),
             ('pupilCenter',np.full((camFrameTimes.size,2),np.nan)),
             ('pupilArea',tracking),
             ('pupilX',deriv),
             ('pupilY',detected),
             ('negSaccades',detectedLickFrames),
             ('posSaccades',np.array([]).astype(int)))
lickDataFile.attrs.create('mmPerPixel',np.nan)
for param,d in paramData:
    lickDataFile.create_dataset(param,data=d,compression='gzip',compression_opts=1)
lickDataFile.close()




# analyze annotated licks
def getFirstLicks(lickTimes,minBoutInterval=0.5):
    lickIntervals = np.diff(lickTimes)
    return lickTimes[np.concatenate(([0],np.where(lickIntervals>minBoutInterval)[0]+1))]
    
def compareLickTimes(annotatedLickTimes,detectedLickTimes,tolerance=0.1):
    falseNegatives = np.absolute(annotatedLickTimes-detectedLickTimes[:,None]).min(axis=0)>tolerance
    falsePositives = np.absolute(detectedLickTimes-annotatedLickTimes[:,None]).min(axis=0)>tolerance
    return falseNegatives,falsePositives
    
    
syncFile = fileIO.getFile('choose sync file')
syncDataset = sync.Dataset(syncFile)
camFrameTimes = probeSync.get_sync_line_data(syncDataset,'cam1_exposure')[0]
detectedLickTimes = probeSync.get_sync_line_data(syncDataset, 'lick_sensor')[0]
detectedLickTimes = detectedLickTimes[(detectedLickTimes>camFrameTimes[0]) & (detectedLickTimes<camFrameTimes[-1])]


lickDataFile = fileIO.getFile('choose lick data file')
lickData = h5py.File(lickDataFile,'r')
annotatedLickFrames = lickData['posSaccades'][:]
annotatedLickTimes = camFrameTimes[annotatedLickFrames]
#detectedLickFrames = lickData['negSaccades'][:]
#detectedLickTimes = camFrameTimes[detectedLickFrames]
lickData.close()


falseNeg,falsePos = compareLickTimes(annotatedLickTimes,detectedLickTimes)
print('all lick false negatives = '+str(falseNeg.sum())+'/'+str(falseNeg.size))

annotatedFirstLicks = getFirstLicks(annotatedLickTimes)
detectedFirstLicks = getFirstLicks(detectedLickTimes)

falseNegFirst,falsePosFirst = compareLickTimes(annotatedFirstLicks,detectedFirstLicks)
print('first lick false negatives = '+str(falseNegFirst.sum())+'/'+str(falseNegFirst.size))


tolerance = np.arange(0,0.2,1/60)
undetected = []
for tol in tolerance:
    fn,fp = compareLickTimes(annotatedLickTimes,detectedLickTimes,tol)
#    fn,fp = compareLickTimes(annotatedFirstLicks,detectedFirstLicks,tol)
    undetected.append(fn.sum())

fig = plt.figure(facecolor='w')
ax = plt.subplot(1,1,1)
ax.plot(tolerance,1-np.array(undetected)/annotatedLickTimes.size,'ko',ms=10)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False,labelsize=12)
ax.set_xlim([0,0.14])
ax.set_ylim([0,1.015])
ax.set_xlabel('Tolerance (s)',fontsize=14)
ax.set_ylabel('Fraction of licks detected',fontsize=14)
plt.tight_layout()

    




