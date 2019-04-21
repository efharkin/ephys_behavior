# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 11:29:39 2018

@author: svc_ccg
"""

from __future__ import division
import os, glob, h5py, nrrd, cv2, datetime
from xml.dom import minidom
import numpy as np
import pandas as pd
import fileIO
import matplotlib.pyplot as plt
from probeData import formatFigure
from sync import sync
import probeSync
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
import scipy
from analysis_utils import find_run_transitions

#Parent directory with sorted data, sync and pkl files
dataDir = 'Z:\\03272019_417882'

#Which probes to run through analysis
probeIDs = ['A', 'B', 'C',  'F']

class behaviorEphys():
    
    def __init__(self, baseDir=None, probes=None):
        if baseDir is None:        
            self.dataDir = dataDir
        else:
            self.dataDir = baseDir
        self.sync_file = glob.glob(os.path.join(self.dataDir, '*.h5'))[0]
        self.syncDataset = sync.Dataset(self.sync_file)
        if probes is None:
            self.probes_to_analyze = probeIDs
        else:
            self.probes_to_analyze = probes
        self.experimentDate = os.path.basename(self.dataDir)[:8]
        if datetime.datetime.strptime(self.experimentDate,'%m%d%Y') < datetime.datetime(2019,3,15):
            fprobe = '4'
        else:
            fprobe = '3'
        self.PXIDict = {'A': 'slot2-probe1', 'B': 'slot2-probe2', 'C': 'slot2-probe3', 'D': 'slot3-probe1', 'E': 'slot3-probe2', 'F': 'slot3-probe'+fprobe}
        
    def saveHDF5(self,filePath=None):
        fileIO.objToHDF5(self,filePath)
    
    def loadFromHDF5(self, filePath=None):
        fileIO.hdf5ToObj(self,filePath)
    
    def loadFromRawData(self):
        self.getUnits()
        #self.getLFP()
        self.getCCFPositions()
        self.getFrameTimes()
        self.getBehaviorData()
        self.getEyeTrackingData()
        self.getRFandFlashStimInfo()
        self.getPassiveStimInfo()
        
    def getUnits(self):    
        self.units = {str(pid): probeSync.getUnitData(self.dataDir, self.syncDataset, pid, self.PXIDict) for pid in self.probes_to_analyze}
    
    def getLFP(self):
        self.lfp = {str(pid): probeSync.getLFPdata(self.dataDir, pid, self.syncDataset) for pid in self.probes_to_analyze}
    
    def getCCFPositions(self):
        # get unit CCF positions
        self.probeCCFFile = glob.glob(os.path.join(self.dataDir,'probePosCCF*.xlsx'))
        if len(self.probeCCFFile)>0:
            probeCCF = pd.read_excel(self.probeCCFFile[0])
            ccfDir = '\\\\allen\\programs\\braintv\\workgroups\\nc-ophys\\corbettb\\CCF'            
            annotationStructures = minidom.parse(os.path.join(ccfDir,'annotationStructures.xml'))
            annotationData = nrrd.read(os.path.join(ccfDir,'annotation_25.nrrd'))[0].transpose((1,2,0))
            tipLength = 201
            for pid in self.probes_to_analyze:
                entry,tip = [np.array(probeCCF[pid+' '+loc]) for loc in ('entry','tip')]
                for u in self.units[pid]:
                    distFromTip = tipLength+self.units[pid][u]['position'][1]
                    dx,dy,dz = [entry[i]-tip[i] for i in range(3)]
                    trackLength = (dx**2+dy**2+dz**2)**0.5
                    self.units[pid][u]['ccf'] = tip+np.array([distFromTip*d/trackLength for d in (dx,dy,dz)])
                    self.units[pid][u]['ccfID'] = annotationData[tuple(int(self.units[pid][u]['ccf'][c]/25) for c in (1,0,2))]
                    self.units[pid][u]['ccfRegion'] = None
                    for ind,structID in enumerate(annotationStructures.getElementsByTagName('id')):
                        if int(structID.childNodes[0].nodeValue)==self.units[pid][u]['ccfID']:
                            self.units[pid][u]['ccfRegion'] = annotationStructures.getElementsByTagName('structure')[ind].childNodes[7].childNodes[0].nodeValue[1:-1]
                            break
        else:
            for pid in self.probes_to_analyze:
                for u in self.units[pid]:
                    for key in ('ccf','ccfID','ccfRegion'):
                        self.units[pid][u][key] = None
    
            if os.path.isfile(os.path.join(os.path.dirname(self.dataDir), 'hippocampusChannels.xlsx')):
                print('assigning hippocampus channels')
                hippoFile = os.path.join(os.path.dirname(self.dataDir), 'hippocampusChannels.xlsx')
                hippodf = pd.read_excel(hippoFile, sheetname=os.path.basename(self.dataDir))
                for pid in self.probes_to_analyze:
                    hippoendchan = int(hippodf[hippodf.Probe==pid].hipp_end_chan)
                    cortexendchan = int(hippodf[hippodf.Probe==pid].cortex_end_chan)
                    for u in self.units[pid]:
                        if self.units[pid][u]['peakChan']<hippoendchan:
                            self.units[pid][u]['ccfRegion'] = 'hipp'
                        elif self.units[pid][u]['peakChan']>cortexendchan:
                            self.units[pid][u]['ccfRegion'] = 'air'
                
    def getFrameTimes(self):
        # Get frame times from sync file
        frameRising, frameFalling = probeSync.get_sync_line_data(self.syncDataset, 'stim_vsync')

        #diode = probeSync.get_sync_line_data(syncDataset, 'photodiode')
        #monitorLags = diode[0][4:4+frameFalling[60::120].size][:100] - frameFalling[60::120][:100]
        
        self.vsyncTimes = frameFalling #use for all data streams except the stimulus frame times, which are subject to monitor lag
        monitorLag = 0.036
        self.frameAppearTimes = frameFalling + monitorLag    
    
    
    def getBehaviorData(self):
        # get behavior data
        if not hasattr(self, 'vsyncTimes'):
            self.getFrameTimes()
            
        self.pkl_file = glob.glob(os.path.join(self.dataDir,'*[0-9].pkl'))[0]
        behaviordata = pd.read_pickle(self.pkl_file)
        self.core_data = data_to_change_detection_core(behaviordata)

        self.images = self.core_data['image_set']['images']
        newSize = tuple(int(s/10) for s in self.images[0].shape[::-1])
        self.imagesDownsampled = [cv2.resize(img,newSize,interpolation=cv2.INTER_AREA) for img in self.images]
        self.imageNames = [i['image_name'] for i in self.core_data['image_set']['image_attributes']]
        
        self.trials = create_extended_dataframe(
            trials=self.core_data['trials'],
            metadata=self.core_data['metadata'],
            licks=self.core_data['licks'],
            time=self.core_data['time'])
        # get running data
        self.behaviorRunTime = self.vsyncTimes[self.core_data['running'].frame]
        self.behaviorRunSpeed = self.core_data['running'].speed
        
        # get run start times
        self.behaviorRunStartTimes = find_run_transitions(self.behaviorRunSpeed, self.behaviorRunTime)
        
        #make_daily_figure(trials)
    
        # align trials to sync
        self.trial_start_frames = np.array(self.trials['startframe'])
        self.trial_end_frames = np.array(self.trials['endframe'])
        self.trial_start_times = self.frameAppearTimes[self.trial_start_frames]
        self.trial_end_times = self.frameAppearTimes[self.trial_end_frames]
        
        # trial info
        self.autoRewarded = np.array(self.trials['auto_rewarded']).astype(bool)
        self.earlyResponse = np.array(self.trials['response_type']=='EARLY_RESPONSE')
        self.ignore = self.earlyResponse | self.autoRewarded
        self.miss = np.array(self.trials['response_type']=='MISS')
        self.hit = np.array(self.trials['response_type']=='HIT')
        self.falseAlarm = np.array(self.trials['response_type']=='FA')
        self.correctReject = np.array(self.trials['response_type']=='CR')
        self.initialImage = np.array(self.trials['initial_image_name'])
        self.changeImage = np.array(self.trials['change_image_name'])
        candidateOmittedFlashFrames = behaviordata['items']['behavior']['stimuli']['images']['flashes_omitted']
        drawlog = behaviordata['items']['behavior']['stimuli']['images']['draw_log']
        self.omittedFlashFrames = np.array([c for c in candidateOmittedFlashFrames if not drawlog[c]])
        imageFrameIndexBeforeOmitted = np.searchsorted(self.core_data['visual_stimuli']['frame'], self.omittedFlashFrames)-1
        self.omittedFlashImage = np.array(self.core_data['visual_stimuli']['image_name'])[imageFrameIndexBeforeOmitted]
        
        self.behaviorStimDur = np.array(self.core_data['visual_stimuli']['duration'])
        self.preGrayDur = np.stack(self.trials['blank_duration_range']) # where is actual gray dur
        self.lastBehaviorTime = self.frameAppearTimes[self.trials['endframe'].values[-1]]    
        
    
    def getEyeTrackingData(self):
        # get eye tracking data
        self.eyeFrameTimes = probeSync.get_sync_line_data(self.syncDataset,'cam2_exposure')[0]
        
        #camPath = glob.glob(os.path.join(dataDir,'cameras','*-1.h5'))[0]
        #camData = h5py.File(camPath)
        #frameIntervals = camData['frame_intervals'][:]
        
        self.eyeDataPath = glob.glob(os.path.join(self.dataDir,'cameras','*_eyetrack_analysis.hdf5'))
        if len(self.eyeDataPath)>0:
            self.eyeData = h5py.File(self.eyeDataPath[0])
            self.pupilArea = self.eyeData['pupilArea'][:]
            self.pupilX = self.eyeData['pupilX'][:]
            self.negSaccades = self.eyeData['negSaccades'][:]
            self.posSaccades = self.eyeData['posSaccades'][:]
        else:
            self.eyeData = None
            
    
    def getRFandFlashStimInfo(self):
        self.rf_pickle_file = glob.glob(os.path.join(self.dataDir, '*brain_observatory_stimulus.pkl'))
        if len(self.rf_pickle_file)>0:
            self.rfFlashStimDict = pd.read_pickle(self.rf_pickle_file[0])
            self.monSizePix = self.rfFlashStimDict['monitor']['sizepix']
            self.monHeightCm = self.monSizePix[1]/self.monSizePix[0]*self.rfFlashStimDict['monitor']['widthcm']
            self.monDistCm = self.rfFlashStimDict['monitor']['distancecm']
            self.monHeightDeg = np.degrees(2*np.arctan(0.5*self.monHeightCm/self.monDistCm))
            self.imagePixPerDeg = self.images[0].shape[0]/self.monHeightDeg 
            self.imageDownsamplePixPerDeg = self.imagesDownsampled[0].shape[0]/self.monHeightDeg
            
            self.rfStimParams = self.rfFlashStimDict['stimuli'][0]
            rf_pre_blank_frames = int(self.rfFlashStimDict['pre_blank_sec']*self.rfFlashStimDict['fps'])
            first_rf_frame = self.trials['endframe'].values[-1] + rf_pre_blank_frames + 1
            self.rf_frameTimes = self.frameAppearTimes[first_rf_frame:]
            self.rf_trial_start_times = self.rf_frameTimes[np.array([f[0] for f in np.array(self.rfStimParams['sweep_frames'])]).astype(np.int)]
            
            self.flashStimParams = self.rfFlashStimDict['stimuli'][1]
            
    def getPassiveStimInfo(self):
        self.passive_pickle_file = glob.glob(os.path.join(self.dataDir, '*-replay-script*.pkl'))
        if len(self.passive_pickle_file)>0:
            passiveStimDict = pd.read_pickle(self.passive_pickle_file[-1])
            self.passiveStimParams = passiveStimDict['stimuli'][0]
            self.passiveFrameImages = np.array(self.passiveStimParams['sweep_params']['ReplaceImage'][0])
            passiveImageNames = [img for img in np.unique(self.passiveFrameImages) if img is not None]
            nonGrayFrames = np.in1d(self.passiveFrameImages,passiveImageNames)
            self.passiveImageOnsetFrames = np.where(np.diff(nonGrayFrames.astype(int))>0)[0]+1
            self.passiveChangeFrames = np.array([frame for i,frame in enumerate(self.passiveImageOnsetFrames[1:]) if self.passiveFrameImages[frame]!=self.passiveFrameImages[self.passiveImageOnsetFrames[i]]])
            self.passiveChangeImages = self.passiveFrameImages[self.passiveChangeFrames]
            
            abortedVsyncs = 0
            if len(self.passive_pickle_file)>1:
                for f in self.passive_pickle_file[:-1]:
                    d = pd.read_pickle(f)
                    abortedVsyncs += d['vsynccount']
            firstPassiveFrame = self.trials['endframe'].values[-1] + self.rfFlashStimDict['vsynccount'] + abortedVsyncs + 1
            self.passiveFrameAppearTimes = self.frameAppearTimes[firstPassiveFrame:]
            
    
    
    
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
    


