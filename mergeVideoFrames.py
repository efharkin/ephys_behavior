# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 12:57:18 2019

@author: svc_ccg
"""

from __future__ import division
import os
import glob
import cv2
import numpy as np


rootDir = r'\\allen\programs\braintv\workgroups\nc-ophys\corbettb\cameratest\07302019_432291_2cams_90fps'


# get frame times from sync file

#import probeSync
#from sync import sync

#syncDataset = sync.Dataset(glob.glob(os.path.join(rootDir,'*_'+('[0-9]'*12)+'.h5'))[0])
#for cam in (0,1):
#    frameTimes = probeSync.get_sync_line_data(syncDataset,'cam'+str(cam+1)+'_exposure')[0]
#    np.save(os.path.join(rootDir,'cam'+str(cam)+'_frametimes.npy'),frameTimes)


# load video files
cam0,cam1 = [cv2.VideoCapture(glob.glob(os.path.join(rootDir,'*-'+str(cam)+'.avi'))[0]) for cam in (0,1)]


# get video frame rate and frame count
(cam0FrameRate,cam0FrameCount),(cam1FrameRate,cam1FrameCount) = \
[[cam.get(p) for p in (cv2.CAP_PROP_FPS,cv2.CAP_PROP_FRAME_COUNT)] for cam in (cam0,cam1)]


# load sync frame times
cam0FrameTimes,cam1FrameTimes = [np.load(os.path.join(rootDir,'cam'+str(cam)+'_frametimes.npy')) for cam in (0,1)]


# align cam1 to cam0 for chosen frame range
firstFrame = int(cam0FrameRate*30)
lastFrame = firstFrame*2 
cam0FramesToShow = np.arange(firstFrame,lastFrame+1).astype(int)
cam0TimesToShow = cam0FrameTimes[cam0FramesToShow]

cam1FramesToShow = np.where((cam1FrameTimes>=cam0TimesToShow[0]) & (cam1FrameTimes<=cam0TimesToShow[-1]))[0]
cam1TimesToShow = cam1FrameTimes[cam1FramesToShow]

alignedCam0Frames = cam0FramesToShow[:-1]
alignedCam1Frames = cam1FramesToShow[np.searchsorted(cam1TimesToShow,cam0TimesToShow)[:-1]]


# calculate merged frame shape
(h0,w0),(h1,w1) = [[int(cam.get(p)) for p in (cv2.CAP_PROP_FRAME_HEIGHT,cv2.CAP_PROP_FRAME_WIDTH)] for cam in (cam0,cam1)]
if h0>h1:
    offset0 = 0
    offset1 = int(0.5*(h0-h1))
else:
    offset0 = int(0.5*(h1-h0))
    offset1 = 0
gap = 2
mergedFrameShape = (h0+h1+gap,max(w0,w1))


# create merged video file
savePath = os.path.join(rootDir,'mergedVideo_frame'+str(firstFrame)+'-'+str(lastFrame)+'.avi')
v = cv2.VideoWriter(savePath,-1,cam0FrameRate,mergedFrameShape[::-1])
mergedFrame = np.zeros(mergedFrameShape,dtype=np.uint8)
for i in range(alignedCam0Frames.size):
    cam0.set(cv2.CAP_PROP_POS_FRAMES,alignedCam0Frames[i])
    mergedFrame[:h0,offset0:offset0+w0] = cv2.cvtColor(cam0.read()[1],cv2.COLOR_BGR2GRAY)
    cam1.set(cv2.CAP_PROP_POS_FRAMES,alignedCam1Frames[i])
    mergedFrame[h0+gap:,offset1:offset1+w1] = cv2.cvtColor(cam1.read()[1],cv2.COLOR_BGR2GRAY)
    v.write(mergedFrame)
v.release()


