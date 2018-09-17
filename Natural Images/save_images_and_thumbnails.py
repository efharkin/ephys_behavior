# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 12:43:05 2018

@author: svc_ccg
"""
import fileIO
import pandas as pd
import cv2
import os
import numpy as np

imageDictPickleFile = fileIO.getFile()
imageDict = pd.read_pickle(imageDictPickleFile)

saveDir = fileIO.getDir()

downSampleFactor = 9
for image in imageDict:
    im = imageDict[image][image]
    im_thumb = cv2.resize(im, tuple(np.array(im.shape)[::-1]/downSampleFactor), interpolation=cv2.INTER_AREA)
    
    cv2.imwrite(os.path.join(saveDir, image + ".jpg"), im)
    cv2.imwrite(os.path.join(saveDir, image + "_thumbnail.jpg"), im_thumb)
    