# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 19:49:27 2019

@author: svc_ccg
"""
import getData
import os

#save multiple days to hdf5
exps = ['03122019_416656', '03212019_409096', '03262019_417882', '04042019_408528', '04102019_408527', '04252019_421323', '04302019_422856']
probesRecorded = ['ABCDEF', 'ABCD', 'ABCEF', 'ABCDEF', 'BCDEF', 'ABCDEF', 'ABCDEF']

failed = []
for exp, probes in zip(exps, probesRecorded):
    try:
        obj = getData.behaviorEphys('Z:\\' + exp, probes=probes)
        obj.loadFromRawData()
        saveFile = os.path.join('Z:\\' + exp, 'data_' + exp + '_noCCFRegistration.hdf5')
        obj.saveHDF5(saveFile)
    except:
        failed.append(exp)