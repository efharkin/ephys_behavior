# -*- coding: utf-8 -*-

"""

import getPickleFiles
mouseID = '423745'
limsDir = r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_784188286'
getPickleFiles.getPickleFiles(mouseID,limsDir)

"""

import os
import glob
import shutil


def getPickleFiles(mouseID,src=r'C:\ProgramData\camstim\output',dst=r'\\EphysRoom342\Data\behavior pickle files'):
    
    newDir = os.path.join(dst,mouseID)
    if not os.path.exists(newDir):
        os.mkdir(newDir)
     
    for f in glob.glob(os.path.join(src,'*_'+mouseID+'_*.pkl')):
        shutil.copy2(f,os.path.join(newDir,os.path.basename(f)))
     
    for i in os.listdir(src):
        ipath = os.path.join(src,i)
        if os.path.isdir(ipath):
            getPickleFiles(mouseID,ipath,dst)
