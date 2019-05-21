# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:12:34 2019

@author: svc_ccg
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
vimg = cv2.imread(r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_760937126\isi_experiment_766321187\766321187_vasculature.tif')
pimg = cv2.imread(r'Z:\03122019_416656\2019_03_12_15_35_40_right.png')

class pointAnnotator:
    def __init__(self, im, ax):
        self.ax = ax
        self.ax.set_xlim([0,im.get_array().shape[1]])
        self.ax.set_ylim([im.get_array().shape[0],0])
        
        self.im = im
        self.xs = []
        self.ys = []
        self.annos = []
        self.cid = im.figure.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        if event.button == 1:
            print('x ' + str(event.xdata) + '\ty ' + str(event.ydata))
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            
            anno, = self.ax.plot(event.xdata, event.ydata, 'ro')
            self.annos.append(anno)
            self.ax.text(event.xdata+0.2, event.ydata+0.2, str(len(self.annos)))
            self.im.figure.canvas.draw()  
        else:# event.button==2:
#            print(event.button)
            self.resetAnnotations()
    
    def resetAnnotations(self):
        self.xs=[]
        self.ys=[]
        for a in self.annos:
            a.remove()

        self.im.figure.canvas.update()
        self.im.figure.canvas.draw()
        self.annos=[]


vfig, vax = plt.subplots()
vim = vax.imshow(vimg, origin='upper')
vpa = pointAnnotator(vim, vax)

pfig, pax = plt.subplots()
pim = pax.imshow(pimg, origin='upper')
ppa = pointAnnotator(pim, pax)

ppoints = np.stack((ppa.xs, ppa.ys)).astype(np.float32).T
vpoints = np.stack((vpa.xs, vpa.ys)).astype(np.float32).T

#aff = cv2.getAffineTransform(ppoints[:3], vpoints[:3])
aff = cv2.estimateRigidTransform(ppoints, vpoints, True)
pimg_aff = cv2.warpAffine(pimg, aff, (vimg.shape[1], vimg.shape[0]))

paffFig, paffax = plt.subplots()
paff_im = paffax.imshow(pimg_aff, origin='upper')
paffp = pointAnnotator(paff_im, paffax)

paffpoints = np.stack((paffp.xs, paffp.ys)).astype(np.float32).T
fig, ax = plt.subplots()
ax.imshow(vimg, origin='upper')
for p in paffpoints:
    ax.plot(p[0], p[1], 'ro')

