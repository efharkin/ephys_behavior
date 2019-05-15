# -*- coding: utf-8 -*-
"""
Created on Tue May 14 18:12:34 2019

@author: svc_ccg
"""

import cv2
import numpy as np
img = cv2.imread(r'\\allen\programs\braintv\production\visualbehavior\prod0\specimen_760937126\isi_experiment_766321187\766321187_vasculature.tif')
type(img)
img.shape
vimg = img
pimg = cv2.imread(r'Z:\03122019_416656\2019_03_12_15_35_40_right.png')

plt.imshow(pimg)
ppoints = [[497.7, 271.2], [703.0, 463.9], [455.9, 468.3]]
plt.figure()
plt.imshow(vimg)
vpoints = [[1265.3, 85.5], [1669.7, 1264.6], [672.8, 692.2]]


ppoints = np.array(ppoints).astype(np.float32)
vpoints = np.array(vpoints).astype(np.float32)

aff = cv2.getAffineTransform(ppoints, vpoints)
pimg_aff = cv2.warpAffine(pimg, aff, (vimg.shape[1], vimg.shape[0]))



fig, ax = plt.subplots()
ax.imshow(pimg_aff)

def onclick(event):
#    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
#          ('double' if event.dblclick else 'single', event.button,
#           event.x, event.y, event.xdata, event.ydata))
#           
   event.canvas.draw_cursor()
#    ax.plot(event.xdata, event.ydata, 'ro')
#    event.canvas.draw()


cid = fig.canvas.mpl_connect('button_press_event', onclick)