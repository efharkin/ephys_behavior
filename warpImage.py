# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:34:07 2019

@author: svc_ccg
"""

import numpy as np
import cv2

def warpImage(refImg,warpImg,refPts,warpPts):
    # append boundaryPts to npts x 2 (x,y) float32 point arrays
    refPts,warpPts = (np.concatenate((pts,getBoundaryPoints(shape[1],shape[0])),axis=0).astype(np.float32) for pts,shape in zip((refPts,warpPts),(refImg.shape,warpImg.shape)))
    
    # get Delaunay triangles as indices of refPts (point1Index,point2Index,point3Index)
    triangles = getDelauneyTriangles(refPts,refImg.shape[1],refImg.shape[0])
    triPtInd = np.zeros((triangles.shape[0],3),dtype=int)
    for i,tri in enumerate(triangles):
        for j in (0,2,4):
            triPtInd[i,j//2] = np.where(np.all(refPts==tri[j:j+2],axis=1))[0][0]
    
    # warp each triangle
    newImg = np.zeros_like(refImg)
    for tri in triPtInd:
        refTri = refPts[tri]
        warpTri = warpPts[tri]
        refRect = cv2.boundingRect(refTri)
        warpRect = cv2.boundingRect(warpTri)
        refTri -= refRect[:2]
        warpTri -= warpRect[:2]
        refSlice,warpSlice = ((slice(r[1],r[1]+r[3]),slice(r[0],r[0]+r[2])) for r in (refRect,warpRect))
        mask = np.zeros(refRect[:-3:-1],dtype=np.uint8)
        cv2.fillConvexPoly(mask,refTri.astype(int),1)
        mask = mask.astype(bool)
        warpMatrix = cv2.getAffineTransform(warpTri,refTri)
        for ch in range(newImg.shape[2]):
            warpData = cv2.warpAffine(warpImg[warpSlice[0],warpSlice[1],ch],warpMatrix,refRect[2:],flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
            newImg[refSlice[0],refSlice[1],ch][mask] = warpData[mask]
    return newImg

def getBoundaryPoints(w,h):
    return [(0,0),(w/2,0),(w-1,0),(w-1,h/2),(w-1,h-1),(w/2,h-1),(0,h-1),(0,h/2)]
    
def getDelauneyTriangles(pts,w,h):
    subdiv = cv2.Subdiv2D((0,0,w,h))
    for p in pts:
        subdiv.insert((p[0],p[1]))
    triangles = subdiv.getTriangleList()
    return triangles[np.all(triangles>=0,axis=1) & np.all(triangles[:,::2]<w,axis=1) & np.all(triangles[:,1::2]<h,axis=1)]