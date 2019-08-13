# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 14:40:37 2019

@author: svc_ccg
"""

import numpy as np
import getData
import analysis_utils

b = getData.behaviorEphys(baseDir=r"Z:\05162019_423749", probes='ABCDEF')
b.loadFromHDF5(r"Z:\analysis\05162019_423749.hdf5")


#Specify source and target structures for ccg calculation
source = 'VISp'
target = 'VISam'

#Find the units in source and target
sourceProbes, sourceUnits = b.getUnitsByArea(source)
targetProbes, targetUnits = b.getUnitsByArea(target)

#Specify the time points to look for spikes during active and passive behavior
firstActiveTimePoint = b.frameAppearTimes[0]
lastActiveTimePoint = b.lastBehaviorTime
getActiveSpikes = lambda x: (x>firstActiveTimePoint)&(x<=lastActiveTimePoint)

firstPassiveTimePoint = b.passiveFrameAppearTimes[0]
lastPassiveTimePoint = b.passiveFrameAppearTimes[-1]
getPassiveSpikes = lambda x: (x>firstPassiveTimePoint)&(x<=lastPassiveTimePoint)


#Specify width/bins for ccg
width=0.5
bin_width = 0.001
n_b = int( np.ceil(width / bin_width) )  # Num. edges per side
bins = np.linspace(-width, width, 2 * n_b+1, endpoint=True)

#Compute CCGs between source and target neurons
active_ccgs = []
passive_ccgs = []

for iu, (sp, su) in enumerate(zip(sourceProbes, sourceUnits)):
    print(str(iu) + ' of ' + str(len(sourceUnits)))
    s_spikes = b.units[sp][su]['times'].flatten()
    
    a_unit_ccg = []
    p_unit_ccg = []
    for (tp, tu) in zip(targetProbes, targetUnits):
        
        #exclude autocorrelograms
        if (sp, su) == (tp, tu):
            continue
    
        
        t_spikes = b.units[tp][tu]['times'].flatten()
        
        #Exclude pairs if they don't have enough spikes during active or passive
        if (np.sum(getActiveSpikes(s_spikes))<1000) or (np.sum(getActiveSpikes(t_spikes))<1000):
            continue
        if (np.sum(getPassiveSpikes(s_spikes))<1000) or (np.sum(getPassiveSpikes(t_spikes))<1000):
            continue
        
        
        ad, ad_j = analysis_utils.get_ccg(s_spikes[getActiveSpikes(s_spikes)], t_spikes[getActiveSpikes(t_spikes)], 
                                                   width=width, bin_width=bin_width, num_jitter=10)
        pd, pd_j = analysis_utils.get_ccg(s_spikes[getPassiveSpikes(s_spikes)], t_spikes[getPassiveSpikes(t_spikes)], 
                                                   width=width, bin_width=bin_width, num_jitter=10)
        
#        [accg, hb] = np.histogram(ad, bins=bins, density=True)
#        [accg_j, hb] = np.histogram(ad_j, bins=bins, density=True)
#        
#        [pccg, hb] = np.histogram(pd, bins=bins, density=True)
#        [pccg_j, hb] = np.histogram(pd_j, bins=bins, density=True)
        
        [accg, hb] = np.histogram(ad, bins=bins)
        [accg_j, hb] = np.histogram(ad_j, bins=bins)
        
        a_geom_mean = (getActiveSpikes(s_spikes).sum()**0.5)*(getActiveSpikes(t_spikes).sum()**0.5)
        accg /= a_geom_mean
        accg_j /= a_geom_mean
        
        
        [pccg, hb] = np.histogram(pd, bins=bins)
        [pccg_j, hb] = np.histogram(pd_j, bins=bins)
        
        p_geom_mean = (getPassiveSpikes(s_spikes).sum()**0.5)*(getPassiveSpikes(t_spikes).sum()**0.5)
        pccg /= p_geom_mean
        pccg_j /= p_geom_mean
        
        
        a_unit_ccg.append(accg-accg_j)
        p_unit_ccg.append(pccg-pccg_j)
    
    active_ccgs.append(a_unit_ccg)
    passive_ccgs.append(p_unit_ccg)


a_cell_means = [np.mean(a, axis=0) for a in active_ccgs if len(a)>0]
p_cell_means = [np.mean(a, axis=0) for a in passive_ccgs if len(a)>0]


fig, ax = plt.subplots()
ax.plot(bins[:-1], np.nanmean(active_ccgs, axis=0))
ax.plot(bins[:-1], np.nanmean(passive_ccgs, axis=0))
ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1])
   
     
v1v1_accgs = np.copy(active_ccgs)
v1v1_pccgs = np.copy(passive_ccgs)  
fig, ax = plt.subplots() 
ax.plot(bins[:-1], np.nanmean(v1v1_accgs, axis=0))
ax.plot(bins[:-1], np.nanmean(v1v1_pccgs, axis=0))
ax.vlines(0, ax.get_ylim()[0], ax.get_ylim()[1])       
        
