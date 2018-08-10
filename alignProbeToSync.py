# -*- coding: utf-8 -*-
"""
Created on Wed Aug 08 13:08:37 2018

@author: svc_ccg
"""
from __future__ import division
from sync import Dataset
from visual_behavior.translator.core import create_extended_dataframe
from visual_behavior.translator.foraging2 import data_to_change_detection_core
from visual_behavior.validation.extended_trials import *
from visual_behavior.visualization.extended_trials.daily import make_daily_figure
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
from load_phy_template import load_phy_template

#First Stab at synchronizing, stealing a lot of code from the ecephys repo (https://github.com/AllenInstitute/ecephys_pipeline)

dataDir = fileIO.getDir()

sync_file = glob.glob(os.path.join(dataDir, '*.h5'))[0]
syncDataset = Dataset(sync_file)

def getSyncLineData(dataset, line_label=None, channel=None):
    if isinstance(line_label, str):
        try:
            channel = syncDataset.line_labels.index(line_label)
        except:
            print('Invalid line label')
            return
    elif channel is None:
        print('Must specify either line label or channel id')
        return
    
    sample_freq = syncDataset.meta_data['ni_daq']['counter_output_freq']
    rising = syncDataset.get_rising_edges(channel)/sample_freq
    falling = syncDataset.get_falling_edges(channel)/sample_freq
    
    return rising, falling

def extract_barcodes_from_times(on_times, off_times, inter_barcode_interval=10, 
                                bar_duration=0.03, barcode_duration_ceiling=2, 
                                nbits=32):
    '''Read barcodes from timestamped rising and falling edges.
    Parameters
    ----------
    on_times : numpy.ndarray
        Timestamps of rising edges on the barcode line
    off_times : numpy.ndarray
        Timestamps of falling edges on the barcode line
    inter_barcode_interval : numeric, optional
        Minimun duration of time between barcodes.
    bar_duration : numeric, optional
        A value slightly shorter than the expected duration of each bar
    barcode_duration_ceiling : numeric, optional 
        The maximum duration of a single barcode
    nbits : int, optional
        The bit-depth of each barcode
    Returns
    -------
    barcode_start_times : list of numeric
        For each detected barcode, the time at which that barcode started
    barcodes : list of int
        For each detected barcode, the value of that barcode as an integer.
    Notes
    -----
    ignores first code in prod (ok, but not intended)
    ignores first on pulse (intended - this is needed to identify that a barcode is starting)
    '''


    start_indices = np.diff(on_times)
    a = np.where(start_indices > inter_barcode_interval)[0]
    barcode_start_times = on_times[a+1]
    
    barcodes = []
    
    for i, t in enumerate(barcode_start_times):
        
        oncode = on_times[np.where(np.logical_and( on_times > t, on_times < t + barcode_duration_ceiling ))[0]]
        offcode = off_times[np.where(np.logical_and( off_times > t, off_times < t + barcode_duration_ceiling ))[0]]
        
        currTime = offcode[0]
        
        bits = np.zeros((nbits,))
        
        for bit in range(0, nbits):
            
            nextOn = np.where(oncode > currTime)[0]
            nextOff = np.where(offcode > currTime)[0]
            
            if nextOn.size > 0:
                nextOn = oncode[nextOn[0]]
            else:
                nextOn = t + inter_barcode_interval
            
            if nextOff.size > 0:
                nextOff = offcode[nextOff[0]]
            else:
                nextOff = t + inter_barcode_interval
            
            if nextOn < nextOff:
                bits[bit] = 1
            
            currTime += bar_duration
            
        barcode = 0        
        
        # least sig left
        for bit in range(0, nbits):
            barcode += bits[bit]*pow(2,bit)
        
        barcodes.append(barcode)
                    
    return barcode_start_times, barcodes

def match_barcodes(master_times, master_barcodes, probe_times, probe_barcodes):
    '''Given sequences of barcode values and (local) times on a probe line and a master 
    line, find the time points on each clock corresponding to the first and last shared 
    barcode.
    Parameters
    ----------
    master_times : np.ndarray
        start times of barcodes (according to the master clock) on the master line. 
        One per barcode.
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_times : np.ndarray
        start times (according to the probe clock) of barcodes on the probe line. 
        One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe_line. One per barcode
    Returns
    -------
    probe_interval : np.ndarray
        Start and end times of the matched interval according to the probe_clock.
    master_interval : np.ndarray
        Start and end times of the matched interval according to the master clock
    '''

    if abs( len(probe_barcodes) - len(master_barcodes) ) < 3:

        if probe_barcodes[0] == master_barcodes[0]:
            t_p_start = probe_times[0]
            t_m_start = master_times[0]
        else:
            t_p_start = probe_times[2]
            t_m_start = master_times[np.where(master_barcodes == probe_barcodes[2])]

        if probe_barcodes[-1] == master_barcodes[-1]:
            t_p_end = probe_times[-1]
            t_m_end = master_times[-1]
        else:
            t_p_end = probe_times[-2]
            t_m_end = master_times[np.where(master_barcodes == probe_barcodes[-2])]

    else:

        for idx, item in enumerate(master_barcodes):

            if item == probe_barcodes[0]:
                print('probe dropped initial barcodes. Start from ' + str(idx))
                t_p_start = probe_times[0]
                t_m_start = master_times[idx]
                
                if probe_barcodes[-1] == master_barcodes[-1]:
                    t_p_end = probe_times[-1]
                    t_m_end = master_times[-1]
                else:
                    t_p_end = probe_times[-2]
                    t_m_end = master_times[np.where(master_barcodes == probe_barcodes[-2])]

                break

    return np.array([t_p_start, t_p_end]), np.array([t_m_start, t_m_end])


def linear_transform_from_intervals(master, probe):
    '''Find a scale and translation which aligns two 1d segments
    Parameters
    ----------
    master : iterable
        Pair of floats defining the master interval. Order is [start, end].
    probe : iterable
        Pair of floats defining the probe interval. Order is [start, end].
    Returns
    -------
    scale : float
        Scale factor. If > 1.0, the probe clock is running fast compared to the 
        master clock. If < 1.0, the probe clock is running slow.
    translation : float
        If > 0, the probe clock started before the master clock. If > 0, after.
    Notes
    -----
    solves 
        (master + translation) * scale = probe
    for scale and translation
    '''

    scale = (probe[1] - probe[0]) / (master[1] - master[0])
    translation = probe[0] / scale - master[0]

    return scale, translation
    

def get_probe_time_offset(master_times, master_barcodes, 
                          probe_times, probe_barcodes, 
                          acq_start_index, local_probe_rate):
    """Time offset between master clock and recording probes. For converting probe time to master clock.
    
    Parameters
    ----------
    master_times : np.ndarray
        start times of barcodes (according to the master clock) on the master line. 
        One per barcode.
    master_barcodes : np.ndarray
        barcode values on the master line. One per barcode
    probe_times : np.ndarray
        start times (according to the probe clock) of barcodes on the probe line. 
        One per barcode
    probe_barcodes : np.ndarray
        barcode values on the probe_line. One per barcode
    acq_start_index : int
        sample index of probe acquisition start time
    local_probe_rate : float
        the probe's apparent sampling rate
    
    Returns
    -------
    total_time_shift : float
        Time at which the probe started acquisition, assessed on 
        the master clock. If < 0, the probe started earlier than the master line.
    probe_rate : float
        The probe's sampling rate, assessed on the master clock
    master_endpoints : iterable
        Defines the start and end times of the sync interval on the master clock
    
    """

    probe_endpoints, master_endpoints = match_barcodes(master_times, master_barcodes, probe_times, probe_barcodes)
    rate_scale, time_offset = linear_transform_from_intervals(master_endpoints, probe_endpoints)

    probe_rate = local_probe_rate * rate_scale
    acq_start_time = acq_start_index / probe_rate

    total_time_shift = time_offset - acq_start_time

    return total_time_shift, probe_rate, master_endpoints



frameRising, frameFalling = getSyncLineData(syncDataset, 'stim_vsync')

#Get frame times from pkl behavior file
pkl_file = glob.glob(os.path.join(dataDir, '*.pkl'))[0]
behaviordata = pd.read_pickle(pkl_file)
core_data = data_to_change_detection_core(behaviordata)
trials = create_extended_dataframe(
    trials=core_data['trials'],
    metadata=core_data['metadata'],
    licks=core_data['licks'],
    time=core_data['time'],
)

#Get barcodes from sync file
bRising, bFalling = getSyncLineData(syncDataset, 'barcode')
bs_t, bs = extract_barcodes_from_times(bRising, bFalling)

#Get barcodes from ephys data
channel_states = np.load(os.path.join(glob.glob(os.path.join(dataDir, '*sorted'))[0], 'events\\Neuropix-3a-100.0\\TTL_1\\channel_states.npy'))
event_times = np.load(os.path.join(glob.glob(os.path.join(dataDir, '*sorted'))[0], 'events\\Neuropix-3a-100.0\\TTL_1\\event_timestamps.npy'))

beRising = event_times[channel_states>0]/30000.
beFalling = event_times[channel_states<0]/30000.
be_t, be = extract_barcodes_from_times(beRising, beFalling)

#Compute time shift between ephys and sync
shift, p_sampleRate, m_endpoints = get_probe_time_offset(bs_t, bs, be_t, be, 0, 30000)
be_t_shifted = (be_t/(p_sampleRate/30000)) - shift #just to check that the shift and scale are right


#Get unit spike times 
spike_data_dir = os.path.join(glob.glob(os.path.join(dataDir, '*sorted'))[0], 'continuous\\Neuropix-3a-100.0')
spike_clusters = np.load(os.path.join(spike_data_dir, 'spike_clusters.npy'))
spike_times = np.load(os.path.join(spike_data_dir, 'spike_times.npy'))
cluster_ids = pd.read_csv(os.path.join(spike_data_dir, 'cluster_groups.csv'), sep='\t')
templates = np.load(os.path.join(spike_data_dir, 'templates.npy'))
spike_templates = np.load(os.path.join(spike_data_dir, 'spike_templates.npy'))
unit_ids = np.unique(spike_clusters)

units = {}
for u in unit_ids:
    units[u] = {}
    units[u]['label'] = cluster_ids[cluster_ids['cluster_id']==u]['group'].tolist()[0]
    
    unit_idx = np.where(spike_clusters==u)[0]
    unit_sp_times = spike_times[unit_idx]/p_sampleRate - shift
    
    units[u]['times'] = unit_sp_times
    
    #choose 1000 spikes with replacement, then average their templates together
    chosen_spikes = np.random.choice(unit_idx, 1000)
    chosen_templates = spike_templates[chosen_spikes].flatten()
    units[u]['template'] = np.mean(templates[chosen_templates], axis=0)
    units[u]['peakChan'] = np.unravel_index(np.argmin(units[u]['template']), units[u]['template'].shape)[1]


#align trials to clock
trial_start_frames = np.array(trials['startframe'])
trial_end_frames = np.array(trials['endframe'])
trial_start_times = frameRising[trial_start_frames]
trial_end_times = frameFalling[trial_end_frames]
trial_ori = np.array(trials['initial_ori'])

change_frames = np.array(trials['change_frame'])
change_times = frameRising[change_frames]
change_ori = np.array(trials['change_ori'])

#make psth for units

def makePSTH(spike_times, trial_start_times, trial_duration, bin_size = 0.1):
    counts = np.zeros(round(trial_duration/bin_size))    
    for ts in trial_start_times:
        for ib, b in enumerate(np.arange(ts, ts+trial_duration, bin_size)):
            c = np.sum((spike_times>=b) & (spike_times<b+bin_size))
            if ib<len(counts): #imprecision of floats means ib sometimes exceeds length (np.arange quirk)
                counts[ib] += c
    return counts/len(trial_start_times)


goodClusters = cluster_ids[cluster_ids['group'] == 'good'].cluster_id.tolist()
for u in goodClusters:
    spikes = units[u]['times']
    
    psthVert = makePSTH(spikes, change_times[np.logical_or(change_ori==90, change_ori==270)]-2, 12)
    psthHorz = makePSTH(spikes, change_times[np.logical_or(change_ori==0, change_ori==180)]-2, 12)
    fig, ax = plt.subplots(1, 2)
    ax[0].plot(psthVert)
    ax[1].plot(psthHorz)
    
    
        











