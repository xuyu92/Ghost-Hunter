import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py

for i in range(26):
    fin = h5py.File('/mnt/stage/K+/%d/wave.h5'%(i+1), 'r')
    Waveform = fin['Waveform']
    print(i, len(Waveform))
    fin.close()

nevt = 17613*20
waveform = np.zeros([nevt, 1000])
hit = np.zeros([nevt, 1000])

for i in range(20):
    print(i+1)
    fin = h5py.File('/mnt/stage/K+/%d/wave.h5'%(i+1), 'r')
    Waveform = fin['Waveform'][:17613]
    GroundTruth = fin['GroundTruth'][:]
    channel_start = i*17613
    hit_idx = 0
    for channel in range(17613):
        baseline = Waveform[channel][2][:100].mean()
        w_channel = channel_start + channel
        waveform[w_channel] = Waveform[channel][2] - baseline
        hit_channel = GroundTruth[hit_idx][1]
        while channel==hit_channel:
            hit_time = int(GroundTruth[hit_idx][2])
            hit[w_channel][hit_time] += 1
            hit_idx += 1
            if hit_idx==len(GroundTruth):
                break
            hit_channel = GroundTruth[hit_idx][1]
        if hit[w_channel][-50:].sum()>0:
            waveform[w_channel][-50:] = waveform[w_channel][:50]
            hit[w_channel][-50:] = hit[w_channel][:50]
    fin.close()

fsave = h5py.File('wave.h5','w')
fsave['waveform'] = waveform
fsave['hit'] = hit
fsave.close()
