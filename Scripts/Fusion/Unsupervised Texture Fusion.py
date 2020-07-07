#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter
from skimage.segmentation import slic
import matplotlib.pyplot as plt


# In[3]:


@njit
def get_distances(arr, P):
    distances = np.empty(arr.shape)
    for r in range(distances.shape[0]):
        for c in range(distances.shape[1]):
            distances[r, c] = np.sqrt(np.power(r - P[0], 2) + np.power(c - P[1], 2))
    
    return distances


# In[4]:


@njit
def texture_fusion(I, prob_nn, prob_dd, fused, segments):
    cx = prob_nn.shape[0] - 1
    cy = int(prob_nn.shape[1]/2) 
    distances = get_distances(prob_nn, (cx, cy))
    max_d = np.amax(distances) / 2
    for r in range(fused.shape[0]):
        for c in range(fused.shape[1]):
            if prob_nn[r, c] == prob_dd[r, c]:
                fused[r, c] = prob_nn[r, c]
            elif prob_nn[r, c] == 1:
                fused[r, c] = 0.75
            elif prob_dd[r, c] == 1 and segments[r, c] == segments[cx, cy]:
                fused[r, c] = (max_d - distances[r, c]) / max_d
                
    return fused


# In[5]:


# A possible sample function for unsupervised tuning of slic
# Generates unstable outputs right now
def find_optimal_segments(I, low, high):
    n_segments = 10
    segments = slic(I, n_segments, convert2lab=True, enforce_connectivity=True)
    if len(np.unique(segments)) < low:
        while len(np.unique(segments)) < low:
            n_segments += 1
            segments = slic(I, n_segments, convert2lab=True, enforce_connectivity=True)
    elif len(np.unique(segments)) > high:
        while len(np.unique(segments)) > high:
            n_segments -= 1
            segments = slic(I, n_segments, convert2lab=True, enforce_connectivity=True)
    
    return segments


# In[17]:


# Sample Code to try the fusion

I = np.random.random((100, 100, 3))
frm = np.random.random((100, 100))
D = np.random.random((100, 100))

segments = slic(I, 10, convert2lab=True, enforce_connectivity=True)
fused = np.zeros(frm.shape)

f_mask = texture_fusion(I, frm, D, fused, segments)
# Clip negative values 
f_mask[np.where(f_mask < 0)] = 0
# Gaussian smoothening
#f_mask = gaussian_filter(f_mask, 7)
# Get binary masks
f_mask = np.around(f_mask)


# In[19]:


plt.imshow(I)
plt.imshow(frm, alpha=0.3)


# In[ ]:




