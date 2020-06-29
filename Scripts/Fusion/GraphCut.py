#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from PIL import Image
import maxflow
import matplotlib.pyplot as plt


# In[ ]:


# Load CNN Mask
N1 = np.load(' ')

# Load stereo mask
# Stereo mask is to be resized
N2_R = np.load(' ')

IN2 = Image.fromarray(N2_R)
IN2 = IN2.resize((1024, 256), Image.LINEAR)
N2 = np.array(IN2)


# In[ ]:


rows, cols = N1.shape
M = np.average(np.arange(0, cols), weights=N1[rows-1])
print("M : ", M)    
baseline = np.array([255, round(M)])
radius = round((rows / 5) * 2)
distances = get_distance(N1, baseline)


# In[ ]:


def get_distance(arr, P):
    distances = np.empty(arr.shape)
    for r in range(distances.shape[0]):
        for c in range(distances.shape[1]):
            distances[r, c] = np.sqrt(np.power(r - P[0], 2) + np.power(c - P[1], 2))
    
    return distances


# In[ ]:


def get_pixel_value(i, j, k, l):
    if distances[i, j] < radius:
        upot = (N1[i, j] + (2 * N2[i, j])) / 3
    else:
        decay = (rows - distances[i, j]) / (rows - radius)
        if decay < 0:
            decay = 0
        upot = (N1[i, j] + (2 * decay * N2[i, j])) / 3
    
    if distances[k, l] < radius:
        upot2 = (N1[k, l] + (2 * N2[k, l])) / 3
    else:
        decay = (rows - distances[k, l]) / (rows - radius)
        if decay < 0:
            decay = 0
        upot2 = (N1[k, l] + (2 * decay * N2[k, l])) / 3
        
    ppot = (upot - upot2) ** 2
    
    return ppot


# In[ ]:


# Create the graph.
g = maxflow.Graph[float]()

# Add the nodes. nodeids has the identifiers of the nodes in the grid.
nodeids = g.add_grid_nodes(N1.shape)


# In[ ]:


# Fill the edges
for i in range(rows - 1):
    for j in range(cols - 1):
        p1 = get_pixel_value(i, j, i+1, j)
        p2 = get_pixel_value(i, j, i, j+1)
        p3 = get_pixel_value(i, j, i+1, j+1)
        
        g.add_edge(rows*i + j,rows*(i + 1) + j, p1, p1)
        g.add_edge(rows*i + j,rows*i + j + 1, p2, p2)
        g.add_edge(rows*i + j,rows*(i + 1) + j + 1, p3, p3)


# In[ ]:


# Fill the terminal edges
g.add_grid_tedges(nodeids, 1 - N2, N2)


# In[ ]:


# Find the maximum flow.
g.maxflow()

# Get the segments of the nodes in the grid.
sgm = g.get_grid_segments(nodeids)


# In[ ]:


np.unique(sgm)


# In[ ]:


# Load the corresponding image
I = Image.open(' ')
I = I.resize((1024, 256), Image.ANTIALIAS)


# In[ ]:


mask = np.int_(np.logical_not(sgm))

plt.figure(figsize=(16, 8))
plt.axis('off')
plt.imshow(np.array(I))
plt.imshow(mask, alpha = 0.5)
plt.show()

