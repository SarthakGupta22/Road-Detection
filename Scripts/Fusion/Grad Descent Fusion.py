#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.ndimage as ndimage
from PIL import Image


# In[3]:


from matplotlib import pyplot as plt
from tqdm.auto import tqdm


# In[140]:


# Energy function to be optimized
def energy_fn(labels, mask1, mask2, deriv=False, decay=0.5, alpha=0.5, beta=0.5):
    if not deriv:
        upot = np.power((alpha*mask1 + beta*mask2 - labels), 2)
        avg_filt = np.array([[1,1,1], [1,-8,1], [1,1,1]])
        ppot = np.power(ndimage.convolve(labels, avg_filt, mode='nearest') / 8, 2)
        # print('Here')
        return decay*upot + (1 - decay)*ppot
    
    d_upot = -2 * (alpha*mask1 + beta*mask2 - labels)
    avg_filt = np.array([[1,1,1], [1,-8,1], [1,1,1]])
    d_ppot = -2 * (ndimage.convolve(labels, avg_filt, mode='nearest') / 8)
    
    return decay*d_upot + (1 - decay)*d_ppot


# In[141]:


# Mask of Stereo based methods
sm = np.load('D:/KITTI_Road/Probabilities/Stereo Mask/uu_000090.npy')


# In[151]:


# Mask of Resnet
ri = Image.open('D:/KITTI_Road/Probabilities/R18T/R18T_mask_uu_000090.png')
ri = ri.resize((1226, 370), Image.NEAREST)
rm = np.array(ri, dtype=np.float64)


# In[152]:


rm.shape


# In[153]:


sm.shape


# In[154]:


# Initialize all labels to 0.5
labels = 0.5 * np.ones(rm.shape)


# In[157]:


energy_fn(labels, rm, sm, True)


# In[162]:


lr = 0.1
decay = 0.5


# In[163]:


energies = []
for it in tqdm(range(1000)):
    labels = labels - (energy_fn(labels, rm, sm, True, decay) * lr)
    energies.append(np.sum(energy_fn(labels, rm, sm, False, decay)))
    if it % 200 == 0:
        lr = lr / 10
        decay += 0.05


# In[164]:


plt.plot(energies)


# In[166]:


labels[np.where(labels > 1)] = 1


# In[167]:


plt.hist(labels.flatten())
plt.show()


# In[172]:


I = Image.open('D:/KITTI_Road/Left/training/image_2/uu_000090.png')
NI = np.array(I)


# In[173]:


plt.imshow(NI)
plt.imshow(labels, alpha = 0.5)


# In[169]:


plt.imshow(np.around(labels))


# In[170]:


plt.imshow(rm)


# In[171]:


plt.imshow(sm)

