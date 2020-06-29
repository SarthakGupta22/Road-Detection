import numpy as np
import matplotlib.pyplot as plt
import cv2
from CrfFusion import Fusion

imName = 'uu_000052'  # Change the name of image
image = plt.imread('Original_Image/' + imName + '.png')
image = cv2.resize(image, (1024, 256))

prob_ML = np.load("VGG_Prob/VGG_Prob_" + imName + ".npy")
vdisp = plt.imread("Disparity_Prob/V_Disparity/VD_" + imName + ".png")
vdisp = cv2.resize(vdisp, (1024, 256))

for i in range(256):
    for j in range(1024):

        if vdisp[i, j] > 0:
            vdisp[i, j] = 1
        else:
            vdisp[i, j] = 0

output = Fusion(prob_ML, vdisp, image)

