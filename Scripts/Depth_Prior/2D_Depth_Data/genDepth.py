import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Read Disparity Map
disp = cv2.imread('D_uu_000099.png',0)
print(disp.shape)

# 2D numpy array storing height and depth of each pixel
DH = np.zeros([disp.shape[0], disp.shape[1], 2], dtype=np.float32)

for i in range(0, disp.shape[0]):
    for j in range(0, disp.shape[1]):
        point = []
        z = float(0.54*721.0/(disp[i][j] + 0.1))
        if z < 100.0:
            x = float((j - disp.shape[1]/2.0)*z/721.0)
            y = float((i - disp.shape[0]/2.0)*z/721.0)
            DH[i,j,0] = z
            DH[i,j,1] = y


maxH = np.max(DH[:,:,1])
minH = np.min(DH[:,:,1])

DH[:,:,1] = (DH[:,:,1] - minH)/(maxH - minH)

print(np.max(DH[:,:,1]))

plt.imshow(DH[:,:,1]*255.0)
plt.show()
cv2.imshow("Depth image", disp)
cv2.waitKey(0)
print(DH)
