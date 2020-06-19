from MLmodels import RPNet
from StereoVison import *
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


# ---------------- H E L P E R   F U N C T I O N S   D O   N O T   E D I T ------------------------

def get_mask_name(x):
    str_list = list(x)
    str_list.insert(-10, 'road_')
    out = ''.join(str_list)
    return out


# ----------------- M A I N   P R O G R A M   B E G I N S ------------------------

model_path = 'D:/Award Winning Network/RP-Net.h5'

left_image = Image.open('D:/KITTI_Road/Left/training/image_2/umm_000063.png')
right_image = Image.open('D:/KITTI_Road/Right/training/image_3/um_000063.png')
left_image = np.array(left_image.resize((1024, 256), Image.ANTIALIAS))
right_image = np.array(right_image.resize((1024, 256), Image.ANTIALIAS))

LI = np.reshape(left_image, (1, 256, 1024, 3))

model = RPNet()
model.load(model_path)
rpnet_pred = model.predict(LI)
rpnet_pred = np.array(rpnet_pred[0, :, :])

cv2.imshow('Nalla', rpnet_pred)
cv2.waitKey(0)

disparity = find_disparity(left_image, right_image)

print("RP Mask", np.unique(rpnet_pred))

v_disp = Vdisparity(left_image, right_image, rpnet_pred)
v_disp.compute(disparity)

# Plot V-Disparity
v_disp.plot_vdisp_line()

v_disp.fit_line()
# View RPNet + V-Disparity Prediction
v_disp.show_road_mask()

plt.imshow(v_disp.line_mask)
plt.show()

