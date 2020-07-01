import numpy as np
import matplotlib.pyplot as plt
import cv2
from Method import Fuse

imName = 'uu_000099'  # Change the name of image
image = plt.imread('C:/Users/Asus/PycharmProjects/CRF/Original_Image/' + imName + '.png')
image = cv2.resize(image, (1024, 256))

prob_NN = np.load("C:/Users/Asus/PycharmProjects/CRF/VGG_Prob/VGG_Prob_" + imName + ".npy")

#prob_DD = plt.imread("C:/Users/Asus/PycharmProjects/CRF/Disparity_Prob_Tough/V_Disparity/VD_" + imName + ".png")
prob_DD = np.load("C:/Users/Asus/PycharmProjects/CRF/Latest3DProb/" + imName + ".npy")
prob_DD = cv2.resize(prob_DD, (1024, 256))

prob_FF = (2*prob_DD + prob_NN)/3

iterations = 5# keep it between 5 - 100
prob_FF = Fuse(image, prob_NN, prob_DD, prob_FF, iterations )

prob_FF[np.where(prob_FF> 0)] = 255

#from PIL import Image
#prob_FF = prob_FF.astype(np.uint8)
#im = Image.fromarray(prob_FF)
#im.save("C:/Users/Asus/PycharmProjects/CRF/Latest3DProb/uu_000099_fusion.png")