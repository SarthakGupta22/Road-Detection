import numpy as np
import matplotlib.pyplot as plt
import cv2
from Method import Fuse

imName = 'umm_000075'  # Change the name of image
image = plt.imread('C:/Users/Asus/PycharmProjects/CRF/Original_Image/' + imName + '.png')
image = cv2.resize(image, (1024, 256))

prob_NN = np.load("C:/Users/Asus/PycharmProjects/CRF/R18T/R18T_Prob_" + imName + ".npy")

prob_DD = np.load("C:/Users/Asus/PycharmProjects/CRF/Prob_Map2/" + imName + ".npy")
prob_DD = cv2.resize(prob_DD, (1024, 256))

prob_FF = (prob_DD + prob_NN)/2

iterations = 50# keep it between 5 - 100

prob_FF = Fuse(image, prob_NN, prob_DD, prob_FF, iterations)





