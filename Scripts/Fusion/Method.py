import numpy as np
import time
import matplotlib.pyplot as plt
from Energy_Fn import energy_new


def get_distance(arr, P):
    distances = np.empty(arr.shape)
    for r in range(distances.shape[0]):
        for c in range(distances.shape[1]):
            distances[r, c] = np.sqrt(np.power(r - P[0], 2) + np.power(c - P[1], 2))

    return distances


def Fuse(image, prob_NN, prob_DD, prob_FF, iterations):
    start = time.time()

    rows, cols = image.shape[0:2]
    textureR = image[:, :, 0]/255
    textureG = image[:, :, 1]/255
    textureB = image[:, :, 2]/255

    # texture = (image[:, :, 0] + image[:, :, 1] + image[:, :, 2])/3

    UC = 1 - (np.abs(textureR[1: rows, :] - textureR[0: rows - 1, :]) +
              np.abs(textureG[1: rows, :] - textureG[0: rows - 1, :]) +
              np.abs(textureB[1: rows, :] - textureB[0: rows - 1, :])) / 3
    # UC = 1 - np.abs(texture[1: rows, :] - texture[0: rows - 1, :])
    UC = np.insert(UC, 0, np.zeros((cols)), axis=0)

    RC = 1 - (np.abs(textureR[:, 0: cols - 1] - textureR[:, 1: cols]) +
              np.abs(textureG[:, 0: cols - 1] - textureG[:, 1: cols]) +
              np.abs(textureB[:, 0: cols - 1] - textureB[:, 1: cols])) / 3

    # RC = 1 - np.abs(texture[:, 0: cols - 1] - texture[:, 1: cols])
    RC = np.append(RC, np.zeros((rows, 1)), axis=1)

    DC = 1 - (np.abs(textureR[0: rows - 1, :] - textureR[1: rows, :]) +
              np.abs(textureG[0: rows - 1, :] - textureG[1: rows, :]) +
              np.abs(textureB[0: rows - 1, :] - textureB[1: rows, :])) / 3
    # DC = 1 - np.abs(texture[0: rows - 1, :] - texture[1: rows, :])
    DC = np.append(DC, np.zeros((1, cols)), axis=0)

    LC = 1 - (np.abs(textureR[:, 1: cols] - textureR[:, 0: cols - 1]) +
              np.abs(textureG[:, 1: cols] - textureG[:, 0: cols - 1]) +
              np.abs(textureB[:, 1: cols] - textureB[:, 0: cols - 1])) / 3
    # LC = 1 - np.abs(texture[:, 1: cols] - texture[:, 0: cols - 1])
    LC = np.insert(LC, 0, np.zeros((rows)), axis=1)

    sum_weights = UC + DC + LC + RC

    M = np.average(np.arange(0, cols), weights=prob_NN[rows - 1])
    #M = np.average(np.arange(0, cols), weights=prob_DD[rows - 1])
    #print("Mean : ", M)

    baseline = np.array([255, round(M)])
    radius = round((rows / 5) * 2)

    distances = get_distance(prob_NN, baseline)
    d1 = np.where(distances < round(cols/4))
    d2 = np.where(distances >= round(cols/4))

    W_DD = 1
    W_NN = 1

    decay = (round(cols / 4) - distances) / round(cols / 4)
    decay = 1 - distances/cols
    # decay = (rows - distances) / (rows - radius)

    decay[np.where(decay < 0)] = 0

    sum_weights[d1] = sum_weights[d1] + W_DD * decay[d1]
    sum_weights[d2] = sum_weights[d2] + W_DD * 0.75
    #sum_weights = sum_weights + W_DD * 1
    sum_weights = sum_weights + W_NN

    # Energy minimization
    xaxis = np.arange(iterations)
    yaxis = np.empty([iterations])
    plotyaxis = np.empty([iterations])
    for i in range(0, iterations):
        #print(np.unique(prob_FF))
        energy = energy_new(prob_NN, prob_DD, prob_FF, W_NN, W_DD, RC, UC, LC, DC)
        yaxis[i] = energy
        prob_FF = np.append(np.multiply(prob_FF[:, 1: cols], RC[:, 0: cols - 1]), np.zeros((rows, 1)), axis=1) + \
                  np.insert(np.multiply(prob_FF[0: rows - 1, :], UC[1:rows, :]), 0, np.zeros((cols)), axis=0) + \
                  np.insert(np.multiply(prob_FF[:, 0:cols - 1], LC[:, 1:cols]), 0, np.zeros((rows)), axis=1) + \
                  np.append(np.multiply(prob_FF[1: rows, :], DC[0: rows - 1, :]), np.zeros((1, cols)), axis=0)

        # prob_FF[d1] = prob_FF[d1] + W_DD * prob_DD[d1]
        # prob_FF[d2] = prob_FF[d2] + W_DD * decay[d2] * prob_DD[d2]
        #prob_FF = prob_FF + W_DD * prob_DD * 1
        prob_FF = prob_FF + W_NN * prob_NN

        prob_FF[d1] = prob_FF[d1] +  W_DD * prob_DD[d1] * decay[d1]
        prob_FF[d2] = prob_FF[d2] + W_DD * prob_DD[d2] * 0.75

        prob_FF = np.divide(prob_FF, sum_weights)


    end = time.time()

    prob_FF[np.where(prob_FF > 0.5)] = 1
    prob_FF[np.where(prob_FF <= 0.5)] = 0


    return prob_FF
