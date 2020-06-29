import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, unary_from_softmax, \
    create_pairwise_gaussian

from Disparity_with_NN_prior_ColProb import Disparity_ColProb
import matplotlib.pyplot as plt


def Fusion(prob_ML, prob_disp, image):   # Make a call for this function and provide prob of ML, prob of disparity and original image

    alpha = 0.5  # Step Size for Col Prob
    gamma = 0.5  # Row Prob exponent
    beta = 4.0     # Col Prob Exponent for decay beyond the disparity pixels

    prob_disparity = Disparity_ColProb(alpha, gamma, beta, prob_disp, prob_ML, image)

    for i in range(256):
        for j in range(1024):
            if prob_disparity[i, j] > 1:
                prob_disparity[i, j] = 1
            if prob_disparity[i, j] == 0:
                prob_disparity[i, j] = 0.0001
            if prob_ML[i, j] == 0:
                prob_ML[i, j] = 0.0001

    rows, columns = prob_ML.shape
    d = dcrf.DenseCRF2D(rows, columns, 2)  # width, height, nlabels

    pml = np.empty((2, rows, columns), dtype='float32')
    pml[0, :, :] = prob_ML
    pml[1, :, :] = (1-prob_ML)

    Prob_ML = unary_from_softmax(pml)
    U1 = Prob_ML     # Potential for

    Prob_Disparity = np.empty((2, rows, columns), dtype='float32')
    Prob_Disparity[0, :, :] = 2*prob_disparity
    Prob_Disparity[1, :, :] = 1-prob_disparity

    U2 = np.empty((2, rows, columns), dtype='float32')
    U2 = -(np.log(Prob_Disparity))
    U2 = U2.reshape((2, -1))

    W = 1
    U = W*(U2 + U1)

    U = np.ascontiguousarray(U)
    d.setUnaryEnergy(U)

    feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                       img=image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                         kernel=dcrf.DIAG_KERNEL,
                         normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(5)

    res = np.argmax(Q, axis=0).reshape((prob_disparity.shape[0], prob_disparity.shape[1]))

    for i in range(256):
        for j in range(1024):
            if prob_ML[i, j] >= 0.5:
                prob_ML[i, j] = 1
            else:
                prob_ML[i,j] = 0

            if prob_disparity[i,j] >=0.5:
                prob_disparity[i,j] = 1
            else:
                prob_disparity[i,j] = 0


    f, (ax0, ax1) = plt.subplots(2, 2, sharex= 'all', sharey= 'all')
    ax0[0].imshow(image, cmap=plt.cm.gray)
    ax0[0].set_title("Original Image")
    ax0[1].imshow(image, cmap=plt.cm.gray)
    ax0[1].imshow(prob_ML, cmap=plt.cm.viridis, alpha=.5)
    ax0[1].set_title("Only Neural Network Prob")
    ax1[0].imshow(image, cmap=plt.cm.gray)
    ax1[0].imshow(prob_disparity, cmap=plt.cm.viridis, alpha=.5)
    ax1[0].set_title("vdisparity")
    ax1[1].imshow(image, cmap=plt.cm.gray)
    ax1[1].imshow(1-res, cmap=plt.cm.viridis, alpha=.5)
    ax1[1].set_title("Fusion")
    plt.show()

    return res

