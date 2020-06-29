import numpy as np
import matplotlib.pyplot as plt


def GetInRange(Prob):
    for i in range (256):
        for j in range (1024):
            if Prob[i, j] > 1:
                Prob[i, j] = 1
    return Prob


def MinMax(disp):
    max = np.amax(disp)
    min = np.amin(disp)
    rows, columns = disp.shape
    for i in range(rows):
        for j in range(columns):
            disp[i, j] = (disp[i, j] - min)/(max - min)
    return disp


def ProbRow( prob_Vdisparity, vRow, disp_rows, disp_columns, mean, alpha, beta):

    prob_disparity = np.zeros((disp_rows, disp_columns), dtype='float')
    for i in range(vRow, disp_rows):
        start = mean[i]
        if start != 0:
            lower_index = 0
            higher_index = disp_columns - 1

            for j in range(start, -1, -1):
                if prob_Vdisparity[i, j] == 0:
                    lower_index = j

                    break
            for j in range(start + 1, disp_columns):
                if prob_Vdisparity[i, j] == 0:
                    higher_index = j
                    break

            if start == lower_index:
                step_size = 0
            else:
                step_size = alpha / (start - lower_index)

            for j in range(start, lower_index, -1):
                val = 0.99 - (step_size * (start - j))

                prob_disparity[i, j] = val * prob_Vdisparity[i, j]

            if start == higher_index:
                step_size = 0
            else:
                step_size = alpha / (higher_index - start)

            for j in range(start + 1, higher_index):
                val = 0.99 - (step_size * (j - start))
                prob_disparity[i, j] = val * prob_Vdisparity[i, j]

            for j in range(lower_index, -1, -1):
                if j < 0:
                    break
                val = pow(abs(j / lower_index), beta)
                prob_disparity[i, j] = val * prob_Vdisparity[i, j]

            for j in range(higher_index, disp_columns):
                if j > disp_columns - 1:
                    break
                val = pow( abs((1024 - j) / (1024 - higher_index)), beta)
                prob_disparity[i, j] = val * prob_Vdisparity[i, j]

    prob_disparity = GetInRange(prob_disparity)

    return prob_disparity


def Disparity_ColProb(alpha, gamma, beta, vdisp, prob_ML, image):

    rows, columns = vdisp.shape

    index = 0
    for i in range(rows):
        for j in range(columns):
            if vdisp[i, j] != 0:

                index = i
                break
        if index != 0:
            break

    vRow = index

    mean = np.zeros(256, dtype= 'int')

    # To calculate Weighted Mean

    for i in range (256):
        sum = 0
        count =0
        for j in range (1024):

            if vdisp[i, j] != 0:
                sum = sum + (j*vdisp[i, j])
                count = count + vdisp[i, j]

        if sum == 0:
            mean[i] = 0
        else:
           val = sum // count
           mean[i] = int(val)


    RowProb = np.zeros((256, 1024), dtype='float')
    ColProb = np.zeros((256, 1024), dtype='float')
    AvgProb = np.zeros((256, 1024), dtype='float')
    DispProb = np.zeros((256, 1024), dtype='float')

    for i in range(vRow,256):
        for j in range(0, 1024):

            RowProb[i, j] = pow(abs((i - vRow) /(256 - vRow)), gamma) * vdisp[i, j]
            AvgProb[i, j] = pow((vdisp[i, j] + prob_ML[i, j])/ 2, 1)

    ColProb = ProbRow(vdisp, vRow, 256, 1024, mean, alpha, beta)
    RowProb = GetInRange(RowProb)
    DispProb = (RowProb + ColProb)/2

    f, (ax0, ax1) = plt.subplots(2, 2, sharex= 'all', sharey= 'all')
    ax0[0].imshow(image, cmap=plt.cm.gray)
    ax0[0].set_title("Original Image")
    ax0[1].imshow(image, cmap=plt.cm.gray)
    ax0[1].imshow(RowProb, cmap=plt.cm.viridis, alpha=.5)
    ax0[1].set_title("Row Probability")
    ax1[0].imshow(image, cmap=plt.cm.gray)
    ax1[0].imshow(ColProb, cmap=plt.cm.viridis, alpha=.5)
    ax1[0].set_title("Column Probability")
    ax1[1].imshow(image, cmap=plt.cm.gray)
    ax1[1].imshow(DispProb, cmap=plt.cm.viridis, alpha=.5)
    ax1[1].set_title("Net prob")
    plt.show()

    return DispProb


